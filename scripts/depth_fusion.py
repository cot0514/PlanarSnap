"""
Fuse the rendered depth maps of a trained BBSplat model into a dense oriented
point cloud, for plane segmentation.

Why this exists: MCMC allocates billboards where the image has texture, so the
flattest surfaces -- exactly the planes we want -- carry the fewest primitives
(flat 8x8 patches average 0.8 billboards and 72% of them are empty).  A rendered
depth map has no such bias: every covered pixel yields one 3D point, so a plain
wall comes out as dense as a patterned rug.  Normals come from
`depth_to_normal`, i.e. the same quantity the `lambda_normal` training loss is
supervised against, and are already in world space.

  python scripts/depth_fusion.py -m output/drjohnson_v1 -s ../data/drjohnson \
      --voxel 0.02 --cam_stride 1 --pixel_stride 2
"""
import os, sys, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


OFF = 1 << 17          # half-width of the packed voxel grid, in voxels


def consolidate(chunks):
    """Merge per-camera (code, possum, nrmsum, cnt) chunks, summing duplicates."""
    code = np.concatenate([c[0] for c in chunks])
    pos = np.concatenate([c[1] for c in chunks])
    nrm = np.concatenate([c[2] for c in chunks])
    cnt = np.concatenate([c[3] for c in chunks])
    uniq, inv = np.unique(code, return_inverse=True)
    P = np.zeros((len(uniq), 3), np.float64)
    N = np.zeros((len(uniq), 3), np.float64)
    C = np.zeros(len(uniq), np.int64)
    np.add.at(P, inv, pos)
    np.add.at(N, inv, nrm)
    np.add.at(C, inv, cnt)
    return [(uniq, P, N, C)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-s", "--source", default=None)
    ap.add_argument("--iteration", type=int, default=32000)
    ap.add_argument("--voxel", type=float, default=0.02, help="fusion voxel size, metres")
    ap.add_argument("--cam_stride", type=int, default=1)
    ap.add_argument("--pixel_stride", type=int, default=2)
    ap.add_argument("--alpha_thresh", type=float, default=0.5,
                    help="skip pixels the billboards do not confidently cover")
    ap.add_argument("--depth_min", type=float, default=0.1)
    ap.add_argument("--depth_max", type=float, default=-1.0,
                    help="metres; <0 means 2x the scene radius")
    ap.add_argument("--edge_rel", type=float, default=0.02,
                    help="reject a pixel whose 4-neighbour depth jump exceeds "
                         "this fraction of its own depth (flying pixels at "
                         "silhouettes produce meaningless normals)")
    ap.add_argument("--clip_billboards", type=float, default=-1.0,
                    help="metres of margin; keep only fused points inside the "
                         "robust bounding box of the model's own billboards. "
                         "Rendered median depth sometimes lands on floaters "
                         "outside the room (11%% of room0's cloud), which "
                         "inflates the scene extent and the auto plane-fit "
                         "threshold derived from it")
    ap.add_argument("--radius_clip", type=float, default=-1.0,
                    help="drop fused points farther than this from the camera "
                         "centroid; excludes the sky-box dome and far-field haze")
    ap.add_argument("--min_samples", "--min_views", dest="min_samples", type=int,
                    default=2,
                    help="drop voxels hit by fewer than this many pixels. Note "
                         "these are pixels, not distinct cameras: at a 2 cm "
                         "voxel a single view already contributes several.")
    ap.add_argument("--depth_ratio", type=float, default=1.0,
                    help="1 = median depth (bounded/indoor), 0 = expected depth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test_cams", action="store_true",
                    help="fuse the held-out cameras too")
    a = ap.parse_args()

    from argparse import ArgumentParser as AP2
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel, render
    from scene import Scene
    from utils.point_utils import depths_to_points

    argv = ["x", "-m", a.model, "--iteration", str(a.iteration)]
    if a.source:
        argv += ["-s", a.source]
    saved_argv, sys.argv = sys.argv, argv
    p2 = AP2(); mp = ModelParams(p2, sentinel=True); pp = PipelineParams(p2)
    p2.add_argument("--iteration", default=-1, type=int)
    args2 = get_combined_args(p2)
    sys.argv = saved_argv
    dataset, pipe = mp.extract(args2), pp.extract(args2)
    pipe.depth_ratio = a.depth_ratio

    gaussians = GaussianModel(dataset.sh_degree, texture_preproc=True)
    scene = Scene(dataset, gaussians, load_iteration=a.iteration, shuffle=False)
    bg = torch.zeros(3, device="cuda")

    cams = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    if a.test_cams:
        cams = cams + sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    cams = cams[::a.cam_stride]
    print(f"{len(cams)} cameras, voxel {a.voxel} m, pixel stride {a.pixel_stride}")

    # Scene scale comes from the cameras, not the billboards: `--add_sky_box`
    # puts a dome of primitives at ~90 m, which would drag any percentile of the
    # billboard radius up with it and make the depth cut meaningless.
    cc_all = torch.stack([c.camera_center for c in cams])
    cam_centre = cc_all.mean(0)
    Rc = torch.quantile((cc_all - cam_centre).norm(dim=1), 0.95).item()
    dmax = a.depth_max if a.depth_max > 0 else max(10.0 * Rc, 8.0)
    print(f"camera spread (p95) {Rc:.2f} m -> depth cut {dmax:.2f} m"
          + ("" if a.depth_max > 0 else "  (override with --depth_max)"))

    chunks, stats = [], dict(px=0, kept=0, lowalpha=0, edge=0, range=0)
    ps = a.pixel_stride
    with torch.no_grad():
        for i, cam in enumerate(cams):
            pkg = render(cam, gaussians, pipe, bg)
            dep = pkg["surf_depth"][0]
            al = pkg["rend_alpha"][0]
            nrm = pkg["surf_normal"]
            nrm = nrm / (nrm.norm(dim=0, keepdim=True) + 1e-8)
            pts = depths_to_points(cam, dep).reshape(*dep.shape, 3)

            # depth discontinuity: compare each pixel against its 4 neighbours
            d = dep
            jump = torch.zeros_like(d)
            jump[1:-1, 1:-1] = torch.maximum(
                (d[2:, 1:-1] - d[1:-1, 1:-1]).abs(),
                torch.maximum((d[:-2, 1:-1] - d[1:-1, 1:-1]).abs(),
                              torch.maximum((d[1:-1, 2:] - d[1:-1, 1:-1]).abs(),
                                            (d[1:-1, :-2] - d[1:-1, 1:-1]).abs())))
            m_alpha = al > a.alpha_thresh
            m_range = (d > a.depth_min) & (d < dmax)
            m_edge = jump < (a.edge_rel * d)
            m_edge[0, :] = m_edge[-1, :] = m_edge[:, 0] = m_edge[:, -1] = False
            keep = m_alpha & m_range & m_edge
            keep = keep[::ps, ::ps]

            stats["px"] += keep.numel()
            stats["lowalpha"] += int((~m_alpha[::ps, ::ps]).sum())
            stats["range"] += int((~m_range[::ps, ::ps]).sum())
            stats["edge"] += int((~m_edge[::ps, ::ps]).sum())
            stats["kept"] += int(keep.sum())
            if not keep.any():
                continue

            P = pts[::ps, ::ps][keep]
            Nn = nrm.permute(1, 2, 0)[::ps, ::ps][keep]
            # orient toward the camera so that averaging across views is valid
            cc = cam.camera_center
            flip = ((cc - P) * Nn).sum(1) < 0
            Nn = torch.where(flip[:, None], -Nn, Nn)

            # Global voxel key, packed into one int64 so that voxels can be
            # merged across cameras.  OFF bounds the grid at +-2621 m for a 2 cm
            # voxel and keeps the packed value under 2^54.
            gi = torch.floor(P / a.voxel).to(torch.int64)
            if int(gi.abs().max()) >= OFF:
                raise SystemExit(f"voxel index {int(gi.abs().max())} exceeds the "
                                 f"packing range {OFF}; raise --voxel or clip the scene")
            code = ((gi[:, 0] + OFF) * (2 * OFF) + (gi[:, 1] + OFF)) * (2 * OFF) + (gi[:, 2] + OFF)

            cpu_code = code.cpu().numpy()
            u, inv = np.unique(cpu_code, return_inverse=True)
            Pc, Nc = P.cpu().numpy().astype(np.float64), Nn.cpu().numpy().astype(np.float64)
            Ps = np.zeros((len(u), 3)); Ns = np.zeros((len(u), 3)); Cs = np.zeros(len(u), np.int64)
            np.add.at(Ps, inv, Pc); np.add.at(Ns, inv, Nc)
            np.add.at(Cs, inv, 1)
            chunks.append((u, Ps, Ns, Cs))
            if len(chunks) >= 25:
                chunks = consolidate(chunks)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(cams)} cams, {len(chunks[0][0]):,} voxels so far")

    if not chunks:
        # Every pixel of every view was rejected.  The usual cause is a model
        # whose texture alpha collapsed during training, so nothing renders:
        # check sigmoid(texture_alpha) -- a healthy indoor scene averages
        # ~0.017, a collapsed one ~0.00001 with a maximum under 0.01.
        import scipy.special as _sp
        ta = os.path.join(a.model, "point_cloud", f"iteration_{a.iteration}",
                          "texture_alpha.npz")
        hint = ""
        if os.path.exists(ta):
            t = np.load(ta); t = t[list(t.keys())[0]]
            al = _sp.expit(t)
            hint = (f" sigmoid(texture_alpha): mean {al.mean():.6f}, "
                    f"max {al.max():.3f}, frac>0.5 {(al > 0.5).mean():.6f}.")
        raise SystemExit(
            "no pixel passed the filters in any view, so there is nothing to "
            f"fuse.{hint} If the alpha is ~0 the model renders nothing and has "
            "to be retrained; otherwise relax --alpha_thresh / --depth_max.")
    code, P, N, C = consolidate(chunks)[0]
    pos = P / C[:, None]
    mag = np.linalg.norm(N, axis=1) / C          # 1 = all views agree, 0 = they cancel
    nrm = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    sel = C >= a.min_samples
    if a.clip_billboards >= 0:
        bb = gaussians.get_xyz.detach().cpu().numpy()
        lo = np.percentile(bb, 0.5, axis=0) - a.clip_billboards
        hi = np.percentile(bb, 99.5, axis=0) + a.clip_billboards
        inb = ((pos >= lo) & (pos <= hi)).all(axis=1)
        print(f"billboard bbox clip (margin {a.clip_billboards} m): "
              f"drops {int((sel & ~inb).sum()):,} voxels outside "
              f"{np.round(lo,2)}..{np.round(hi,2)}")
        sel &= inb
    if a.radius_clip > 0:
        cc_np = cam_centre.cpu().numpy()
        inside = np.linalg.norm(pos - cc_np, axis=1) <= a.radius_clip
        print(f"radius clip {a.radius_clip} m from the camera centroid: "
              f"drops {int((sel & ~inside).sum()):,} voxels")
        sel &= inside
    print(f"\npixels considered {stats['px']:,}; kept {stats['kept']:,} "
          f"({100*stats['kept']/max(stats['px'],1):.1f}%)")
    print(f"  rejected: low alpha {stats['lowalpha']:,}  out of range {stats['range']:,}  "
          f"depth edge {stats['edge']:,}")
    print(f"voxels {len(pos):,} -> {int(sel.sum()):,} with >= {a.min_samples} pixels")
    # |sum n| / count: 1 means every contributing pixel agreed on the normal.
    # Within one view neighbouring pixels are highly correlated, so this mostly
    # reports local smoothness and only partly cross-view consistency.
    print(f"normal agreement: median {np.median(mag[sel]):.3f} "
          f"(frac > 0.9: {(mag[sel] > 0.9).mean()*100:.1f}%)")

    out = a.out or os.path.join(a.model, f"depth_cloud_v{a.voxel}.ply")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pos[sel])
    pc.normals = o3d.utility.Vector3dVector(nrm[sel])
    o3d.io.write_point_cloud(out, pc)
    npz = os.path.splitext(out)[0] + ".npz"
    np.savez_compressed(npz, xyz=pos[sel].astype(np.float32),
                        normal=nrm[sel].astype(np.float32),
                        count=C[sel].astype(np.int32),
                        agreement=mag[sel].astype(np.float32))
    print(f"wrote {out}\nwrote {npz}")


if __name__ == "__main__":
    main()
