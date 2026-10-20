"""
Oracle variant of scripts/depth_fusion.py: fuse Replica's ground-truth depth
maps instead of the ones the trained model renders.

This is NOT the method -- real captures (drjohnson, Mip-NeRF 360) have no
ground-truth depth, and scoring a GT-depth cloud against the GT mesh compares
one ground truth with another. It exists to put an upper bound on the
pipeline: if perfect depth still cannot recover a wall, the limit is in the
segmentation, not in the model's depth.

  python scripts/gt_depth_fusion.py --scene room0 --out <...>.ply
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", default="/home/cot0514/projects/3DGS/data/replica/Replica")
    ap.add_argument("--colmap_root", default="/home/cot0514/projects/3DGS/data/replica_colmap")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--pixel_stride", type=int, default=2)
    ap.add_argument("--edge_rel", type=float, default=0.02)
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cv2
    meta = json.load(open(os.path.join(a.colmap_root, a.scene, "replica_meta.json")))
    cam = meta["intrinsics"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H, scale = cam["w"], cam["h"], cam["scale"]
    traj = np.loadtxt(os.path.join(a.replica_root, a.scene, "traj.txt")).reshape(-1, 4, 4)
    res = os.path.join(a.replica_root, a.scene, "results")

    ps = a.pixel_stride
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # OpenCV convention, matching traj.txt: x right, y down, z forward
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1)

    OFF = 1 << 17
    chunks = []
    idxs = meta["frame_indices"]
    for n, i in enumerate(idxs):
        dp = os.path.join(res, "depth%06d.png" % i)
        d = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float64) / scale
        ok = d > 0
        # same discontinuity rejection as the rendered-depth path, so the two
        # clouds differ only in where the depth came from
        jump = np.zeros_like(d)
        jump[1:-1, 1:-1] = np.maximum.reduce([
            np.abs(d[2:, 1:-1] - d[1:-1, 1:-1]), np.abs(d[:-2, 1:-1] - d[1:-1, 1:-1]),
            np.abs(d[1:-1, 2:] - d[1:-1, 1:-1]), np.abs(d[1:-1, :-2] - d[1:-1, 1:-1])])
        ok &= jump < a.edge_rel * np.maximum(d, 1e-6)
        ok[0, :] = ok[-1, :] = ok[:, 0] = ok[:, -1] = False
        ok = ok[::ps, ::ps]
        pc = (ray[::ps, ::ps] * d[::ps, ::ps, None])[ok]          # camera space
        c2w = traj[i]
        P = pc @ c2w[:3, :3].T + c2w[:3, 3]

        # normals from the same cross-product of neighbouring 3D points that
        # utils/point_utils.depth_to_normal uses
        pts = (ray * d[:, :, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        nm = np.zeros_like(pts)
        dx = pts[2:, 1:-1] - pts[:-2, 1:-1]
        dy = pts[1:-1, 2:] - pts[1:-1, :-2]
        cr = np.cross(dx, dy)
        nm[1:-1, 1:-1] = cr / (np.linalg.norm(cr, axis=-1, keepdims=True) + 1e-12)
        N = nm[::ps, ::ps][ok]
        cc = c2w[:3, 3]
        flip = ((cc - P) * N).sum(1) < 0
        N[flip] *= -1

        gi = np.floor(P / a.voxel).astype(np.int64)
        code = ((gi[:, 0] + OFF) * (2 * OFF) + (gi[:, 1] + OFF)) * (2 * OFF) + (gi[:, 2] + OFF)
        uq, inv = np.unique(code, return_inverse=True)
        Ps = np.zeros((len(uq), 3)); Ns = np.zeros((len(uq), 3)); Cs = np.zeros(len(uq), np.int64)
        np.add.at(Ps, inv, P); np.add.at(Ns, inv, N); np.add.at(Cs, inv, 1)
        chunks.append((uq, Ps, Ns, Cs))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(idxs)} frames")

    code = np.concatenate([c[0] for c in chunks])
    P = np.concatenate([c[1] for c in chunks]); N = np.concatenate([c[2] for c in chunks])
    C = np.concatenate([c[3] for c in chunks])
    uq, inv = np.unique(code, return_inverse=True)
    Pa = np.zeros((len(uq), 3)); Na = np.zeros((len(uq), 3)); Ca = np.zeros(len(uq), np.int64)
    np.add.at(Pa, inv, P); np.add.at(Na, inv, N); np.add.at(Ca, inv, C)
    pos = Pa / Ca[:, None]
    nrm = Na / (np.linalg.norm(Na, axis=1, keepdims=True) + 1e-12)
    sel = Ca >= a.min_samples
    print(f"voxels {len(pos):,} -> {int(sel.sum()):,} with >= {a.min_samples} pixels")

    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pos[sel])
    pc.normals = o3d.utility.Vector3dVector(nrm[sel])
    o3d.io.write_point_cloud(a.out, pc)
    np.savez_compressed(os.path.splitext(a.out)[0] + ".npz",
                        xyz=pos[sel].astype(np.float32), normal=nrm[sel].astype(np.float32),
                        count=Ca[sel].astype(np.int32),
                        agreement=(np.linalg.norm(Na, axis=1) / Ca)[sel].astype(np.float32))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
