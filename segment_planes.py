"""
Planar segmentation from trained BBSplat model.

Methods:
  --method pca  (recommended)
    Iterative RANSAC where a point is an inlier only if it is close to the
    plane AND its local surface normal (from a k-NN PCA fit of the point
    cloud, not the billboard's stored normal) agrees with the plane normal.
    Without that test, a plane tilted through scattered furniture points
    collects thousands of inliers while being no real surface, and since
    inliers are removed as they are found, those phantom slabs consume the
    points of the walls behind them. On room_v17 plain `ransac` finds 6 walls
    in a single orientation family (yaw 15-30) although the scene has walls
    at yaw 71 and 143 too; `pca` finds 19 walls across 11 orientations, and
    planes whose own residual exceeds the fit threshold drop from 15 to 2.
    The cost is coverage: 54% of billboards land on a plane versus 83%,
    because clutter with no locally planar neighbourhood is now rejected.

  --method ransac  (default, kept for comparison)
    Iterative RANSAC plane fitting on billboard xyz positions, followed by
    the same connectivity split. Position-only, so it is vulnerable to the
    phantom-slab failure described above.

  --method normal
    KMeans on billboard normal directions, then splits each direction cluster
    into spatially connected components (DBSCAN) to separate physically
    distinct surfaces that happen to share an orientation.
    Not viable: verified at both lambda_normal=0.01 (room_v16) and the paper's
    reference value lambda_normal=0.05 (room_v17) that the billboard's own
    stored normal stays directionally biased on floor-like surfaces (mean
    angular error ~60°, statistically indistinguishable from random unit
    vectors) — raising the loss weight 5x did not fix it, so this isn't a
    tuning problem. Wall surfaces fare better (~30-37°, real but noisy
    signal) but not reliably enough to drive segmentation on its own.
    --method ransac (position-based) is the supported path for all reported
    results; --method normal is kept only for reference/comparison.

Both methods use a density-adaptive connectivity radius (`--connect_eps`,
auto-estimated from k-NN point spacing by default) to decide when two
coplanar-looking regions are actually the same physical surface.

After clustering, two cleanup passes run by default:
  1. Coplanarity merge (`--merge_angle`, `--merge_dist`) unions fragments
     that are actually the same surface — greedy RANSAC (or a connectivity
     split) commonly refits/splits one wall or floor into several IDs with
     near-identical normal and offset (e.g. occluded by furniture, or local
     fitting noise on a large surface).
  2. Color split (`--color_*`, on by default) further splits each plane by
     real color to separate materials that are geometrically coplanar with
     another surface — a rug flush on the floor, a low sofa cushion near
     floor height — which position/normal fitting alone can't distinguish.
     Colors come from projecting each billboard into its nearest training
     photo (falls back to the billboard's own learned texture color).
     Clustering is shading-robust (won't split a sunlit vs. shadowed patch
     of the same material) and each resulting region gets one more
     connectivity split, since two different objects that land in the same
     color bucket (a rug and a sofa cushion both "warm neutral") are almost
     never spatially touching.
     Tuned on room_v17 by checking a real rug-on-floor view: the rug only
     separates from the floor at `--color_max_depth` 2 (the first bisection
     of the floor plane splits something else; the rug is the second), with
     `--color_min_delta` 0.12 and `--color_min_samples` 600 (each side of a
     split must be a substantial region). Depth 1 / delta >= 0.15 leaves the
     rug merged with the floor, and the older depth 3 / min_samples 150
     over-fragments (85 regions). Color stays a refinement of the geometric
     result, not a general-purpose material classifier.
  3. k-NN majority-vote smoothing (`--smooth_k`, `--smooth_iters`) cleans up
     salt-and-pepper mislabeled billboards near cluster boundaries caused by
     normal-direction jitter from MCMC noise / texture training.
     The vote has no planarity constraint, so it trades geometric fidelity for
     tidier boundaries: with `--method ransac` it inflates the floor's height
     spread from 0.70 to 3.39 and takes planes whose residual exceeds the fit
     threshold from 4 to 15, and with `--method pca` it costs 3 of the 11
     recovered wall orientations. Pass `--smooth_k 0` when the geometry matters
     more than the boundaries.

Usage:
  python segment_planes.py -m output/room_v15 --method ransac
  python segment_planes.py -m output/room_v15 --method ransac --save --dist 0.2
  python segment_planes.py -m output/room_v15 --method normal --k_dirs 12
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from sklearn.cluster import MiniBatchKMeans
import open3d as o3d


# ── helpers ──────────────────────────────────────────────────────────────────

def scene_extent(xyz):
    """
    Robust diameter of the point cloud, used to scale the plane-fit distance
    threshold.

    `max(|xyz|)` is NOT a scene size: it measures how far the cloud sits from
    the coordinate origin.  Replica puts the origin at a room corner, so the
    same room gave 0.034 for the billboards (after a sky-box filter that is
    also origin-relative) and 0.075 for the depth cloud, and the two
    segmentations were not comparable.  A percentile span is origin-free and
    ignores the handful of SfM outliers that a min/max would follow.
    """
    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    return float(np.max(hi - lo))


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = q.unbind(-1)
    s = 2.0 / (q * q).sum(-1)
    return torch.stack([
        1 - s*(j*j + k*k), s*(i*j - k*r),     s*(i*k + j*r),
        s*(i*j + k*r),     1 - s*(i*i + k*k), s*(j*k - i*r),
        s*(i*k - j*r),     s*(j*k + i*r),     1 - s*(i*i + j*j),
    ], dim=-1).reshape(-1, 3, 3)


def load_billboards(ply_path: str):
    """Return xyz [N,3] and normals [N,3] from point_cloud.ply."""
    plydata = PlyData.read(ply_path)
    data = plydata.elements[0].data

    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
    quats = np.stack(
        [data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=1
    ).astype(np.float32)

    q = F.normalize(torch.tensor(quats), p=2, dim=-1)
    R = quaternion_to_matrix(q)
    normals = R[:, :, 2].numpy()

    return xyz, normals


def load_depth_cloud(path: str):
    """
    Return xyz [N,3] and normals [N,3] from a cloud produced by
    `scripts/depth_fusion.py`.

    The point of using this instead of the billboards: MCMC allocates
    billboards where the image has texture, so a plain wall ends up with
    almost no primitives, while every covered pixel of a rendered depth map
    yields a point regardless of how flat the surface looks.
    """
    if path.endswith(".npz"):
        d = np.load(path)
        return d["xyz"].astype(np.float32), d["normal"].astype(np.float32)
    import open3d as o3d
    pc = o3d.io.read_point_cloud(path)
    if not pc.has_normals():
        raise SystemExit(f"{path} has no normals; pass the .npz written beside it")
    return (np.asarray(pc.points, dtype=np.float32),
            np.asarray(pc.normals, dtype=np.float32))


def filter_skybox(xyz, normals, percentile=95):
    """Remove sky-box points (large radius) that distort clustering."""
    radii = np.linalg.norm(xyz, axis=1)
    thresh = np.percentile(radii, percentile)
    mask = radii < thresh
    print(f"  Sky-box filter (r < {thresh:.1f}): keep {mask.sum():,} / {len(xyz):,}")
    return xyz[mask], normals[mask], mask


def estimate_connect_eps(xyz, k=8, factor=4.0, sample_size=20000):
    """
    Density-adaptive connectivity radius: median distance to the k-th nearest
    neighbor (over a sample), scaled by `factor`. A fixed scene_extent-based
    eps is too loose for dense indoor scenes — it chains distinct surfaces
    together through clutter. This scales with actual point spacing instead.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(xyz)
    rng = np.random.RandomState(42)
    query = xyz[rng.choice(n, sample_size, replace=False)] if n > sample_size else xyz
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
    dists, _ = nn.kneighbors(query)
    median_knn = np.median(dists[:, -1])
    eps = median_knn * factor
    print(f"  median {k}-NN dist={median_knn:.4f}  →  auto connect_eps={eps:.4f} (density-based)")
    return eps


def estimate_pca_normals(xyz, k=20):
    """
    Surface normal per point, from the smallest principal direction of its k
    nearest neighbours. Unlike the billboard's own stored normal (which tracks
    the training initialisation more than the surface — see the README), this
    is derived from the point cloud itself and works on any trained model.
    """
    from scipy.spatial import cKDTree
    pts = xyz.astype(np.float64)
    _, nbr = cKDTree(pts).query(pts, k=k, workers=-1)
    local = pts[nbr] - pts[nbr].mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', local, local)
    return np.linalg.eigh(cov)[1][:, :, 0]


def cluster_planes_pca_ransac(xyz, dist_threshold=None, normal_angle=45.0,
                              min_plane_samples=400, max_planes=80,
                              n_hypotheses=1024, connect_eps=None, k_normals=20,
                              normals=None):
    """
    Iterative RANSAC where a point is an inlier only if it is close to the plane
    *and* its local surface normal agrees with the plane normal.

    Position-only RANSAC fails badly in a cluttered room: a plane tilted through
    scattered furniture points collects thousands of inliers without being any
    real surface, and because inliers are removed as they are found, these
    phantom slabs consume the points of the walls behind them. On room_v17 that
    left every detected wall in a single orientation family (yaw 15-30 deg) even
    though the scene has walls at yaw 71 and 143 as well. Requiring normal
    agreement removes the phantoms — their "inliers" face every which way — and
    both missing walls come back.

    Plane hypotheses are seeded from a point's own normal rather than from three
    random points, which is far more likely to propose a real surface, then
    refitted to their inliers a few times.
    """
    import torch as _torch

    if dist_threshold is None:
        dist_threshold = scene_extent(xyz) * 0.005
        print(f"  auto dist_threshold={dist_threshold:.3f}")
    if normals is not None:
        print(f"  Using supplied per-point normals ({len(normals):,})")
        pca_n = normals.astype(np.float64)
    else:
        print(f"  Local PCA normals (k={k_normals})...")
        pca_n = estimate_pca_normals(xyz, k=k_normals)

    dev = "cuda" if _torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        n_hypotheses = min(n_hypotheses, 256)
    X = _torch.tensor(xyz, dtype=_torch.float32, device=dev)
    N = _torch.tensor(pca_n, dtype=_torch.float32, device=dev)
    cos_thresh = float(np.cos(np.deg2rad(normal_angle)))
    gen = _torch.Generator(device=dev); gen.manual_seed(0)

    alive = _torch.ones(len(xyz), dtype=_torch.bool, device=dev)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    plane_id = 0
    print(f"  RANSAC with normal agreement < {normal_angle}° (device: {dev})...")
    for _ in range(max_planes):
        idx_alive = _torch.nonzero(alive, as_tuple=True)[0]
        if len(idx_alive) < min_plane_samples:
            break
        seeds = idx_alive[_torch.randint(0, len(idx_alive), (n_hypotheses,),
                                         device=dev, generator=gen)]
        best_count, best = -1, None
        for a in range(0, n_hypotheses, 64):
            nh = N[seeds[a:a + 64]]
            oh = (X[seeds[a:a + 64]] * nh).sum(1)
            hit = (((X[idx_alive] @ nh.T - oh[None]).abs() < dist_threshold)
                   & ((N[idx_alive] @ nh.T).abs() >= cos_thresh)).sum(0)
            c = int(hit.max())
            if c > best_count:
                w = int(hit.argmax())
                best_count, best = c, (nh[w].clone(), float(oh[w]))
        if best_count < min_plane_samples:
            break
        nrm_h, off_h = best
        for _refit in range(3):
            sel = idx_alive[((X[idx_alive] @ nrm_h - off_h).abs() < dist_threshold)
                            & ((N[idx_alive] @ nrm_h).abs() >= cos_thresh)]
            if len(sel) < 3:
                break
            centroid = X[sel].mean(0)
            nrm_h = _torch.linalg.svd(X[sel] - centroid, full_matrices=False)[2][-1]
            nrm_h = nrm_h / nrm_h.norm()
            off_h = float((centroid * nrm_h).sum())
        sel = idx_alive[((X[idx_alive] @ nrm_h - off_h).abs() < dist_threshold)
                        & ((N[idx_alive] @ nrm_h).abs() >= cos_thresh)]
        if len(sel) < min_plane_samples:
            break
        labels[sel.cpu().numpy()] = plane_id
        alive[sel] = False
        v = nrm_h.cpu().numpy()
        print(f"  Plane {plane_id:>2}: {len(sel):>6,} pts  "
              f"normal=({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})")
        plane_id += 1

    print(f"  Remaining (unassigned): {int((labels < 0).sum()):,}")
    if connect_eps is None:
        connect_eps = estimate_connect_eps(xyz)
    print("  Splitting by spatial connectivity...")
    return _split_by_connectivity(xyz, labels, plane_id, connect_eps, min_plane_samples)


# ── method A: normal-based clustering ────────────────────────────────────────

def cluster_planes_normal(xyz, normals, k_dirs=12, connect_eps=None, min_plane_samples=50):
    norms_unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    print(f"  KMeans direction clustering (k={k_dirs})...")
    km = MiniBatchKMeans(n_clusters=k_dirs, n_init=5, random_state=42, batch_size=10000)
    dir_labels = km.fit_predict(norms_unit)

    if connect_eps is None:
        connect_eps = estimate_connect_eps(xyz)

    # Same direction != same physical surface (e.g. opposite walls, separate
    # furniture faces). Split each direction cluster into spatially connected
    # components — this is what actually separates distinct surfaces, the 1D
    # offset-gap heuristic this used to rely on doesn't handle cluttered scenes.
    print("  Splitting direction clusters by spatial connectivity...")
    final_labels, plane_id = _split_by_connectivity(
        xyz, dir_labels, k_dirs, connect_eps, min_plane_samples
    )

    return final_labels, plane_id


# ── method B: RANSAC position-based ──────────────────────────────────────────

def cluster_planes_ransac(xyz, normals, dist_threshold=None,
                          min_plane_samples=200, max_planes=20,
                          ransac_iters=2000, connect_eps=None):
    """
    Iterative RANSAC plane fitting on xyz positions.
    Each iteration: fit best plane, extract inliers, remove them, repeat.
    Stops when no plane has >= min_plane_samples inliers.
    """
    extent = scene_extent(xyz)
    if dist_threshold is None:
        dist_threshold = extent * 0.005
        print(f"  scene extent={extent:.1f}  →  auto dist_threshold={dist_threshold:.3f}")
    if connect_eps is None:
        connect_eps = estimate_connect_eps(xyz)

    # Build Open3D point cloud (xyz only — no colors yet)
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(xyz)

    remaining_idx = np.arange(len(xyz), dtype=np.int32)
    final_labels  = np.full(len(xyz), -1, dtype=np.int32)
    plane_models  = []
    plane_id = 0

    for iteration in range(max_planes):
        if len(remaining_idx) < min_plane_samples:
            break

        pcd_rem = pcd_full.select_by_index(remaining_idx.tolist())
        plane_model, local_inliers = pcd_rem.segment_plane(
            distance_threshold=dist_threshold,
            ransac_n=3,
            num_iterations=ransac_iters,
        )

        if len(local_inliers) < min_plane_samples:
            print(f"  Plane {plane_id}: only {len(local_inliers)} inliers — stopping.")
            break

        global_inliers = remaining_idx[np.array(local_inliers)]
        final_labels[global_inliers] = plane_id
        plane_models.append(plane_model)

        a, b, c, d = plane_model
        print(f"  Plane {plane_id:>2}: {len(global_inliers):>6,} pts  "
              f"normal=({a:+.3f},{b:+.3f},{c:+.3f})  d={d:+.3f}")

        remaining_idx = remaining_idx[
            np.setdiff1d(np.arange(len(remaining_idx)), local_inliers)
        ]
        plane_id += 1

    print(f"  Remaining (unassigned): {len(remaining_idx):,}")

    # Stage 2: split each RANSAC plane into spatially connected components.
    # A RANSAC plane can contain disconnected surfaces (floor + ceiling + tabletop
    # all at the same height in tilted COLMAP coords). Spatial clustering separates them.
    print("  Splitting by spatial connectivity...")
    final_labels, plane_id = _split_by_connectivity(
        xyz, final_labels, plane_id, connect_eps, min_plane_samples
    )

    return final_labels, plane_id, plane_models


def smooth_labels_knn(xyz, labels, k=16, iterations=2):
    """
    Majority-vote label smoothing over the spatial k-NN graph.

    Per-billboard independent clustering leaves salt-and-pepper noise near
    cluster boundaries (a few billboards on an otherwise uniform wall get
    assigned to a neighboring plane because their normal drifted slightly
    during MCMC noise injection / texture training). Relabeling each point
    to the majority label among its k nearest neighbors removes this without
    needing a stricter distance/normal threshold upstream.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import mode as _mode

    if len(xyz) == 0:
        return labels

    nn = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
    _, neighbor_idx = nn.kneighbors(xyz)
    neighbor_idx = neighbor_idx[:, 1:]  # drop self

    current = labels.copy()
    for it in range(iterations):
        neighbor_labels = current[neighbor_idx]  # [N, k]
        new_labels = _mode(neighbor_labels, axis=1, keepdims=False).mode.astype(np.int32)
        changed = (new_labels != current).sum()
        current = new_labels
        print(f"  Smoothing pass {it+1}/{iterations}: {changed:,} labels changed")
        if changed == 0:
            break

    return current


def find_images_dir(model_path, source_path=None):
    """Locate the dataset's image folder (smallest available resolution first)."""
    import re

    if source_path is None:
        cfg_path = os.path.join(model_path, "cfg_args")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                m = re.search(r"source_path=['\"]([^'\"]*)['\"]", f.read())
            if m:
                source_path = m.group(1)
    if not source_path:
        return None
    for cand in ("images_8", "images_4", "images_2", "images"):
        p = os.path.join(source_path, cand)
        if os.path.isdir(p):
            return p
    return None


def project_colors_from_photos(xyz, cams, images_dir):
    """Color each point by projecting it into its nearest valid training camera."""
    from PIL import Image

    n = len(xyz)
    best_dist = np.full(n, np.inf)
    best_color = np.zeros((n, 3))
    hit = np.zeros(n, dtype=bool)

    for c in cams:
        img_path = os.path.join(images_dir, c["img_name"] + ".JPG")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, c["img_name"] + ".jpg")
            if not os.path.exists(img_path):
                continue
        R = np.array(c["rotation"])
        C = np.array(c["position"])
        fx_full, fy_full, w_full = c["fx"], c["fy"], c["width"]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        h_s, w_s = img.shape[:2]
        s = w_s / w_full
        fx, fy = fx_full * s, fy_full * s
        cx, cy = w_s / 2, h_s / 2

        Xc = (xyz - C) @ R
        depth = Xc[:, 2]
        valid = depth > 1e-4
        u = fx * Xc[:, 0] / np.clip(depth, 1e-6, None) + cx
        v = fy * Xc[:, 1] / np.clip(depth, 1e-6, None) + cy
        valid &= (u >= 0) & (u < w_s) & (v >= 0) & (v < h_s)
        if not np.any(valid):
            continue
        dist = np.linalg.norm(xyz - C, axis=1)
        better = valid & (dist < best_dist)
        idxs = np.where(better)[0]
        if len(idxs) == 0:
            continue
        uu, vv = u[idxs].astype(np.int32), v[idxs].astype(np.int32)
        best_color[idxs] = img[vv, uu] / 255.0
        best_dist[idxs] = dist[idxs]
        hit[idxs] = True

    print(f"  Color projection hit rate: {100*hit.mean():.1f}%")
    return best_color, hit


def texture_mean_colors(pc_iter_dir, orig_idx):
    """Fallback: alpha-weighted mean of each billboard's own learned texture."""
    ta_path = os.path.join(pc_iter_dir, "texture_alpha.npz")
    tc_path = os.path.join(pc_iter_dir, "texture_color.npz")
    if not (os.path.exists(ta_path) and os.path.exists(tc_path)):
        return None
    ta = np.load(ta_path)["texture_alpha"][orig_idx]
    tc = np.load(tc_path)["texture_color"][orig_idx]
    alpha = 1 / (1 + np.exp(-ta))
    color = 1 / (1 + np.exp(-tc))
    w = alpha[:, None, :, :]
    wsum = w.sum(axis=(2, 3)) + 1e-6
    mean_color = (color * w).sum(axis=(2, 3)) / wsum
    return np.clip(mean_color, 0, 1)


def get_point_colors(model_path, pc_iter_dir, xyz, orig_idx, source_path=None,
                     allow_texture_fallback=True):
    """
    Real per-billboard RGB (xyz order), used to tell apart materials that are
    geometrically coplanar (a rug lying flush on the floor, a low sofa
    cushion near floor height) but visually distinct.

    Preferred source: project each billboard into its nearest training
    camera and sample the actual photo (needs the original dataset — found
    via cfg_args unless `source_path` overrides it). Falls back to the
    alpha-weighted mean of the billboard's own learned texture for any point
    no camera resolves (or entirely, if the dataset isn't reachable).
    """
    import json

    images_dir = find_images_dir(model_path, source_path)
    cam_json_path = os.path.join(model_path, "cameras.json")

    if images_dir and os.path.exists(cam_json_path):
        print(f"  Projecting real colors from {images_dir}...")
        with open(cam_json_path) as f:
            cams = json.load(f)
        colors, hit = project_colors_from_photos(xyz, cams, images_dir)
        if not hit.all() and allow_texture_fallback:
            tex_colors = texture_mean_colors(pc_iter_dir, orig_idx)
            if tex_colors is not None:
                colors[~hit] = tex_colors[~hit]
        return colors

    if not allow_texture_fallback:
        print("  No dataset images found; color splitting disabled (the learned "
              "textures are per-billboard and cannot index a depth cloud).")
        return None
    print("  No dataset images found for real-color projection — falling back to "
          "learned texture color.")
    colors = texture_mean_colors(pc_iter_dir, orig_idx)
    if colors is None:
        print("  No texture files either — color-based splitting disabled.")
    return colors


def _shading_robust_feature(colors, luma_weight=0.3):
    """
    Recast RGB so Euclidean distance responds mainly to hue/chroma and only
    weakly to brightness — the same wood floor lit near a window vs. in
    shadow must not look like two materials. Luma (perceptual brightness) is
    split off and down-weighted; the chroma residual (color minus its own
    gray) carries the real material signal.
    """
    luma = colors @ np.array([0.299, 0.587, 0.114])
    chroma = colors - luma[:, None]
    return np.concatenate([chroma, (luma * luma_weight)[:, None]], axis=1)


def split_by_color(xyz, colors, labels, n_planes, k_smooth=20, min_delta=0.12,
                    min_samples=600, max_depth=2, luma_weight=0.3, connect_eps=None,
                    min_fragment=200):
    """
    Within each detected plane, further split by color to separate different
    materials that happen to be geometrically coplanar — a rug flush on the
    floor, a low sofa cushion near floor height. Pure position/normal fitting
    is blind to this: the points really are on (almost) the same plane.

    A single patterned material (a busy rug) must not get fragmented by its
    own internal color variation, so colors are locally averaged (k-NN mean,
    `k_smooth`) before clustering — this washes out within-material pattern
    noise while preserving the material-to-material color gap. Clustering
    then runs on a shading-robust feature (`luma_weight`, see
    `_shading_robust_feature`) so a lit-vs-shadowed patch of the *same*
    material doesn't masquerade as two materials. A plane is only ever split
    (via recursive 2-way KMeans) when the best split found is a confident gap
    (`min_delta`); otherwise it's left as one region. `max_depth` caps how
    many times a single plane can be re-split (default 2: enough for
    floor -> {wood, other} -> {wood, rug}) — this is a refinement of the
    geometric result, not an independent material classifier, so it's kept
    shallow.

    Color alone can't tell apart two *different* objects that happen to share
    a similar hue-bucket after luma-downweighting (e.g. a rug and a sofa
    cushion both landing in "warm neutral") — but they're essentially never
    spatially touching. So each resulting color region gets one more
    connectivity split (same routine the geometric stage uses) to separate
    color-alike-but-disconnected objects.

    `min_samples` gates whether a *split* is attempted/accepted (each side
    of a bisection must clear it) — it is not a keep/discard filter on
    whole planes. `min_fragment` is the separate, smaller cutoff for the
    final connectivity split (pieces below it are dropped); it should match
    the geometric stage's `min_plane_samples`, otherwise this stage would
    discard small-but-legitimate surface pieces (e.g. floor visible between
    furniture) that the geometric stage had kept. A plane that arrived here already smaller than
    `min_samples` passed the geometric stage's own size filter upstream, so
    it's kept intact rather than silently dropped just because color
    splitting won't touch it.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans

    new_labels = np.full(len(xyz), -1, dtype=np.int32)
    next_id = 0

    for pid in range(n_planes):
        idx = np.where(labels == pid)[0]
        if len(idx) == 0:
            continue
        pts, cols = xyz[idx], colors[idx]

        k = min(k_smooth, len(idx) - 1)
        if k >= 1:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
            _, nbr = nn.kneighbors(pts)
            smoothed = cols[nbr].mean(axis=1)
        else:
            smoothed = cols
        smoothed = _shading_robust_feature(smoothed, luma_weight)

        # recursively bisect by locally-smoothed color; stop when the split
        # isn't a confident color gap, or a side would be too small
        stack = [(np.arange(len(idx)), 0)]
        groups = []
        while stack:
            sub, depth = stack.pop()
            if len(sub) < 2 * min_samples or depth >= max_depth:
                groups.append(sub)
                continue
            km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(smoothed[sub])
            c0, c1 = km.cluster_centers_
            delta = np.linalg.norm(c0 - c1)
            side0 = sub[km.labels_ == 0]
            side1 = sub[km.labels_ == 1]
            if delta < min_delta or len(side0) < min_samples or len(side1) < min_samples:
                groups.append(sub)
                continue
            stack.append((side0, depth + 1))
            stack.append((side1, depth + 1))

        for g in groups:
            if len(g) == 0:
                continue
            new_labels[idx[g]] = next_id
            next_id += 1

    print(f"  Color split: {n_planes} → {next_id} regions "
          f"(k_smooth={k_smooth}, min_delta={min_delta}, min_samples={min_samples})")

    if connect_eps is None:
        connect_eps = estimate_connect_eps(xyz)
    print("  Splitting color regions by spatial connectivity...")
    final_labels, final_id = _split_by_connectivity(xyz, new_labels, next_id, connect_eps, min_fragment)
    print(f"  Color regions after connectivity split: {next_id} → {final_id}")
    return final_labels, final_id


def merge_coplanar_planes(xyz, labels, n_planes, angle_thresh_deg=5.0, dist_thresh=0.15,
                          max_rms=None):
    """
    Merge plane fragments that are actually the same physical surface.

    Greedy RANSAC (and, to a lesser extent, direction-clustering + connectivity
    split) can refit the same wall/floor multiple times across iterations with
    a near-identical normal but a slightly different offset — a wall occluded
    by furniture leaves disconnected islands, or a large near-planar surface
    has enough local noise that one RANSAC pass doesn't capture all of it. Each
    such fragment gets its own ID even though it is the same surface. This pass
    fits a plane to each label's points and unions labels whose normals are
    nearly parallel and whose planes are nearly coincident.

    Compatibility is re-evaluated against the *growing group*, not the two
    original fragments, and a merge is rejected unless the combined group is
    still flat (RMS distance to its own refitted plane <= `max_rms`, which
    defaults to `dist_thresh`). Checking only the original pairs lets a chain
    of individually-fine merges drift arbitrarily far: A~B and B~C each pass,
    so A and C land in one group even when they are nowhere near coplanar.
    Observed on room_v18 before this check existed — 84 fragments collapsed to
    11 "planes", one of which mixed surfaces 1.5 units apart under a 0.19
    threshold (floor merged with ceiling/furniture tiers).

    A group's plane fit only needs its point count, sum and sum of outer
    products, and all three are additive, so merging is O(1) per candidate
    pair rather than a fresh SVD over the union's points.
    """
    if max_rms is None:
        max_rms = dist_thresh

    stats = {}
    for pid in range(n_planes):
        pts = xyz[labels == pid].astype(np.float64)
        if len(pts) < 3:
            continue
        stats[pid] = (len(pts), pts.sum(0), pts.T @ pts)

    def fit(s):
        """(centroid, unit normal, RMS distance of the points to that plane)."""
        cnt, ssum, souter = s
        centroid = ssum / cnt
        cov = souter / cnt - np.outer(centroid, centroid)
        evals, evecs = np.linalg.eigh(cov)          # ascending: [0] is the normal direction
        normal = evecs[:, 0]
        return centroid, normal / (np.linalg.norm(normal) + 1e-8), np.sqrt(max(evals[0], 0.0))

    pids = list(stats.keys())
    parent = {pid: pid for pid in pids}
    group = dict(stats)                              # keyed by root only

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cos_thresh = np.cos(np.deg2rad(angle_thresh_deg))
    n_merges = 0
    changed = True
    while changed:                                   # group fits move as groups grow
        changed = False
        roots = sorted(group.keys())
        for i in range(len(roots)):
            a = roots[i]
            if a not in group:
                continue
            ca, na, _ = fit(group[a])
            for j in range(i + 1, len(roots)):
                b = roots[j]
                if b not in group:
                    continue
                cb, nb, _ = fit(group[b])
                if abs(np.dot(na, nb)) < cos_thresh:
                    continue
                if abs(np.dot(cb - ca, na)) >= dist_thresh:
                    continue
                combined = tuple(x + y for x, y in zip(group[a], group[b]))
                if fit(combined)[2] > max_rms:
                    continue                         # union would not be one flat surface
                parent[b] = a
                group[a] = combined
                del group[b]
                ca, na, _ = fit(group[a])
                n_merges += 1
                changed = True

    root_to_new = {}
    new_labels = np.full(len(xyz), -1, dtype=np.int32)
    next_new = 0
    for pid in pids:
        root = find(pid)
        if root not in root_to_new:
            root_to_new[root] = next_new
            next_new += 1
        new_labels[labels == pid] = root_to_new[root]

    worst_rms = max((fit(group[find(p)])[2] for p in pids), default=0.0)
    print(f"  Coplanarity merge: {len(pids)} → {next_new} planes "
          f"(angle<{angle_thresh_deg}°, offset<{dist_thresh:.3f}, "
          f"flatness RMS<{max_rms:.3f}; worst merged group RMS={worst_rms:.3f})")
    return new_labels, next_new


def _split_by_connectivity(xyz, labels, next_id, connect_eps, min_samples):
    """
    For each existing plane label, cluster points spatially with DBSCAN
    and assign separate IDs to disconnected components.
    """
    from sklearn.cluster import DBSCAN as _DBSCAN

    old_labels = labels.copy()
    new_labels = np.full(len(xyz), -1, dtype=np.int32)
    new_id = 0

    for pid in range(next_id):
        mask = old_labels == pid
        if mask.sum() < min_samples:
            continue
        pts = xyz[mask]
        idx = np.where(mask)[0]

        cc = _DBSCAN(eps=connect_eps, min_samples=5,
                     metric='euclidean', algorithm='ball_tree', n_jobs=-1
                     ).fit_predict(pts)

        for comp in np.unique(cc):
            if comp == -1:
                continue
            members = idx[cc == comp]
            if len(members) < min_samples:
                continue
            new_labels[members] = new_id
            new_id += 1

    return new_labels, new_id


# ── stats ─────────────────────────────────────────────────────────────────────

def print_stats(xyz, normals, labels, n_planes, plane_models=None):
    total = len(labels)
    noise = (labels == -1).sum()
    print(f"\n{'='*62}")
    print(f"  Planes detected : {n_planes}")
    print(f"  Total billboards: {total:,}")
    print(f"  Assigned        : {total - noise:,}  ({100*(total-noise)/total:.1f}%)")
    print(f"  Noise / outliers: {noise:,}")
    print(f"{'='*62}")
    sizes = sorted([(pid, (labels==pid).sum()) for pid in range(n_planes)],
                   key=lambda x: -x[1])
    header = "plane normal (RANSAC)" if plane_models else "plane normal (fit from xyz)"
    print(f"  {'ID':>4}  {'count':>7}  {header:^34}")
    for pid, cnt in sizes:
        if cnt == 0:
            continue  # smoothing can empty out a small id entirely
        if plane_models and pid < len(plane_models):
            a, b, c, d = plane_models[pid]
            print(f"  {pid:>4}  {cnt:>7,}  ({a:+.3f}, {b:+.3f}, {c:+.3f})  d={d:+.3f}")
        else:
            # Geometric plane normal fit to the points themselves — the
            # billboard's own rotation (`normals` arg) doesn't reliably track
            # the true surface orientation unless normal-supervised training
            # was used, so it's not a trustworthy fallback here.
            pts = xyz[labels == pid]
            centroid = pts.mean(0)
            _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
            mn = vt[-1]
            mn /= np.linalg.norm(mn) + 1e-8
            print(f"  {pid:>4}  {cnt:>7,}  ({mn[0]:+.3f}, {mn[1]:+.3f}, {mn[2]:+.3f})")
    print()


# ── save labeled PLY ──────────────────────────────────────────────────────────

def save_labeled_ply(save_path, xyz, normals, labels):
    dtype = [('x','f4'),('y','f4'),('z','f4'),
             ('nx','f4'),('ny','f4'),('nz','f4'),
             ('plane_id','i4')]
    arr = np.zeros(len(xyz), dtype=dtype)
    arr['x'],arr['y'],arr['z']    = xyz[:,0],    xyz[:,1],    xyz[:,2]
    arr['nx'],arr['ny'],arr['nz'] = normals[:,0], normals[:,1], normals[:,2]
    arr['plane_id'] = labels
    PlyData([PlyElement.describe(arr, 'vertex')]).write(save_path)
    print(f"  Saved: {save_path}")


# ── visualization ─────────────────────────────────────────────────────────────

def make_palette(n):
    np.random.seed(42)
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = np.zeros((n, 3))
    for i, h in enumerate(hues):
        hi = int(h * 6)
        f  = h * 6 - hi
        p, q, t, v = 0.9*(1-0.8), 0.9*(1-0.8*f), 0.9*(1-0.8*(1-f)), 0.9
        colors[i] = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][hi % 6]
    return colors


def save_colored_ply(save_path, xyz, normals, labels, n_planes):
    """Save PLY with per-point RGB color — open in MeshLab / CloudCompare."""
    palette = (make_palette(max(n_planes, 1)) * 255).astype(np.uint8)
    noise_color = np.array([30, 30, 30], dtype=np.uint8)

    dtype = [('x','f4'),('y','f4'),('z','f4'),
             ('nx','f4'),('ny','f4'),('nz','f4'),
             ('red','u1'),('green','u1'),('blue','u1'),
             ('plane_id','i4')]
    arr = np.zeros(len(xyz), dtype=dtype)
    arr['x'],arr['y'],arr['z']    = xyz[:,0],    xyz[:,1],    xyz[:,2]
    arr['nx'],arr['ny'],arr['nz'] = normals[:,0], normals[:,1], normals[:,2]
    arr['plane_id'] = labels

    rgb = np.tile(noise_color, (len(xyz), 1))
    for pid in range(n_planes):
        rgb[labels == pid] = palette[pid]
    arr['red'],arr['green'],arr['blue'] = rgb[:,0], rgb[:,1], rgb[:,2]

    PlyData([PlyElement.describe(arr, 'vertex')]).write(save_path)
    print(f"  Colored PLY saved: {save_path}")
    print(f"  → Open in MeshLab or CloudCompare to view")


def save_preview_png(save_path, xyz, labels, n_planes):
    """Save top-down and side-view scatter plots as PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    palette = make_palette(max(n_planes, 1))
    point_colors = np.full((len(xyz), 3), 0.12)
    for pid in range(n_planes):
        point_colors[labels == pid] = palette[pid]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0d0d0d')
    views = [
        ('Top (XZ)',  xyz[:,0], xyz[:,2]),
        ('Front (XY)',xyz[:,0], xyz[:,1]),
        ('Side (YZ)', xyz[:,1], xyz[:,2]),
    ]
    for ax, (title, vx, vy) in zip(axes, views):
        ax.set_facecolor('#0d0d0d')
        # subsample for speed
        idx = np.random.choice(len(xyz), min(50000, len(xyz)), replace=False)
        ax.scatter(vx[idx], vy[idx], c=point_colors[idx], s=0.3, linewidths=0)
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Preview PNG saved: {save_path}")


def visualize(xyz, normals, labels, n_planes, out_dir="."):
    """Try Open3D window; on failure save colored PLY + PNG instead."""
    palette = make_palette(max(n_planes, 1))
    point_colors = np.full((len(xyz), 3), 0.12)
    for pid in range(n_planes):
        point_colors[labels == pid] = palette[pid]

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz)
    pcd.colors  = o3d.utility.Vector3dVector(point_colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    try:
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window("Planar Segmentation — BBSplat", width=1280, height=720)
        if not ok:
            raise RuntimeError("create_window returned False")
        vis.add_geometry(pcd)
        ro = vis.get_render_option()
        if ro is None:
            raise RuntimeError("get_render_option returned None (no OpenGL)")
        ro.background_color = np.array([0.05, 0.05, 0.05])
        ro.point_size = 2.5
        ro.light_on   = True
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"\n[Open3D window unavailable: {e}]")
        print("  Saving colored PLY + PNG preview instead...")
        ply_path = os.path.join(out_dir, "planes_colored.ply")
        png_path = os.path.join(out_dir, "planes_preview.png")
        save_colored_ply(ply_path, xyz, normals, labels, n_planes)
        save_preview_png(png_path, xyz, labels, n_planes)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Planar segmentation of BBSplat trained model")
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--iteration", "-i", type=int, default=None,
                        help="Checkpoint iteration (default: latest)")
    parser.add_argument("--depth_cloud", default=None,
                        help="segment a fused depth cloud from "
                             "scripts/depth_fusion.py (.npz or .ply) instead of "
                             "the billboards; with --method pca its depth-derived "
                             "normals are used in place of local PCA")
    parser.add_argument("--method", choices=["normal", "ransac", "pca"], default="ransac",
                        help="Segmentation method (default: ransac). 'pca' is RANSAC with a "
                             "local-surface-normal agreement test, which stops phantom tilted "
                             "planes from forming and is the only method that recovers all of "
                             "this room's wall orientations — recommended, see --normal_angle")
    parser.add_argument("--normal_source", choices=["auto", "pca", "own"], default="auto",
                        help="which normals constrain --method pca. 'auto' uses "
                             "the depth-map normals for --depth_cloud and a local "
                             "k-NN PCA fit for the billboards. 'own' uses the "
                             "billboard's stored quaternion normal, which against "
                             "the Replica GT mesh is 11-19 deg more accurate than "
                             "local PCA (20.7-25.6 deg vs 31.4-44.1) once the "
                             "model was trained with --rotation_init pca")
    parser.add_argument("--normal_angle", type=float, default=45.0,
                        help="[pca] Max angle (deg) between a point's local surface normal and "
                             "the plane normal for it to count as an inlier (default: 45). "
                             "Lower is stricter: 25 leaves only 37%% of billboards assigned, "
                             "45 leaves 63%% and finds the most wall orientations")
    parser.add_argument("--k_normals", type=int, default=20,
                        help="[pca] Neighbours used to estimate each point's surface normal "
                             "(default: 20)")
    # RANSAC options
    parser.add_argument("--dist", type=float, default=None,
                        help="[ransac] Inlier distance threshold (default: auto = 0.5%% of scene)")
    parser.add_argument("--max_planes", type=int, default=None,
                        help="Max planes to extract (default: 20 for ransac, 80 for pca — pca "
                             "rejects phantom planes, so it needs more rounds to cover the scene)")
    parser.add_argument("--ransac_iters", type=int, default=2000,
                        help="[ransac] RANSAC iterations per plane (default: 2000)")
    # Normal-method options
    parser.add_argument("--k_dirs", type=int, default=12,
                        help="[normal] KMeans direction clusters (default: 12)")
    # Shared
    parser.add_argument("--connect_eps", type=float, default=None,
                        help="Spatial connectivity radius used to split a candidate plane into "
                             "disconnected surfaces (default: auto, density-based via k-NN distance)")
    parser.add_argument("--min_plane_samples", type=int, default=200,
                        help="Min billboards to keep a plane (default: 200)")
    parser.add_argument("--smooth_k", type=int, default=0,
                        help="k-NN neighbors used for majority-vote label smoothing "
                             "(tidies cluster boundaries but has no planarity "
                             "constraint, so it degrades the geometry it smooths — "
                             "see the module docstring; default: 0 = off, 16 was "
                             "the old default)")
    parser.add_argument("--smooth_iters", type=int, default=2,
                        help="Number of smoothing passes (default: 2)")
    parser.add_argument("--merge_angle", type=float, default=5.0,
                        help="Coplanarity merge: max angle (deg) between plane normals "
                             "to consider two fragments the same surface (default: 5.0, "
                             "0 disables merging)")
    parser.add_argument("--merge_dist", type=float, default=None,
                        help="Coplanarity merge: max point-to-plane offset to merge two "
                             "fragments (default: auto = 2x the plane-fit distance threshold)")
    parser.add_argument("--no_color_split", action="store_true",
                        help="Disable color-based sub-splitting (see --color_* below). "
                             "Separates materials that are geometrically coplanar with "
                             "another surface — a rug flush on the floor, a low sofa "
                             "cushion — which position/normal fitting can't tell apart.")
    parser.add_argument("--color_k_smooth", type=int, default=20,
                        help="Color split: k-NN neighbors used to locally average color "
                             "before clustering, so a single patterned material (a busy "
                             "rug) isn't fragmented by its own texture (default: 20)")
    parser.add_argument("--color_min_delta", type=float, default=0.12,
                        help="Color split: min distance (shading-robust feature space) between "
                             "two candidate regions to accept a split (default: 0.12; >=0.15 "
                             "stopped separating a rug from the floor on room_v17)")
    parser.add_argument("--color_min_samples", type=int, default=600,
                        help="Color split: each side of a split must have at least this many "
                             "billboards (default: 600). Pieces left after the final "
                             "connectivity split are kept down to --min_plane_samples.")
    parser.add_argument("--color_max_depth", type=int, default=2,
                        help="Color split: max times a single plane may be recursively "
                             "re-split (default: 2 — the rug/floor split on room_v17 is the "
                             "second-level split; depth 3 over-fragments)")
    parser.add_argument("--color_luma_weight", type=float, default=0.3,
                        help="Color split: weight on brightness vs. hue/chroma when "
                             "clustering, 0-1 (default: 0.3 — keeps the split from "
                             "firing on lit-vs-shadowed patches of the same material)")
    parser.add_argument("--source_path", default=None,
                        help="Dataset root for real-color projection (default: read from "
                             "the model's cfg_args)")
    parser.add_argument("--radius_clip", type=float, default=None,
                        help="keep only points within this distance of the "
                             "camera centroid (from cameras.json). Unlike "
                             "--no_skybox_filter's percentile on |xyz|, this is "
                             "an absolute, scene-anchored region, so a billboard "
                             "run and a --depth_cloud run can be compared over "
                             "exactly the same volume")
    parser.add_argument("--no_skybox_filter", action="store_true",
                        help="Keep sky-box points (default: filter out top 5%% by radius)")
    parser.add_argument("--save", action="store_true",
                        help="Save labeled PLY (planes.ply)")
    parser.add_argument("--no_vis", action="store_true",
                        help="Skip Open3D visualization")
    args = parser.parse_args()
    if args.max_planes is None:
        args.max_planes = 80 if args.method == "pca" else 20

    # Resolve checkpoint
    pc_root = os.path.join(args.model_path, "point_cloud")
    if args.iteration is None:
        iters = [int(d.split("_")[-1]) for d in os.listdir(pc_root)
                 if d.startswith("iteration_")]
        args.iteration = max(iters)
    ply_path = os.path.join(pc_root, f"iteration_{args.iteration}", "point_cloud.ply")
    print(f"\nLoading: {ply_path}")

    pc_iter_dir = os.path.join(pc_root, f"iteration_{args.iteration}")

    if args.depth_cloud:
        xyz, normals = load_depth_cloud(args.depth_cloud)
        print(f"  Depth-fused points: {len(xyz):,} from {args.depth_cloud}")
    else:
        xyz, normals = load_billboards(ply_path)
        print(f"  Billboards: {len(xyz):,}")

    # Sky-box filter
    full_xyz, full_normals = xyz.copy(), normals.copy()
    keep_mask = np.ones(len(xyz), dtype=bool)
    if args.radius_clip is not None:
        import json as _json
        cj = os.path.join(args.model_path, "cameras.json")
        with open(cj) as f:
            cc = np.array([c["position"] for c in _json.load(f)]).mean(0)
        keep_mask = np.linalg.norm(xyz - cc, axis=1) <= args.radius_clip
        print(f"  Radius clip r<{args.radius_clip} from the camera centroid: "
              f"keep {keep_mask.sum():,} / {len(xyz):,}")
        xyz, normals = xyz[keep_mask], normals[keep_mask]
    elif args.depth_cloud:
        print("  Sky-box filter skipped (depth cloud is already range-limited "
              "by scripts/depth_fusion.py).")
    elif not args.no_skybox_filter:
        xyz, normals, keep_mask = filter_skybox(xyz, normals)
    orig_idx = np.where(keep_mask)[0]

    # Segment
    print(f"\nSegmenting ({args.method})...")
    plane_models = None
    if args.method == "ransac":
        labels, n_planes, plane_models = cluster_planes_ransac(
            xyz, normals,
            dist_threshold=args.dist,
            min_plane_samples=args.min_plane_samples,
            max_planes=args.max_planes,
            ransac_iters=args.ransac_iters,
            connect_eps=args.connect_eps,
        )
    elif args.method == "pca":
        if args.depth_cloud or args.normal_source == "own":
            # depth clouds carry their own per-point normals; for billboards
            # this is the stored quaternion normal
            src_normals = normals
            print(f"  normal source: {'depth map' if args.depth_cloud else 'billboard quaternion'}")
        else:
            src_normals = None
            print("  normal source: local k-NN PCA")
        labels, n_planes = cluster_planes_pca_ransac(
            xyz,
            normals=src_normals,
            dist_threshold=args.dist,
            normal_angle=args.normal_angle,
            min_plane_samples=args.min_plane_samples,
            max_planes=args.max_planes,
            connect_eps=args.connect_eps,
            k_normals=args.k_normals,
        )
    else:
        labels, n_planes = cluster_planes_normal(
            xyz, normals,
            k_dirs=args.k_dirs,
            connect_eps=args.connect_eps,
            min_plane_samples=args.min_plane_samples,
        )

    if args.merge_angle > 0:
        merge_dist = args.merge_dist
        if merge_dist is None:
            merge_dist = scene_extent(xyz) * 0.005 * 2
            print(f"  auto merge_dist={merge_dist:.3f} (2x plane-fit distance threshold)")
        print("\nMerging coplanar fragments...")
        labels, n_planes = merge_coplanar_planes(
            xyz, labels, n_planes,
            angle_thresh_deg=args.merge_angle, dist_thresh=merge_dist,
        )
        plane_models = None  # fragment→plane_model indices no longer valid after merge

    if not args.no_color_split:
        print("\nSplitting by color (materials coplanar with another surface)...")
        colors = get_point_colors(args.model_path, pc_iter_dir, xyz, orig_idx,
                                   source_path=args.source_path,
                                   allow_texture_fallback=not args.depth_cloud)
        if colors is not None:
            labels, n_planes = split_by_color(
                xyz, colors, labels, n_planes,
                k_smooth=args.color_k_smooth, min_delta=args.color_min_delta,
                min_samples=args.color_min_samples, max_depth=args.color_max_depth,
                luma_weight=args.color_luma_weight,
                connect_eps=args.connect_eps, min_fragment=args.min_plane_samples,
            )
            plane_models = None  # fragment→plane_model indices no longer valid after split

    if args.smooth_k > 0:
        print(f"\nSmoothing labels (k={args.smooth_k}, iters={args.smooth_iters})...")
        labels = smooth_labels_knn(xyz, labels, k=args.smooth_k, iterations=args.smooth_iters)

    plane_models_arg = plane_models if args.method == "ransac" else None
    print_stats(xyz, normals, labels, n_planes, plane_models=plane_models_arg)

    # Map back to full point cloud (sky-box = -1)
    full_labels = np.full(len(full_xyz), -1, dtype=np.int32)
    full_labels[keep_mask] = labels

    if args.save:
        save_labeled_ply(os.path.join(pc_iter_dir, "planes.ply"),
                         full_xyz, full_normals, full_labels)

    if not args.no_vis:
        visualize(xyz, normals, labels, n_planes, out_dir=pc_iter_dir)


if __name__ == "__main__":
    main()
