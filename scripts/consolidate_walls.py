"""
Merge wall fragments into one plane per wall (floorplan step 1).

Depth-cloud segmentation finds the right wall planes but splits each wall into
pieces: on room0 the largest fragment of a GT wall holds 20-57% of its points,
and 16-47% are left unassigned.  At a 50% single-plane overlap criterion that
erases depth's advantage over billboards (4 vs 6 of 22 walls, oracle 17).

Two passes, on the labelled cloud from `segment_planes.py --save`:

  1. merge   -- near-vertical planes are grouped greedily, largest first, when
               their normals agree within --merge_angle and the fragment's
               points sit within --merge_offset of the group's plane.  The
               group plane is refitted after every join.  The offset is wider
               than the segmentation distance on purpose: rendered depth lies
               5-9 cm off the surface, and blinds/pictures/TVs a few cm in
               front of a wall belong to the wall for a floorplan.
  2. absorb  -- unassigned points within --absorb_dist of a merged wall plane,
               with normals within --absorb_angle of it, and within
               --absorb_reach of the wall's existing points, join that wall.
               The reach limit stops a plane from swallowing points along its
               infinite extension (through a doorway into the next wall).
  3. prune   -- each merged wall is refitted robustly (Cauchy weights) and its
               points farther than --prune from that plane are unassigned.
               Without it a fragment that joined a few degrees tilted drags
               the least-squares plane along (median angle error 1.8 -> 2.5
               deg on Replica); with it the angle returns to 1.8 deg and the
               wall counts are unchanged.

Replica, 22 GT walls (depth + normal RANSAC): single-plane 50% overlap
6 -> 13, 25% overlap 13 -> 17.  Merging does almost all of it; absorbing
unassigned points adds little.  A 0.20 m merge offset starts fusing distinct
parallel surfaces (depth + RANSAC 18 -> 15 geometric matches).

  python scripts/consolidate_walls.py \
      --labeled .../planes_depth_pca.ply --cameras <model>/cameras.json \
      --out .../planes_depth_pca_cons.ply
"""
import argparse, os, sys
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_walls import up_from_cameras, plane_area, refine_up


def fit(q):
    c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    return c, n / np.linalg.norm(n)


def robust_fit(q, scale=0.03, iters=10):
    """Plane through q with Cauchy weights on the residuals, so that a merged
    fragment sitting a few cm off, or a few degrees tilted, cannot drag the
    whole wall plane with it the way a plain least-squares fit does."""
    c, n = fit(q)
    for _ in range(iters):
        r = (q - c) @ n
        w = 1.0 / (1.0 + (r / scale) ** 2)
        c = (w[:, None] * q).sum(0) / w.sum()
        qc = (q - c) * np.sqrt(w)[:, None]
        n = np.linalg.svd(qc, full_matrices=False)[2][-1]
    return c, n / np.linalg.norm(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vertical_deg", type=float, default=20.0)
    ap.add_argument("--min_area", type=float, default=0.1,
                    help="vertical planes smaller than this (m^2) are not merged")
    ap.add_argument("--merge_angle", type=float, default=15.0)
    ap.add_argument("--merge_offset", type=float, default=0.15)
    ap.add_argument("--absorb_dist", type=float, default=0.10)
    ap.add_argument("--absorb_angle", type=float, default=30.0)
    ap.add_argument("--absorb_reach", type=float, default=0.30)
    ap.add_argument("--prune", type=float, default=0.10,
                    help="after merging, unassign wall points farther than this "
                         "from the group's robust plane (0 = keep all)")
    a = ap.parse_args()

    ply = PlyData.read(a.labeled)
    d = ply.elements[0].data
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    lab = np.asarray(d["plane_id"]).astype(np.int64).copy()
    has_n = all(k in d.dtype.names for k in ("nx", "ny", "nz"))
    nrm = np.stack([d["nx"], d["ny"], d["nz"]], 1).astype(np.float64) if has_n else None

    ids = [int(i) for i in np.unique(lab) if i >= 0]
    fits = {i: fit(xyz[lab == i]) for i in ids}
    areas = {i: plane_area(xyz[lab == i]) for i in ids}
    up, _ = up_from_cameras(a.cameras)
    up = refine_up(up, [fits[i][1] for i in ids], [areas[i] for i in ids])

    sin_v = np.sin(np.radians(a.vertical_deg))
    vert = sorted((i for i in ids if abs(float(fits[i][1] @ up)) < sin_v
                   and areas[i] >= a.min_area), key=lambda i: -areas[i])
    print(f"{len(xyz):,} points, {len(ids)} planes, {len(vert)} vertical >= {a.min_area} m^2")

    # ---- 1. merge -------------------------------------------------------
    cos_m = np.cos(np.radians(a.merge_angle))
    groups = []                       # each: {"ids": [...], "c": c, "n": n}
    for i in vert:
        q = xyz[lab == i]
        n_i = fits[i][1]
        for g in groups:
            if abs(float(n_i @ g["n"])) < cos_m:
                continue
            if abs(float(np.median((q - g["c"]) @ g["n"]))) > a.merge_offset:
                continue
            g["ids"].append(i)
            g["c"], g["n"] = fit(xyz[np.isin(lab, g["ids"])])
            break
        else:
            groups.append({"ids": [i], "c": fits[i][0], "n": n_i})

    for g in groups:
        keep = g["ids"][0]
        lab[np.isin(lab, g["ids"])] = keep
        g["id"] = keep
    merged = [g for g in groups if len(g["ids"]) > 1]
    print(f"merge : {len(vert)} vertical planes -> {len(groups)} "
          f"({len(merged)} groups absorbed {sum(len(g['ids']) - 1 for g in merged)} fragments)")

    # ---- 2. absorb ------------------------------------------------------
    free = np.where(lab < 0)[0]
    if len(free) and groups:
        best = np.full(len(free), -1)
        best_d = np.full(len(free), np.inf)
        cos_a = np.cos(np.radians(a.absorb_angle))
        for g in groups:
            dist = np.abs((xyz[free] - g["c"]) @ g["n"])
            ok = dist < np.minimum(a.absorb_dist, best_d)
            if nrm is not None:
                ok &= np.abs(nrm[free] @ g["n"]) > cos_a
            if not ok.any():
                continue
            tree = cKDTree(xyz[lab == g["id"]])
            near = tree.query(xyz[free[ok]], distance_upper_bound=a.absorb_reach)[0]
            sel = np.where(ok)[0][np.isfinite(near)]
            best[sel] = g["id"]
            best_d[sel] = dist[sel]
        got = best >= 0
        lab[free[got]] = best[got]
        print(f"absorb: {got.sum():,} of {len(free):,} unassigned points joined a wall")

    # ---- 3. robust refit / prune --------------------------------------
    if a.prune > 0:
        dropped = 0
        for g in groups:
            m = np.where(lab == g["id"])[0]
            c, n = robust_fit(xyz[m])
            far = np.abs((xyz[m] - c) @ n) > a.prune
            lab[m[far]] = -1
            dropped += int(far.sum())
        print(f"prune : {dropped:,} wall points farther than {a.prune} m from "
              f"their robust plane unassigned")

    out = np.array(d)
    out["plane_id"] = lab.astype(out["plane_id"].dtype)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
