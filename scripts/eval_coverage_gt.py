"""
How much of each ground-truth surface class actually carries a primitive?

This is the measurement that explains why depth fusion recovers walls that the
billboards do not, even on scenes where the fused cloud has FEWER points: the
question is not how many points there are but whether they sit on the surfaces
we are trying to fit. MCMC allocates billboards where the image has texture, so
a plain wall can hold almost none.

Samples the GT semantic mesh uniformly by area and asks, per class, what
fraction of samples have a point within --radius.

  python scripts/eval_coverage_gt.py -m output/replica_room0_v1 \
      --semantic .../room_0/habitat --depth_cloud <...>.npz
"""
import argparse, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_walls_gt import load_semantic
from segment_planes import load_billboards


def sample_class(verts, tris, mask, n, rng):
    """n points spread uniformly over the area of the selected triangles."""
    t = tris[mask]
    a = 0.5 * np.linalg.norm(np.cross(verts[t[:, 1]] - verts[t[:, 0]],
                                      verts[t[:, 2]] - verts[t[:, 0]]), axis=1)
    if a.sum() <= 0:
        return np.zeros((0, 3)), 0.0
    pick = rng.choice(len(t), size=n, p=a / a.sum())
    u = rng.random((n, 1)); v = rng.random((n, 1))
    over = (u + v) > 1
    u[over] = 1 - u[over]; v[over] = 1 - v[over]
    A, B, C = verts[t[pick, 0]], verts[t[pick, 1]], verts[t[pick, 2]]
    return A + u * (B - A) + v * (C - A), float(a.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--iteration", type=int, default=32000)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--depth_cloud", required=True)
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--classes", nargs="*", default=["wall", "floor", "ceiling"])
    a = ap.parse_args()

    verts, tris, oid, cls = load_semantic(a.semantic)
    bxyz, _ = load_billboards(os.path.join(a.model, "point_cloud",
                                           f"iteration_{a.iteration}", "point_cloud.ply"))
    dxyz = np.load(a.depth_cloud)["xyz"]
    tb, td = cKDTree(bxyz.astype(np.float64)), cKDTree(dxyz.astype(np.float64))
    rng = np.random.default_rng(0)

    print(f"billboards {len(bxyz):,}   depth points {len(dxyz):,}   "
          f"radius {a.radius*100:.0f} cm")
    print(f"{'class':>10} {'area m2':>9} {'billboards':>11} {'depth':>8}")
    out = {}
    for c in a.classes:
        m = cls == c
        if not m.any():
            continue
        P, area = sample_class(verts, tris, m, a.samples, rng)
        cb = (tb.query(P, k=1, workers=-1)[0] < a.radius).mean()
        cd = (td.query(P, k=1, workers=-1)[0] < a.radius).mean()
        print(f"{c:>10} {area:>9.1f} {100*cb:>10.1f}% {100*cd:>7.1f}%")
        out[c] = (area, cb, cd)
    return out


if __name__ == "__main__":
    main()
