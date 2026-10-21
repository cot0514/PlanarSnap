"""
Up-direction and floor-height accuracy of `analyze_walls.py` against Replica's
semantic mesh.

  python scripts/eval_floor_gt.py --walls output/replica_room0_v1/walls_depth_pca.json \
      --semantic .../official/semantic/room_0/habitat

GT up is the direction most parallel to the floor and ceiling normals and most
perpendicular to the wall normals, area-weighted.  It is NOT simply +z: room_2's
world frame is tilted 8.7 deg (floor, ceiling and walls all agree on that).

GT floor height is the area-weighted median of the LARGEST `floor` instance,
measured along the ESTIMATED up, so the two numbers compared share an axis and
the floor error is not polluted by the up error.  The largest instance, because
office_1 also labels a 1.06 m^2 platform 0.47 m above the floor as `floor`.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_walls_gt import load_semantic


def weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def gt_up(verts, tris, cls):
    t = verts[tris]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(n, axis=1)
    n = n / (a[:, None] + 1e-12)
    z = np.array([0.0, 0, 1])
    M = np.zeros((3, 3))
    for name, horiz in (("floor", True), ("ceiling", True), ("wall", False)):
        m = (cls == name) & ((np.abs(n @ z) > 0.7) if horiz else (np.abs(n @ z) < 0.3))
        sign = -1.0 if horiz else 1.0
        M += sign * (a[m, None, None] * n[m, :, None] * n[m, None, :]).sum(0)
    u = np.linalg.eigh(M)[1][:, 0]
    return u if u[2] > 0 else -u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walls", required=True, help="walls_*.json from analyze_walls --json")
    ap.add_argument("--semantic", required=True, help=".../habitat folder")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    w = json.load(open(a.walls))
    up = np.asarray(w["up"], dtype=np.float64)
    seed = np.asarray(w.get("up_seed", w["up"]), dtype=np.float64)

    verts, tris, tri_oid, cls = load_semantic(a.semantic)
    g = gt_up(verts, tris, cls)
    ang = lambda u: float(np.degrees(np.arccos(min(1.0, abs(float(u @ g))))))

    t = verts[tris]
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    fm = cls == "floor"
    ids, inv = np.unique(tri_oid[fm], return_inverse=True)
    inst_area = np.bincount(inv, weights=area[fm])
    keep = fm & (tri_oid == ids[inst_area.argmax()])
    gt_floor = weighted_median(t[keep].mean(1) @ up, area[keep])
    err = float(w["floor_height"]) - gt_floor

    print(f"up error   : seed {ang(seed):.2f} deg -> used {ang(up):.2f} deg")
    print(f"floor      : est {w['floor_height']:.3f}  GT {gt_floor:.3f}  "
          f"error {100 * err:+.1f} cm  (source: {w.get('floor_source', 'plane')})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"walls": a.walls, "up_err_seed_deg": ang(seed),
                       "up_err_deg": ang(up), "gt_floor": gt_floor,
                       "floor_height": w["floor_height"], "floor_err": err,
                       "floor_source": w.get("floor_source"),
                       "gt_up": g.tolist()}, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
