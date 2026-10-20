"""
Geometric accuracy of the recovered walls, against Replica's semantic mesh.

Point-membership metrics (precision / recall / IoU over labelled points) are
dominated by how well-localised the point cloud is, not by how well the room
was recovered: on room0 the depth cloud sits a median 6-7 cm off the true
surface, so points from furniture fall inside a wall's distance threshold and
the per-point precision reads 37% even though the fitted wall planes are
within a fraction of a degree of each other.

What a floorplan actually needs is the plane: its orientation and its offset.
This scores exactly that -- for each ground-truth wall instance, the angle and
the distance between its plane and the best matching predicted wall plane.

  python scripts/eval_wall_geometry_gt.py \
      --labeled <...>/planes_depth_pca.ply --walls <...>/walls_depth_pca.json \
      --semantic <...>/room_0/habitat
"""
import argparse, json, os, sys
import numpy as np
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_walls_gt import load_semantic


def fit(P, w=None):
    """Area- or count-weighted least-squares plane: (centroid, unit normal)."""
    if w is None:
        c = P.mean(0)
        M = P - c
    else:
        w = w / w.sum()
        c = (P * w[:, None]).sum(0)
        M = (P - c) * np.sqrt(w)[:, None]
    n = np.linalg.svd(M, full_matrices=False)[2][-1]
    return c, n / np.linalg.norm(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--walls", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--max_angle", type=float, default=20.0,
                    help="a predicted plane beyond this angle is not the wall")
    ap.add_argument("--max_offset", type=float, default=0.30,
                    help="metres between the two planes, measured at the GT wall's centroid")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    d = PlyData.read(a.labeled).elements[0].data
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    lab = np.asarray(d["plane_id"]).astype(np.int64)
    wall_ids = json.load(open(a.walls))["wall_ids"]

    verts, tris, tri_oid, tri_cls = load_semantic(a.semantic)
    cent = verts[tris].mean(1)
    area = 0.5 * np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                                         verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)

    preds = {}
    for pid in wall_ids:
        m = lab == pid
        if m.sum() >= 3:
            preds[pid] = fit(xyz[m])
    print(f"{len(preds)} predicted wall planes; GT walls:")

    gt_ids = np.unique(tri_oid[tri_cls == "wall"])
    out, matched = [], 0
    print(f"\n{'GT id':>6} {'area m2':>8} {'match':>6} {'angle':>8} {'offset':>9}")
    for g in gt_ids:
        sel = (tri_oid == g) & (tri_cls == "wall")
        c_g, n_g = fit(cent[sel], area[sel])
        best = None
        for pid, (c_p, n_p) in preds.items():
            ang = np.degrees(np.arccos(min(1.0, abs(float(n_g @ n_p)))))
            off = abs(float((c_g - c_p) @ n_p))
            if ang <= a.max_angle and off <= a.max_offset:
                if best is None or ang < best[0]:
                    best = (ang, off, pid)
        if best:
            matched += 1
            print(f"{int(g):>6} {area[sel].sum():>8.1f} {best[2]:>6} "
                  f"{best[0]:>7.2f}° {best[1]*100:>7.1f}cm")
            out.append({"gt_id": int(g), "area": float(area[sel].sum()),
                        "plane": int(best[2]), "angle_deg": best[0],
                        "offset_cm": best[1] * 100})
        else:
            print(f"{int(g):>6} {area[sel].sum():>8.1f} {'-':>6} {'-':>8} {'-':>9}")
            out.append({"gt_id": int(g), "area": float(area[sel].sum()), "plane": None})

    ok = [o for o in out if o.get("plane") is not None]
    print(f"\nGT walls matched within {a.max_angle}° and {a.max_offset*100:.0f} cm: "
          f"{matched}/{len(gt_ids)}")
    if ok:
        angs = [o["angle_deg"] for o in ok]; offs = [o["offset_cm"] for o in ok]
        print(f"  angle  : median {np.median(angs):.2f}°, max {max(angs):.2f}°")
        print(f"  offset : median {np.median(offs):.1f} cm, max {max(offs):.1f} cm")
    if a.json:
        json.dump({"labeled": a.labeled, "matched": matched,
                   "gt_walls": int(len(gt_ids)), "walls": out}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
