"""
Floorplan accuracy beyond footprint IoU (floorplan steps 3-4).

Footprint IoU saturates on Replica: the rooms are rectangles, so the extent of
accurate points alone scores 91% without a single wall.  This script measures
what IoU hides.

2-D, the polygon against the GT outline (GT ceiling projected down, holes
filled, simplified at --corner_tol):
  * boundary Chamfer -- mean distance between the two outlines, both ways (cm)
  * corners          -- precision / recall of polygon corners within --corner_dist
  * rotation         -- the polygon's Manhattan orientation against the GT's
  * wall-backed      -- share of the polygon's perimeter lying on a detected
                        wall line rather than a virtual one (room_polygon.py
                        closes wall-less sides at the points' extent)

3-D, the room's walls rebuilt amodally: every polygon edge extruded from the
estimated floor to the estimated ceiling (step 3).
  * wall area recall -- share of the GT `wall` surface within --dist of the
                        extruded walls, against the same share for the
                        observed wall-plane points (what segmentation alone
                        gives, hidden parts missing)
  * wall instances   -- GT walls with at least half their area recovered

No 3-D precision: Replica's windows sit in recesses deeper than 10 cm, so the
extruded wall across a window is "in the air" for every method alike (the
oracle scores 73%, all of the loss above sill height); wall placement is what
the 2-D Chamfer measures.
  * ceiling height   -- error against the GT ceiling

  python scripts/eval_floorplan_gt.py --room <model>/room_depth_pca_cons.json \
      --semantic .../official/semantic/room_0/habitat
"""
import argparse, json, os, sys
import numpy as np
import cv2
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_walls_gt import load_semantic
from eval_footprint_gt import raster_tris
from analyze_walls import load_labeled


def sample_boundary(poly, step):
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        t = np.arange(n)[:, None] / n
        pts.append(a + t * (b - a))
    return np.concatenate(pts)


def sample_triangles(t, per_m2, rng):
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    n = rng.poisson(area * per_m2)
    idx = np.repeat(np.arange(len(t)), n)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    tt = t[idx]
    return tt[:, 0] + u[:, None] * (tt[:, 1] - tt[:, 0]) + v[:, None] * (tt[:, 2] - tt[:, 0]), idx


def match_corners(pred, gt, dist):
    """Greedy one-to-one matching by distance."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, []
    D = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    used_p, used_g, errs = set(), set(), []
    for k in np.argsort(D, axis=None):
        i, j = np.unravel_index(k, D.shape)
        if D[i, j] > dist:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); errs.append(float(D[i, j]))
    return len(errs), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="room_*.json from room_polygon.py")
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--dist", type=float, default=0.10)
    ap.add_argument("--corner_tol", type=float, default=0.10,
                    help="Douglas-Peucker tolerance for the GT outline (m)")
    ap.add_argument("--corner_dist", type=float, default=0.20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    r = json.load(open(a.room))
    up, e1, e2 = (np.asarray(r[k]) for k in ("up", "e1", "e2"))
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    shape = tuple(r["raster"]["shape"])
    floor, ceil = float(r["floor_height"]), float(r["ceiling_height"])
    poly = np.asarray(r["polygon"])
    to2 = lambda X: np.stack([X @ e1, X @ e2], -1)

    verts, tris, oid, cls = load_semantic(a.semantic)

    # ---- GT outline ----
    ceil_m = raster_tris(to2(verts[tris[cls == "ceiling"]]), lo, res, shape)
    cs, _ = cv2.findContours(ceil_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cs, key=cv2.contourArea)
    gt_poly = cv2.approxPolyDP(cnt, a.corner_tol / res, True)[:, 0, :] * res + lo
    gt_ceil = float(np.median(verts[tris[cls == "ceiling"]].mean(1) @ up))

    # ---- 2-D: boundary Chamfer and corners ----
    bp, bg = sample_boundary(poly, 0.01), sample_boundary(gt_poly, 0.01)
    d_pg = cKDTree(bg).query(bp)[0]
    d_gp = cKDTree(bp).query(bg)[0]
    chamfer = 0.5 * (d_pg.mean() + d_gp.mean())
    n_ok, errs = match_corners(poly, gt_poly, a.corner_dist)

    # room orientation: length-weighted 90-degree-periodic mean edge direction
    def orient(P):
        e = np.roll(P, -1, axis=0) - P
        return np.angle(np.sum(np.linalg.norm(e, axis=1) * np.exp(4j * np.arctan2(e[:, 1], e[:, 0])))) / 4
    rot_err = abs(np.degrees(np.angle(np.exp(4j * (orient(poly) - orient(gt_poly)))) / 4))

    # ---- wall-backed perimeter ----
    lines = r.get("lines", [])
    backed = total = 0.0
    for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
        L = float(np.linalg.norm(p1 - p0))
        total += L
        if L < 1e-6:
            continue
        en = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / L
        mid = 0.5 * (p0 + p1)
        best = None
        for ln in lines:
            n = np.asarray(ln["n"])
            if abs(float(n @ en)) < np.cos(np.radians(10)):
                continue
            d = abs(float((mid - np.asarray(ln["p"])) @ n))
            if d < a.dist and (best is None or d < best[0]):
                best = (d, ln["wall_id"])
        if best is not None and best[1] is not None:
            backed += L

    # ---- 3-D: extruded walls vs GT wall surface ----
    wall_t = verts[tris[cls == "wall"]]
    wall_oid = oid[cls == "wall"]
    S, idx = sample_triangles(wall_t, 400.0, rng)
    s_oid = wall_oid[idx]
    contour = np.round((poly - lo) / res).astype(np.int32).reshape(-1, 1, 2)
    s2 = (to2(S) - lo) / res
    d2 = np.array([abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
                   for x, y in s2]) * res
    hs = S @ up
    dh = np.maximum(0, np.maximum(floor - hs, hs - ceil))
    hit = np.hypot(d2, dh) < a.dist
    recall_3d = float(hit.mean())

    # per-instance (instances under 0.5 m^2 are degenerate labels, as elsewhere)
    t_area = 0.5 * np.linalg.norm(np.cross(wall_t[:, 1] - wall_t[:, 0], wall_t[:, 2] - wall_t[:, 0]), axis=1)
    inst = [i for i in np.unique(wall_oid) if t_area[wall_oid == i].sum() >= 0.5]
    inst_rec = {int(i): float(hit[s_oid == i].mean()) for i in inst if (s_oid == i).any()}
    n_inst = sum(v >= 0.5 for v in inst_rec.values())

    # observed baseline: the points of the planes judged walls
    xyz, lab = load_labeled(r["labeled"])
    wall_ids = json.load(open(r["walls"]))["wall_ids"]
    W = xyz[np.isin(lab, wall_ids)]
    if len(W):
        obs_hit = cKDTree(W).query(S, distance_upper_bound=a.dist)[0] < a.dist
    else:
        obs_hit = np.zeros(len(S), bool)
    recall_obs = float(obs_hit.mean())
    inst_obs = sum(obs_hit[s_oid == i].mean() >= 0.5 for i in inst if (s_oid == i).any())

    out = {"room": a.room,
           "chamfer_cm": 100 * chamfer, "chamfer_p90_cm": 100 * float(np.percentile(np.r_[d_pg, d_gp], 90)),
           "corners_pred": int(len(poly)), "corners_gt": int(len(gt_poly)),
           "corner_matched": int(n_ok), "corner_precision": n_ok / max(len(poly), 1),
           "corner_recall": n_ok / max(len(gt_poly), 1),
           "corner_err_cm": 100 * float(np.mean(errs)) if errs else None,
           "wall_backed": backed / max(total, 1e-9), "rotation_err_deg": float(rot_err),
           "ceiling_err_cm": 100 * (ceil - gt_ceil),
           "wall_recall_3d": recall_3d, "wall_recall_observed": recall_obs,
           "wall_instances": len(inst), "inst_3d": int(n_inst), "inst_observed": int(inst_obs),
           "inst_recall_3d": inst_rec}
    print(f"2-D  chamfer {out['chamfer_cm']:.1f} cm (p90 {out['chamfer_p90_cm']:.1f}) | corners "
          f"{n_ok}/{len(gt_poly)} GT, {n_ok}/{len(poly)} pred, err "
          f"{out['corner_err_cm'] if errs else float('nan'):.1f} cm | rotation {rot_err:.1f} deg | "
          f"wall-backed perimeter {out['wall_backed']:.0%}")
    print(f"3-D  GT wall area within {a.dist} m: extruded {recall_3d:.1%} vs observed {recall_obs:.1%} | "
          f"instances >=50%: {n_inst} vs {inst_obs} of {len(inst)} | "
          f"ceiling {out['ceiling_err_cm']:+.1f} cm")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
