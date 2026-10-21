"""
Room footprint accuracy against Replica's semantic mesh.

  python scripts/eval_footprint_gt.py --room <model>/room_depth_pca_cons.json \
      --semantic .../official/semantic/room_0/habitat

GT footprint = the `ceiling` triangles projected straight down.  The ceiling
is the one surface nothing hides from above, so its projection is the room's
real floor area -- the floor itself is partly under furniture and rugs (and
office0's `floor` label covers 1.3 m^2 of a 16 m^2 room).  Holes in it are
filled: lights and vents are their own classes and punch 1.6-3.0 m^2 out of
the ceiling, which capped even the oracle's IoU at 89%; filled, it is 98%.  Both footprints
are rasterised in the room's own horizontal basis and compared as IoU.

Two baselines from the same labelled cloud, neither using the walls:
  * visible floor -- points within --floor_band of the estimated floor,
                     splatted at 10 cm
  * oriented box  -- the 0.5-99.5 percentile extent of all points above the
                     floor along the polygon's two dominant directions.  Most
                     Replica rooms are rectangles, so this is a strong
                     baseline and the polygon has to beat it to matter.
"""
import argparse, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_walls_gt import load_semantic
from analyze_walls import load_labeled


def raster_tris(tris2d, lo, res, shape):
    m = np.zeros(shape, np.uint8)
    px = np.round((tris2d - lo) / res).astype(np.int32)
    cv2.fillPoly(m, list(px), 1)
    return m


def iou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i) / max(float(u), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="room_*.json from room_polygon.py")
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--floor_band", type=float, default=0.10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    r = json.load(open(a.room))
    e1, e2 = np.asarray(r["e1"]), np.asarray(r["e2"])
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    shape = tuple(r["raster"]["shape"])
    pred = np.load(os.path.splitext(a.room)[0] + "_mask.npy") > 0

    verts, tris, _, cls = load_semantic(a.semantic)
    t = verts[tris[cls == "ceiling"]]
    t2 = np.stack([t @ e1, t @ e2], -1)
    ceil = raster_tris(t2, lo, res, shape)
    cs, _ = cv2.findContours(ceil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt = np.zeros_like(ceil)
    cv2.drawContours(gt, cs, -1, 1, -1)
    gt = gt > 0
    # a few ceiling triangles may fall outside the raster frame; report it
    t_all = t2.reshape(-1, 2)
    outside = np.mean(np.any((t_all < lo) | (t_all > lo + res * np.array(shape[::-1])), 1))

    # visible-floor baseline from the same labelled cloud
    xyz, _ = load_labeled(r["labeled"])
    up = np.asarray(r["up"])
    band = np.abs(xyz @ up - r["floor_height"]) < a.floor_band
    vf = np.zeros(shape, np.uint8)
    px = np.round((np.stack([xyz[band] @ e1, xyz[band] @ e2], 1) - lo) / res).astype(np.int64)
    ok = (px[:, 0] >= 0) & (px[:, 0] < shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < shape[0])
    vf[px[ok, 1], px[ok, 0]] = 1
    k = max(1, int(round(0.10 / res)))
    vf = cv2.dilate(vf, np.ones((k, k), np.uint8)) > 0

    # oriented-box baseline
    above = xyz @ up > r["floor_height"] - 0.2
    Q = np.stack([xyz[above] @ e1, xyz[above] @ e2], 1)
    box = np.zeros(shape, np.uint8)
    d0, d1 = (np.asarray(d) for d in r["dirs"])
    a0, a1 = np.percentile(Q @ d0, [0.5, 99.5]), np.percentile(Q @ d1, [0.5, 99.5])
    corners = np.array([u * d0 + v * d1 for u, v in
                        ((a0[0], a1[0]), (a0[1], a1[0]), (a0[1], a1[1]), (a0[0], a1[1]))])
    cv2.fillPoly(box, [np.round((corners - lo) / res).astype(np.int32)], 1)
    box = box > 0

    A = res * res
    out = {"room": a.room, "gt_area": float(gt.sum() * A), "pred_area": float(pred.sum() * A),
           "iou": iou(pred, gt),
           "precision": float(np.logical_and(pred, gt).sum()) / max(float(pred.sum()), 1.0),
           "recall": float(np.logical_and(pred, gt).sum()) / max(float(gt.sum()), 1.0),
           "visible_floor_area": float(vf.sum() * A), "visible_floor_iou": iou(vf, gt),
           "box_area": float(box.sum() * A), "box_iou": iou(box, gt),
           "gt_outside_frame": float(outside)}
    print(f"GT footprint {out['gt_area']:.1f} m^2 | polygon {out['pred_area']:.1f} m^2 "
          f"IoU {out['iou']:.1%} (prec {out['precision']:.1%}, rec {out['recall']:.1%}) | "
          f"visible floor IoU {out['visible_floor_iou']:.1%} | box IoU {out['box_iou']:.1%}")
    if outside > 0:
        print(f"  warning: {outside:.1%} of GT ceiling vertices fall outside the raster frame")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
