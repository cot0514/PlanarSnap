"""
Room footprint polygon from the wall planes (floorplan step 2).

Each architectural wall (from `analyze_walls.py`) becomes a line in the
horizontal plane.  The lines, extended across the scene, cut the plane into
cells -- a line arrangement, built here on a raster so no computational-
geometry package is needed.  A cell belongs to the room when enough of it is
covered by the scene's points seen from above; the room is the union of those
cells, and its outline, simplified, is the polygon.

Walls the segmentation missed leave the room open, and an open room touches
the frame border and is rejected whole.  So the two dominant wall directions
(the largest wall's and its perpendicular; the points' principal axes when
there is no wall) get a "virtual" line at the 2nd/98th percentile of the
points along them -- where a missing wall would have to be at the latest in a
closed room -- but only on a side with no detected wall in its outer half.
Adding them everywhere hurts: rendered depth leaves floaters behind the walls,
so the virtual line lands outside the real wall and the strip between is
taken for floor (depth polygons drop from ~88% to ~78% IoU).  --no_virtual
turns them off.

The room's orientation is voted by every near-vertical plane (90-degree-
periodic mean, refined twice over the planes within 5 deg).  Voting with the
walls alone let office1's few, noisy depth walls rotate the room by 5 deg.

Depth finds the walls but places them coarsely (points 5-9 cm off the
surface, floaters behind the walls); billboards place them to ~1 cm but find
few.  --refine_cloud takes the billboards to re-place the depth walls and the
virtual lines.  Replica, 5 scenes, boundary Chamfer to the GT outline:
billboards alone 7.9 cm, hybrid+consolidated 9.6 cm, the same refined with
billboards 7.5 cm; GT wall area rebuilt by the extruded walls 63% / 64% / 75%.

Why an arrangement and not, say, the outline of the points themselves: the
walls give straight, exactly placed edges, and a cell is accepted or rejected
as a whole, so a corner hidden behind a wardrobe still comes out square as
long as the rest of its cell is observed.

  python scripts/room_polygon.py --labeled .../planes_depth_pca_cons.ply \
      --walls <model>/walls_depth_pca_cons.json --json <model>/room_depth_pca_cons.json

The output holds the polygon in the 2-D basis (e1, e2) of the horizontal
plane, the basis itself and the floor height, so the 3-D corners are
floor * up + x * e1 + y * e2.
"""
import argparse, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_walls import load_labeled, plane_area


def horizontal_basis(up):
    seed = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = seed - up * (seed @ up)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--walls", required=True, help="walls_*.json from analyze_walls")
    ap.add_argument("--json", required=True)
    ap.add_argument("--png", default=None, help="optional top-down picture")
    ap.add_argument("--res", type=float, default=0.02, help="raster cell (m)")
    ap.add_argument("--evidence_cell", type=float, default=0.10,
                    help="a point marks this much of the floor as observed (m)")
    ap.add_argument("--min_fill", type=float, default=0.5,
                    help="a cell is inside the room when this fraction of it is observed")
    ap.add_argument("--min_cell", type=float, default=0.05,
                    help="cells smaller than this (m^2) are slivers between near-"
                         "parallel lines and take their larger neighbour's label")
    ap.add_argument("--virtual", choices=["open", "all", "none"], default="open",
                    help="close the room with lines at the points' extent: only "
                         "on sides without a detected wall (open), on every side, "
                         "or never")
    ap.add_argument("--refine_cloud", default=None,
                    help="positionally accurate cloud (e.g. the billboards' "
                         "planes_bb_ransac.ply) to place wall lines and virtual lines")
    ap.add_argument("--refine_band", type=float, default=0.15)
    ap.add_argument("--dominant", choices=["walls", "vertical"], default="vertical",
                    help="planes that vote for the room's orientation: the walls, "
                         "or every near-vertical plane")
    ap.add_argument("--virtual_pct", type=float, default=2.0,
                    help="virtual lines sit at this percentile of the points "
                         "(and 100 minus it) along each dominant direction")
    ap.add_argument("--manhattan_tol", type=float, default=15.0,
                    help="snap walls within this of the dominant directions and "
                         "drop the rest; 0 keeps every wall as fitted")
    ap.add_argument("--simplify", type=float, default=0.05,
                    help="Douglas-Peucker tolerance for the outline (m)")
    a = ap.parse_args()

    w = json.load(open(a.walls))
    up = np.asarray(w["up"], dtype=np.float64)
    floor = float(w["floor_height"])
    e1, e2 = horizontal_basis(up)
    xyz, lab = load_labeled(a.labeled)
    P = np.stack([xyz @ e1, xyz @ e2], 1)
    h = xyz @ up

    # raster frame: the bulk of the points plus a margin.  The margin grows
    # with the room: in a room rotated against e1/e2 (office1, ~35 deg) the
    # lines closing a corner meet outside a tight axis-aligned frame, the room
    # cell then touches the border and is thrown out as "outside".
    lo = np.percentile(P, 0.5, axis=0)
    hi = np.percentile(P, 99.5, axis=0)
    margin = 0.5 + 0.5 * float(np.max(hi - lo))
    lo, hi = lo - margin, hi + margin
    W, H = (np.ceil((hi - lo) / a.res).astype(int) + 1)
    to_px = lambda q: np.round((q - lo) / a.res).astype(np.int64)

    # every plane's 2-D normal and size, for the dominant direction
    def normal2(i):
        q = xyz[lab == i]
        c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        n2 = np.array([n @ e1, n @ e2])
        return c, n, n2 / max(np.linalg.norm(n2), 1e-9)

    # dominant direction: the 90-degree-periodic mean of the wall normals,
    # weighted by point count (yaw*4 folds the four Manhattan directions onto
    # one).  With no wall at all, every near-vertical plane votes instead --
    # furniture faces mostly align with the room too.  The points' principal
    # axis is NOT a usable fallback: on room1's billboards it is 45 deg off.
    # The mean is then re-taken over the planes within 5 deg of it, twice: one
    # oblique depth "wall" otherwise rotates the whole room (office1, ~3 deg).
    def dominant(ids):
        yaw = {i: np.arctan2(normal2(i)[2][1], normal2(i)[2][0]) for i in ids}
        cnt = {i: int((lab == i).sum()) for i in ids}
        t = np.angle(sum(cnt[i] * np.exp(4j * yaw[i]) for i in ids)) / 4
        for _ in range(2):
            near = [i for i in ids
                    if abs(np.angle(np.exp(4j * (yaw[i] - t)))) / 4 < np.radians(5)]
            if not near:
                break
            t = np.angle(sum(cnt[i] * np.exp(4j * yaw[i]) for i in near)) / 4
        return np.array([np.cos(t), np.sin(t)])
    voters = list(w["wall_ids"]) if a.dominant == "walls" else []
    if not voters:
        voters = [i for i in np.unique(lab) if i >= 0
                  and abs(float(normal2(i)[1] @ up)) < np.sin(np.radians(20))]
    d0 = dominant(voters) if voters else np.array([1.0, 0.0])
    dirs = [d0, np.array([-d0[1], d0[0]])]

    # wall lines through each wall's centroid.  Under --manhattan a wall within
    # --manhattan_tol of a dominant direction is snapped onto it and any other
    # wall is dropped: the oblique "walls" in these scenes are mis-segmented
    # furniture or blinds, and their lines cut wedges into the room (office1).
    lines, dropped = [], []
    for i in w["wall_ids"]:
        c, _, n2 = normal2(i)
        if a.manhattan_tol > 0:
            best = max(dirs, key=lambda d: abs(float(n2 @ d)))
            if np.degrees(np.arccos(min(1.0, abs(float(n2 @ best))))) > a.manhattan_tol:
                dropped.append(int(i))
                continue
            n2 = best
        lines.append((i, np.array([c @ e1, c @ e2]), n2))
    band = (h > floor - 0.2)
    Pb = P[band]

    # --refine_cloud: a second, positionally accurate point set (the billboard
    # centres, ~1 cm off the surface against 5-9 cm for rendered depth).  Each
    # wall line keeps its direction but moves to the median offset of the
    # accurate points within --refine_band of it, along the wall's own extent;
    # and the virtual lines are placed at the accurate points' extent instead
    # of the depth cloud's, which floaters behind the walls push outward.
    if a.refine_cloud:
        R3 = load_labeled(a.refine_cloud)[0]
        hr = R3 @ up
        R2all = np.stack([R3 @ e1, R3 @ e2], 1)
        R2 = R2all[(hr > floor + 0.1) & (hr < floor + 2.0)]      # wall height band
        moved = []
        for k, (i, p, n2) in enumerate(lines):
            d = np.array([-n2[1], n2[0]])
            q = xyz[lab == i]
            t = np.stack([q @ e1, q @ e2], 1) @ d
            tr = R2 @ d
            r = (R2 - p) @ n2
            near = (np.abs(r) < a.refine_band) & (tr > np.percentile(t, 2.5)) & (tr < np.percentile(t, 97.5))
            if near.sum() >= 30:
                dr = float(np.median(r[near]))
                lines[k] = (i, p + dr * n2, n2)
                moved.append(dr)
        if moved:
            print(f"refine: moved {len(moved)} of {len(lines)} wall lines by median "
                  f"{100 * np.median(np.abs(moved)):.1f} cm")
        Pb = R2all[hr > floor - 0.2]

    # ceiling height, the mirror of analyze_walls' floor rule: the HIGHEST
    # horizontal plane of real size (>= 1 m^2 and a quarter of the largest
    # candidate) in the upper half.  Falls back to the 99.5th percentile.
    lo_h, hi_h = np.percentile(h[band], [0.5, 99.5])
    cands = []
    for i in np.unique(lab):
        if i < 0:
            continue
        c, n, _ = normal2(i)
        hi_i = h[lab == i]
        if abs(float(n @ up)) > np.cos(np.radians(20)) and np.median(hi_i) > 0.5 * (lo_h + hi_h):
            cands.append((plane_area(xyz[lab == i]), float(np.median(hi_i))))
    if cands:
        min_a = max(1.0, 0.25 * max(c[0] for c in cands))
        cands = [c for c in cands if c[0] >= min_a]
    ceiling = max(c[1] for c in cands) if cands else float(hi_h)
    virtual = []
    if a.virtual != "none":
        cos_par = np.cos(np.radians(15))
        for d in dirs:
            proj = Pb @ d
            mid = np.median(proj)
            # offsets of the walls parallel to d, in d's sign convention
            offs = [float(p @ d) for _, p, n2 in lines if abs(float(n2 @ d)) > cos_par]
            for pct, sign in ((a.virtual_pct, -1), (100 - a.virtual_pct, 1)):
                ext = float(np.percentile(proj, pct))
                outer = mid + 0.5 * (ext - mid)
                if a.virtual == "open" and any(sign * (o - outer) >= 0 for o in offs):
                    continue                     # a real wall closes this side
                virtual.append((None, ext * d, d))
    barrier = np.zeros((H, W), np.uint8)
    L = float(np.linalg.norm(hi - lo))
    for _, p, n2 in lines + virtual:
        d = np.array([-n2[1], n2[0]])
        a0, b0 = to_px(p - L * d), to_px(p + L * d)
        cv2.line(barrier, (int(a0[0]), int(a0[1])), (int(b0[0]), int(b0[1])), 1, 1)

    # cells of the arrangement = connected components of the complement
    ncell, cells = cv2.connectedComponents((1 - barrier).astype(np.uint8), connectivity=4)

    # observed footprint: every point between the floor and the ceiling,
    # splatted at evidence_cell, marks the floor beneath it as inside
    ev = np.zeros((H, W), np.uint8)
    keep = h > floor - 0.2
    px = to_px(P[keep])
    ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    ev[px[ok, 1], px[ok, 0]] = 1
    k = max(1, int(round(a.evidence_cell / a.res)))
    ev = cv2.dilate(ev, np.ones((k, k), np.uint8))

    area = np.bincount(cells.ravel(), minlength=ncell).astype(np.float64)
    seen = np.bincount(cells.ravel(), weights=ev.ravel().astype(np.float64), minlength=ncell)
    fill = seen / np.maximum(area, 1)
    inside = fill >= a.min_fill
    inside[0] = False                                   # label 0 is the barrier
    # a cell touching the frame border is outside the room by construction
    border = np.unique(np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]]))
    inside[border] = False

    # slivers between near-parallel lines: give them their neighbours' vote
    small = np.where((area * a.res ** 2 < a.min_cell) & (np.arange(ncell) > 0))[0]
    if len(small):
        dil = cv2.dilate(cells.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int64)
        for s in small:
            nb = np.unique(dil[cells == s])
            nb = nb[(nb != s) & (nb > 0)]
            if len(nb):
                big = nb[np.argmax(area[nb])]
                inside[s] = inside[big]

    room = inside[cells].astype(np.uint8)
    # the barrier pixels between two inside cells are inside too
    room = cv2.morphologyEx(room, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_comp, comp = cv2.connectedComponents(room, connectivity=4)
    if n_comp > 1:                                       # keep the largest piece
        sizes = np.bincount(comp.ravel())[1:]
        room = (comp == 1 + int(np.argmax(sizes))).astype(np.uint8)
    contours, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    room_area = float(room.sum()) * a.res ** 2
    if not contours:
        raise SystemExit("no cell passed the evidence test")
    cnt = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(cnt, a.simplify / a.res, True)[:, 0, :]
    poly_m = poly * a.res + lo

    if dropped:
        print(f"dropped {len(dropped)} oblique wall(s): {dropped}")
    print(f"{len(lines)} wall lines (+{len(virtual)} virtual), {ncell - 1} cells, {int(inside.sum())} inside; "
          f"room {room_area:.1f} m^2, polygon {len(poly_m)} corners")

    out = {"labeled": a.labeled, "walls": a.walls, "up": up.tolist(),
           "e1": e1.tolist(), "e2": e2.tolist(), "floor_height": floor,
           "polygon": poly_m.tolist(), "area": room_area,
           "raster": {"lo": lo.tolist(), "res": a.res, "shape": [int(H), int(W)]},
           "n_wall_lines": len(lines), "dirs": [d.tolist() for d in dirs],
           "ceiling_height": ceiling,
           "lines": [{"wall_id": None if i is None else int(i), "p": p.tolist(),
                      "n": n2.tolist()} for i, p, n2 in lines + virtual]}
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    np.save(os.path.splitext(a.json)[0] + "_mask.npy", room)
    print(f"wrote {a.json}")

    if a.png:
        img = np.full((H, W, 3), 255, np.uint8)
        img[ev > 0] = (225, 225, 225)
        img[room > 0] = (190, 225, 190)
        img[barrier > 0] = (200, 200, 255)
        cv2.polylines(img, [poly.reshape(-1, 1, 2).astype(np.int32)], True, (0, 120, 0), 2)
        cv2.imwrite(a.png, img[::-1])
        print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
