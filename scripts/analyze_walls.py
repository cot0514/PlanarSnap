"""
Room-structure summary of a plane segmentation: which planes are architectural
walls, how many distinct wall orientations the scene yields, and how many of
those orientation pairs are perpendicular.

Run on the labelled cloud written by `segment_planes.py --save`.

  python scripts/analyze_walls.py --labeled <...>/planes.ply \
      --cameras output/drjohnson_v1/cameras.json

The up direction comes from the camera poses (mean of -R[:,1] over the training
views), not from the coordinate axes: a COLMAP frame is not gravity-aligned, so
"height" along x/y/z is meaningless.  Guessing up from the plane normals alone
does not work either -- a room has many more wall planes than floor planes, so
the walls outvote the floor.

The camera estimate is only a seed, though: handheld trajectories look slightly
down and roll, so the mean camera up sits 4-9 deg off gravity on Replica, which
over a 5 m room moves "height" by up to 0.6 m.  `refine_up` then tightens it
from the planes themselves -- up should be parallel to the horizontal planes'
normals and perpendicular to the vertical ones' -- which brings it to within
0.1-3.2 deg of the GT mesh's up on all five Replica scenes and every point set.

The wall test is the one the README documents: a plane counts as an
architectural wall when it is near-vertical, reaches down close to the floor,
spans enough height, and is long enough to be a wall rather than a furniture
face.  Orientations are yaw angles in the horizontal plane, clustered mod 180
because a wall normal's sign is arbitrary.
"""
import argparse, json, os
import numpy as np
from plyfile import PlyData


def load_labeled(path):
    d = PlyData.read(path).elements[0].data
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    names = d.dtype.names
    key = next((k for k in ("plane_id", "label", "segment", "scalar_label") if k in names), None)
    if key is None:
        raise SystemExit(f"no label field in {path}; found {names}")
    return xyz, np.asarray(d[key]).astype(np.int64)


def up_from_cameras(path):
    """Mean camera up: world -R[:,1], since COLMAP's camera y axis points down."""
    with open(path) as f:
        cams = json.load(f)
    ups = []
    for c in cams:
        R = np.array(c["rotation"], dtype=np.float64).reshape(3, 3)
        ups.append(-R[:, 1])
    u = np.mean(ups, axis=0)
    n = np.linalg.norm(u)
    if n < 0.3:
        raise SystemExit(f"camera up vectors disagree (|mean| = {n:.2f}); pass --up_vec")
    return u / n, n


def up_auto(fits):
    """
    Fallback: the axis that the most points' plane normals align with *tightly*.
    Counting exact alignment (within 10 deg) rather than summing |n| keeps the
    many obliquely-oriented wall planes from outvoting the floor and ceiling.
    """
    best, score = 0, -1.0
    for ax in range(3):
        e = np.zeros(3); e[ax] = 1.0
        s = sum(len(q) for _, n, q in fits.values()
                if abs(float(n @ e)) > np.cos(np.radians(10)))
        if s > score:
            best, score = ax, s
    e = np.zeros(3); e[best] = 1.0
    return e, score


def plane_area(q, cell=0.1):
    """Area a plane covers, as occupied cells of its own 2-D basis.  A point
    count is not an area: billboard centres and fused depth pixels sample
    surfaces at very different densities."""
    c = q.mean(0)
    vt = np.linalg.svd(q - c, full_matrices=False)[2]
    k = np.floor(np.stack([(q - c) @ vt[0], (q - c) @ vt[1]], 1) / cell).astype(np.int64)
    return len(np.unique(k, axis=0)) * cell * cell


def refine_up(up0, normals, areas, schedule=(12, 8, 5, 3)):
    """
    Up as the direction most parallel to the horizontal planes' normals and
    most perpendicular to the vertical planes', area-weighted: the smallest
    eigenvector of  sum_vert A n n^T - sum_horiz A n n^T.  Which planes count
    as horizontal or vertical depends on up itself, so the tolerance shrinks
    over a few rounds starting from the camera seed; a single wide round lets
    tilted furniture and stair-like fragments pull it off (mean error 1.4 deg
    vs 0.8 deg coarse-to-fine on Replica).
    """
    up = up0.copy()
    for tol in schedule:
        M = np.zeros((3, 3))
        for n, w in zip(normals, areas):
            c = abs(float(n @ up))
            if c > np.cos(np.radians(tol)):
                M -= w * np.outer(n, n)
            elif c < np.sin(np.radians(tol)):
                M += w * np.outer(n, n)
        u = np.linalg.eigh(M)[1][:, 0]
        up = u if u @ up0 > 0 else -u
    return up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--cameras", default=None,
                    help="cameras.json from the model dir (preferred up source)")
    ap.add_argument("--up_vec", default=None, help="explicit up, e.g. '0,1,0'")
    ap.add_argument("--vertical_deg", type=float, default=20.0,
                    help="a wall normal must be within this of horizontal")
    ap.add_argument("--floor_gap", type=float, default=0.7)
    ap.add_argument("--min_height", type=float, default=1.5)
    ap.add_argument("--min_length", type=float, default=2.0)
    ap.add_argument("--yaw_tol", type=float, default=10.0)
    ap.add_argument("--perp_tol", type=float, default=10.0)
    ap.add_argument("--no_refine_up", action="store_true",
                    help="use the camera (or given) up as is, the old behaviour")
    ap.add_argument("--floor_min_area", type=float, default=1.0,
                    help="smallest horizontal plane (m^2) that may be the floor")
    ap.add_argument("--floor_cap_margin", type=float, default=0.15,
                    help="replace the floor plane by the wall-bottom bound only "
                         "when the plane sits this far above it; wall bottoms "
                         "scatter a few cm under a correctly found floor "
                         "(drjohnson: 7-14 cm)")
    ap.add_argument("--floor_height", type=float, default=None,
                    help="use this floor height (along up) instead of estimating "
                         "it. Passing one shared value lets different point sets "
                         "be judged against the same floor")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    xyz, lab = load_labeled(a.labeled)
    ids = np.array(sorted(i for i in np.unique(lab) if i >= 0))
    print(f"{len(xyz):,} points, {len(ids)} planes")

    fits = {}
    for i in ids:
        q = xyz[lab == i]
        c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        fits[i] = (c, n / np.linalg.norm(n), q)
    areas = {i: plane_area(q) for i, (_, _, q) in fits.items()}

    if a.up_vec:
        up = np.array([float(x) for x in a.up_vec.split(",")]); up /= np.linalg.norm(up)
        print(f"up (given)      = {np.round(up,3)}")
    elif a.cameras and os.path.exists(a.cameras):
        up, agree = up_from_cameras(a.cameras)
        print(f"up (cameras)    = {np.round(up,3)}   (|mean of {os.path.basename(a.cameras)}| = {agree:.2f})")
    else:
        up, _ = up_auto(fits)
        print(f"up (auto, axis) = {np.round(up,3)}   -- pass --cameras for a better estimate")
    up_seed = up
    if not a.no_refine_up:
        up = refine_up(up_seed, [fits[i][1] for i in ids], [areas[i] for i in ids])
        tilt = np.degrees(np.arccos(min(1.0, abs(float(up @ up_seed)))))
        print(f"up (refined)    = {np.round(up,3)}   ({tilt:.1f} deg from the seed)")

    # orthonormal basis of the horizontal plane, for heights and yaw angles
    seed = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = seed - up * (seed @ up); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    heights = {i: q @ up for i, (_, _, q) in fits.items()}

    # Floor height.  Three things go wrong with simpler rules:
    #   * the minimum over all planes -- one spurious plane below the floor
    #     (drjohnson, 1.9 m under it at a 5 cm fusion voxel) rejects every wall;
    #   * the largest low horizontal plane -- when the floor is hidden or
    #     sparsely covered it is a table or bed top (0.7-0.9 m high on room1
    #     billboards, 0.3 m on office0 depth);
    #   * the floor plane may not exist at all: office1's rendered depth holds
    #     ~2 m^2 of floor spread over +-0.3 m, which never becomes a plane.
    # So: the LOWEST horizontal plane of real size (>= floor_min_area and a
    # quarter of the largest one, which keeps small spurious planes from below
    # the floor out), capped by where the large vertical planes end -- walls
    # stand on the floor, so it cannot be well above most of their bottoms.
    all_h = np.concatenate([h for h in heights.values()])
    lo, hi = np.percentile(all_h, 0.5), np.percentile(all_h, 99.5)
    mid = 0.5 * (lo + hi)
    cos_h = np.cos(np.radians(a.vertical_deg))
    cands = [(areas[i], int(i), float(np.median(heights[i])))
             for i, (c, n, q) in fits.items()
             if abs(float(n @ up)) > cos_h and np.median(heights[i]) < mid]
    if cands:
        min_a = max(a.floor_min_area, 0.25 * max(c[0] for c in cands))
        cands = [c for c in cands if c[0] >= min_a]
    bottoms = [float(np.percentile(heights[i], 2)) for i, (c, n, q) in fits.items()
               if abs(float(n @ up)) < np.sin(np.radians(a.vertical_deg))
               and areas[i] >= a.floor_min_area]
    wall_cap = float(np.percentile(bottoms, 10)) if bottoms else None
    floor_src = "given"
    if a.floor_height is not None:
        floor_h, floor_id = float(a.floor_height), None
        print(f"floor height given: {floor_h:.2f}")
    elif cands:
        floor_a, floor_id, floor_h = min(cands, key=lambda c: c[2])
        floor_src = "plane"
        print(f"floor plane = id {floor_id} ({floor_a:.1f} m^2), "
              f"median height {floor_h:.2f}")
    else:
        floor_h = float(np.percentile(all_h, 1.0))
        floor_id, floor_src = None, "percentile"
        print(f"no horizontal plane found low in the scene; "
              f"using the 1st percentile of all heights, {floor_h:.2f}")
    if a.floor_height is None and wall_cap is not None \
            and floor_h - wall_cap > a.floor_cap_margin:
        print(f"  ...but {len(bottoms)} large vertical planes end lower (10th pct "
              f"of their bottoms {wall_cap:.2f}); the floor plane is furniture or "
              f"a platform, using {wall_cap:.2f}")
        floor_h, floor_id, floor_src = wall_cap, None, "wall_bottoms"
    print(f"scene height range {lo:.2f} .. {hi:.2f} along up")

    walls, rows = [], []
    cos_v = np.cos(np.radians(90 - a.vertical_deg))
    for i, (c, n, q) in fits.items():
        h = heights[i]
        is_vert = abs(float(n @ up)) < cos_v
        h_lo, h_hi = float(np.percentile(h, 5)), float(np.percentile(h, 95))
        height = h_hi - h_lo
        ext = np.stack([q @ e1, q @ e2], 1)
        length = float(np.linalg.norm(np.percentile(ext, 97.5, axis=0)
                                      - np.percentile(ext, 2.5, axis=0)))
        # percentiles, not extremes, so that a handful of stray points cannot
        # make a small furniture face look like a floor-to-ceiling wall
        reaches = (h_lo - floor_h) < a.floor_gap
        ok = bool(is_vert and reaches and height > a.min_height and length > a.min_length)
        yaw = float(np.degrees(np.arctan2(n @ e2, n @ e1)) % 180.0)
        rows.append((len(q), int(i), is_vert, reaches, height, length, yaw, ok))
        if ok:
            walls.append((int(i), yaw, len(q)))

    rows.sort(reverse=True)
    print(f"\n{'id':>4} {'pts':>9} {'vert':>5} {'floor':>6} {'height':>7} "
          f"{'length':>7} {'yaw':>6}  wall")
    for cnt, i, iv, rc, hh, l, y, ok in rows[:30]:
        print(f"{i:>4} {cnt:>9,} {str(iv):>5} {str(rc):>6} {hh:>7.2f} {l:>7.2f} "
              f"{y:>5.1f}°  {'YES' if ok else ''}")

    clusters = []
    for i, y, c in sorted(walls, key=lambda w: -w[2]):
        for cl in clusters:
            d = abs(y - cl["yaw"]) % 180
            if min(d, 180 - d) <= a.yaw_tol:
                cl["ids"].append(i); break
        else:
            clusters.append({"yaw": y, "ids": [i]})

    perp = []
    for x in range(len(clusters)):
        for y2 in range(x + 1, len(clusters)):
            d = abs(clusters[x]["yaw"] - clusters[y2]["yaw"]) % 180
            if abs(min(d, 180 - d) - 90) <= a.perp_tol:
                perp.append((round(clusters[x]["yaw"], 1), round(clusters[y2]["yaw"], 1)))

    # Pairs of individual walls that are mutually perpendicular.  Reported
    # alongside the orientation-pair count because they are different numbers:
    # two perpendicular orientation families holding 5 and 2 walls give 1
    # orientation pair but 10 wall pairs.
    wall_pairs = 0
    for x in range(len(walls)):
        for y2 in range(x + 1, len(walls)):
            d = abs(walls[x][1] - walls[y2][1]) % 180
            if abs(min(d, 180 - d) - 90) <= a.perp_tol:
                wall_pairs += 1

    print(f"\narchitectural walls   : {len(walls)}")
    print(f"distinct orientations : {len(clusters)}")
    for cl in clusters:
        print(f"   yaw {cl['yaw']:5.1f}°  walls {cl['ids']}")
    print(f"perpendicular orientation pairs : {len(perp)}  {perp}")
    print(f"perpendicular wall pairs        : {wall_pairs}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"labeled": a.labeled, "n_points": len(xyz), "n_planes": len(ids),
                       "up": up.tolist(), "up_seed": up_seed.tolist(),
                       "floor_height": floor_h, "floor_plane_id": floor_id,
                       "floor_source": floor_src, "walls": len(walls),
                       "wall_ids": [int(i) for i, _, _ in walls],
                       "orientations": [{"yaw": c["yaw"], "ids": c["ids"]} for c in clusters],
                       "perpendicular_orientation_pairs": len(perp),
                       "perpendicular_wall_pairs": wall_pairs}, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
