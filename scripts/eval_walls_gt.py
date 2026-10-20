"""
Wall / floor / ceiling detection accuracy against Replica's semantic ground truth.

Replica ships a per-face instance mesh (`mesh_semantic.ply`, object_id per
quad) plus `info_semantic.json` mapping instances to 101 classes.  Its vertices
are bit-identical to `cull_replica_mesh/<scene>.ply`, so no registration is
needed.

  python scripts/eval_walls_gt.py \
      --labeled output/replica_room0_v1/point_cloud/iteration_32000/planes_depth.ply \
      --walls   output/replica_room0_v1/walls_depth.json \
      --semantic .../official/semantic/room_0/habitat

Reports two things:
  * point level  -- of the points we call wall, how many are on a GT wall
                    (precision), and of the GT wall points we hold, how many we
                    call wall (recall)
  * instance level -- how many of the scene's GT wall instances we recover, and
                    how many of our wall planes sit on a real wall.  This is
                    the number that answers "the room has 5 walls; how many did
                    we find?"
"""
import argparse, json, os
import numpy as np
import open3d as o3d
from plyfile import PlyData


def load_semantic(folder):
    """Triangles + per-triangle class name, from Replica's instance mesh."""
    info = json.load(open(os.path.join(folder, "info_semantic.json")))
    cls_name = {c["id"]: c["name"] for c in info["classes"]}
    # id_to_label[object_id] = class id (negative or 0 means unlabelled)
    id2label = np.asarray(info["id_to_label"])

    p = PlyData.read(os.path.join(folder, "mesh_semantic.ply"))
    v = p["vertex"].data
    verts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    f = p["face"].data
    quads = np.vstack(f["vertex_indices"]).astype(np.int64)   # all quads
    oid = np.asarray(f["object_id"]).astype(np.int64)
    # split each quad into two triangles, both inheriting the instance id
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    tri_oid = np.concatenate([oid, oid])

    lab = np.where((tri_oid >= 0) & (tri_oid < len(id2label)), id2label[tri_oid], -1)
    names = np.array([cls_name.get(int(l), "unlabelled") for l in lab])
    return verts, tris, tri_oid, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, help="planes_*.ply from segment_planes --save")
    ap.add_argument("--walls", required=True, help="walls_*.json from analyze_walls --json")
    ap.add_argument("--semantic", required=True, help=".../habitat folder")
    ap.add_argument("--max_dist", type=float, default=0.10,
                    help="points farther than this from the GT surface are "
                         "floaters and are left out of the scoring")
    ap.add_argument("--min_overlap", type=float, default=0.25,
                    help="fraction of a GT wall instance's held points that a "
                         "predicted plane must cover to count as recovering it")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    d = PlyData.read(a.labeled).elements[0].data
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    lab = np.asarray(d["plane_id"]).astype(np.int64)
    wall_ids = set(json.load(open(a.walls))["wall_ids"])
    print(f"{len(xyz):,} points; {len(wall_ids)} planes predicted as walls")

    verts, tris, tri_oid, tri_cls = load_semantic(a.semantic)
    print(f"GT semantic mesh: {len(verts):,} verts, {len(tris):,} triangles, "
          f"{len(np.unique(tri_oid))} instances")

    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(verts.astype(np.float32))
    m.triangle.indices = o3d.core.Tensor(tris.astype(np.int32))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(m)
    res = sc.compute_closest_points(o3d.core.Tensor(xyz.astype(np.float32)))
    prim = res["primitive_ids"].numpy().astype(np.int64)
    dist = np.linalg.norm(xyz - res["points"].numpy(), axis=1)

    keep = (dist <= a.max_dist) & (lab >= 0)
    print(f"scored: {keep.sum():,} points ({100*keep.mean():.1f}%) that are "
          f"assigned to a plane and within {a.max_dist} m of the GT surface")

    gt_cls = tri_cls[prim]
    gt_oid = tri_oid[prim]
    pred_wall = np.isin(lab, list(wall_ids))

    out = {"labeled": a.labeled, "n_points": int(len(xyz)), "n_scored": int(keep.sum())}
    print(f"\n{'class':>10} {'GT pts':>10} {'pred pts':>10} {'prec':>7} {'recall':>7} {'IoU':>7}")
    for cname in ("wall", "floor", "ceiling"):
        g = keep & (gt_cls == cname)
        pr = keep & pred_wall if cname == "wall" else None
        if cname != "wall":
            # we only have an explicit wall decision; for floor/ceiling report
            # how the GT points distribute instead of a precision we cannot form
            print(f"{cname:>10} {g.sum():>10,} {'-':>10} {'-':>7} {'-':>7} {'-':>7}")
            out[cname] = {"gt_points": int(g.sum())}
            continue
        tp = int((g & pr).sum()); fp = int((~g & pr & keep).sum()); fn = int((g & ~pr).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        print(f"{cname:>10} {g.sum():>10,} {pr.sum():>10,} {prec:>6.1%} {rec:>6.1%} {iou:>6.1%}")
        out[cname] = {"gt_points": int(g.sum()), "pred_points": int(pr.sum()),
                      "tp": tp, "fp": fp, "fn": fn,
                      "precision": prec, "recall": rec, "iou": iou}

    # ---- instance level ----
    gt_wall = keep & (gt_cls == "wall")
    inst, inst_n = np.unique(gt_oid[gt_wall], return_counts=True)
    # The denominator is the number of wall instances the SCENE has, not the
    # number that happen to appear among this arm's scored points -- otherwise
    # an arm that covers less of the room gets an easier denominator and the
    # arms cannot be compared.
    scene_inst = np.unique(tri_oid[tri_cls == "wall"])
    print(f"\nGT wall instances: {len(scene_inst)} in the scene, "
          f"{len(inst)} touched by this arm's scored points")
    recovered, assign = [], {}
    for pid in sorted(wall_ids):
        msk = keep & (lab == pid)
        if not msk.any():
            continue
        names, counts = np.unique(gt_cls[msk], return_counts=True)
        dom = names[counts.argmax()]
        assign[pid] = (dom, int(counts.max()), int(msk.sum()))
    n_on_wall = sum(1 for v in assign.values() if v[0] == "wall")
    print(f"predicted wall planes whose majority GT class is 'wall': "
          f"{n_on_wall}/{len(assign)}  ({n_on_wall/max(len(assign),1):.0%})")
    for i, n in zip(inst, inst_n):
        best = 0.0
        for pid in sorted(wall_ids):
            msk = keep & (lab == pid) & (gt_oid == i)
            best = max(best, msk.sum() / n)
        if best >= a.min_overlap:
            recovered.append(int(i))
        print(f"  instance {int(i):>3}: {n:>8,} pts, best single plane covers {best:>5.1%}"
              f"{'   RECOVERED' if best >= a.min_overlap else ''}")
    print(f"\nGT wall instances recovered: {len(recovered)}/{len(scene_inst)}")
    out["instances"] = {"gt_wall_instances": int(len(scene_inst)),
                        "gt_instances_touched": int(len(inst)),
                        "recovered": len(recovered),
                        "pred_wall_planes": len(assign),
                        "pred_planes_on_wall": n_on_wall,
                        "assignment": {str(k): v for k, v in assign.items()}}

    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
