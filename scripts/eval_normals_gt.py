"""
Ground-truth normal accuracy for a trained BBSplat model, against a Replica mesh.

For every billboard we take the closest point on the ground-truth mesh and
compare that triangle's normal with the billboard's own stored normal (and,
optionally, with the local-PCA normal that `--method pca` segmentation uses).
Normals are sign-ambiguous, so the error is acos(|dot|) in [0, 90].

  python scripts/eval_normals_gt.py -m output/replica_room0_v1 \
      --mesh ../data/replica/cull_replica_mesh/room0.ply [--pca] [--json out.json]
"""
import argparse, json, os, sys
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_planes import load_billboards


def angdeg(a, b):
    """Unsigned angle between unit-ish vectors, in degrees."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), 0.0, 1.0)))


def summarize(err):
    return {"n": int(err.size), "mean": float(err.mean()), "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)),
            "frac_under_10": float((err < 10).mean()), "frac_under_20": float((err < 20).mean()),
            "frac_under_30": float((err < 30).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--iteration", type=int, default=32000)
    ap.add_argument("--max_dist", type=float, default=0.10,
                    help="billboards farther than this from the mesh are floaters, "
                         "reported separately rather than scored")
    ap.add_argument("--unseen", default=None,
                    help="<scene>_pc_unseen.npy; billboards near these points are excluded")
    ap.add_argument("--unseen_radius", type=float, default=0.20)
    ap.add_argument("--pca", action="store_true", help="also score local-PCA normals")
    ap.add_argument("--depth_cloud", default=None,
                    help="score the depth-fused cloud's normals instead of the "
                         "billboards' (scripts/depth_fusion.py .npz)")
    ap.add_argument("--k_normals", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.depth_cloud:
        d = np.load(a.depth_cloud)
        xyz, own = d["xyz"].astype(np.float32), d["normal"].astype(np.float32)
        print(f"{len(xyz):,} depth-fused points from {a.depth_cloud}")
        own_label = "depth_map"
    else:
        ply = os.path.join(a.model, "point_cloud", f"iteration_{a.iteration}",
                           "point_cloud.ply")
        xyz, own = load_billboards(ply)
        print(f"{len(xyz):,} billboards from {ply}")
        own_label = "own_quat"

    mesh = o3d.io.read_triangle_mesh(a.mesh)
    mesh.compute_triangle_normals()
    tri_n = np.asarray(mesh.triangle_normals)
    print(f"GT mesh: {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    q = o3d.core.Tensor(xyz.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    res = scene.compute_closest_points(q)
    closest = res["points"].numpy()
    prim = res["primitive_ids"].numpy().astype(np.int64)
    dist = np.linalg.norm(xyz - closest, axis=1)

    keep = dist <= a.max_dist
    print(f"within {a.max_dist} m of the GT surface: {keep.sum():,} "
          f"({100*keep.mean():.1f}%)  -- the rest are floaters")

    if a.unseen and os.path.exists(a.unseen):
        unseen = np.load(a.unseen)
        if unseen.ndim == 2 and unseen.shape[1] == 3 and len(unseen):
            from scipy.spatial import cKDTree
            d_un, _ = cKDTree(unseen).query(xyz, k=1, workers=-1)
            drop = d_un < a.unseen_radius
            print(f"excluding {int((keep & drop).sum()):,} billboards in never-observed regions")
            keep &= ~drop

    gt = tri_n[prim]
    # Which way is up?  Floor plus ceiling dominate the area of an indoor mesh,
    # so the axis carrying the most area-weighted normal mass is the up axis.
    tri = np.asarray(mesh.triangles)
    v = np.asarray(mesh.vertices)
    area = 0.5 * np.linalg.norm(np.cross(v[tri[:, 1]] - v[tri[:, 0]],
                                         v[tri[:, 2]] - v[tri[:, 0]]), axis=1)
    mass = (np.abs(tri_n) * area[:, None]).sum(0)
    up_axis = int(np.argmax(mass))
    print(f"area-weighted normal mass per axis: {np.round(mass / mass.sum(), 3)} "
          f"-> up axis = {'xyz'[up_axis]}")

    vert = np.abs(gt[:, up_axis])
    cls = np.where(vert > np.cos(np.radians(20)), "horizontal",
                   np.where(vert < np.cos(np.radians(70)), "vertical", "slanted"))

    sources = {own_label: own}
    if a.pca:
        from scipy.spatial import cKDTree
        pts = xyz.astype(np.float64)
        _, nbr = cKDTree(pts).query(pts, k=a.k_normals, workers=-1)
        loc = pts[nbr] - pts[nbr].mean(1, keepdims=True)
        sources["local_pca"] = np.linalg.eigh(np.einsum('nki,nkj->nij', loc, loc))[1][:, :, 0]

    out = {"model": a.model, "mesh": a.mesh, "n_points": int(len(xyz)),
           "source": a.depth_cloud or "billboards",
           "frac_within_surface": float(keep.mean()), "up_axis": "xyz"[up_axis],
           "max_dist": a.max_dist, "results": {}}
    for sname, nrm in sources.items():
        err = angdeg(nrm[keep], gt[keep])
        print(f"\n--- {sname} (GT = closest-triangle normal)")
        hdr = f"{'subset':>12} {'n':>9} {'mean':>7} {'median':>7} {'p90':>7} {'<10d':>7} {'<20d':>7} {'<30d':>7}"
        print(hdr)
        rows = {"all": err}
        for c in ("horizontal", "vertical", "slanted"):
            msk = cls[keep] == c
            if msk.sum():
                rows[c] = err[msk]
        for rn, re_ in rows.items():
            s = summarize(re_)
            print(f"{rn:>12} {s['n']:>9,} {s['mean']:>6.1f}° {s['median']:>6.1f}° "
                  f"{s['p90']:>6.1f}° {100*s['frac_under_10']:>6.1f}% "
                  f"{100*s['frac_under_20']:>6.1f}% {100*s['frac_under_30']:>6.1f}%")
        out["results"][sname] = {k: summarize(v) for k, v in rows.items()}
    print("\n(a random normal direction averages ~57 deg under this metric)")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
