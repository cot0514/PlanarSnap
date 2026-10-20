"""
Refine depth-detected planes using the billboards, then re-collect inliers.

Measured on Replica room0, against the ground-truth mesh:

  billboards      median 1.4 cm from the true surface, 96.7% within 10 cm,
                  but they cluster where the image has texture, so a plain
                  wall is almost empty and no single plane spans it (the
                  tallest vertical billboard plane reached 1.7 m of a 2.8 m
                  wall, and the floor plane covered 4.7 m of a 7.8 m room)
  depth fusion    covers every rendered pixel, so walls and floor come out
                  whole, but the rendered depth sits a median 6-7.5 cm off
                  the true surface (only 59-65% within 10 cm)

So depth says WHERE the planes are and billboards say what they ARE.  This
pass takes each plane found on the depth cloud, refits it to the billboards
that agree with it, and then re-collects the depth points with the refined
plane, which tightens the point assignment without touching training.

  python scripts/hybrid_planes.py -m output/replica_room0_v1 \
      --labeled <...>/planes_depth_pca.ply --depth_cloud <...>.npz \
      --out <...>/planes_hybrid.ply
"""
import argparse, os, sys
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_planes import load_billboards, scene_extent


def fit_plane(P):
    c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    return c, n / np.linalg.norm(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--iteration", type=int, default=32000)
    ap.add_argument("--labeled", required=True, help="planes_*.ply from the depth cloud")
    ap.add_argument("--depth_cloud", required=True)
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--normal_angle", type=float, default=30.0,
                    help="billboard normals must agree with the plane within this")
    ap.add_argument("--min_billboards", type=int, default=50,
                    help="planes supported by fewer billboards keep their "
                         "depth-only fit rather than being refit on noise")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ply = os.path.join(a.model, "point_cloud", f"iteration_{a.iteration}", "point_cloud.ply")
    bxyz, bnrm = load_billboards(ply)
    d = np.load(a.depth_cloud)
    dxyz = d["xyz"].astype(np.float64)
    lab = np.asarray(PlyData.read(a.labeled).elements[0].data["plane_id"]).astype(np.int64)
    if len(lab) != len(dxyz):
        raise SystemExit(f"{a.labeled} has {len(lab)} points but the cloud has {len(dxyz)}")

    dist = a.dist if a.dist else scene_extent(bxyz) * 0.005
    cos_t = np.cos(np.radians(a.normal_angle))
    print(f"{len(bxyz):,} billboards, {len(dxyz):,} depth points, "
          f"{lab.max()+1} depth planes; dist {dist:.4f}, normal {a.normal_angle}°")

    ids = [i for i in np.unique(lab) if i >= 0]
    new = np.full(len(dxyz), -1, np.int64)
    refit = kept = 0
    rows = []
    for pid in ids:
        m = lab == pid
        c0, n0 = fit_plane(dxyz[m])
        # billboards that lie on this plane and face the same way
        db = np.abs((bxyz - c0) @ n0)
        ok = (db < dist) & (np.abs(bnrm @ n0) > cos_t)
        if ok.sum() >= a.min_billboards:
            c1, n1 = fit_plane(bxyz[ok].astype(np.float64))
            if np.dot(n1, n0) < 0:
                n1 = -n1
            refit += 1
        else:
            c1, n1, kept = c0, n0, kept + 1
        # re-collect the depth points with the refined plane
        dd = np.abs((dxyz - c1) @ n1)
        take = (dd < dist) & (new < 0)
        new[take] = pid
        rows.append((int(m.sum()), int(ok.sum()), int(take.sum()),
                     float(np.degrees(np.arccos(min(1, abs(float(np.dot(n0, n1))))))),
                     int(pid)))

    print(f"refit on billboards: {refit}/{len(ids)} planes "
          f"({kept} kept their depth-only fit for lack of support)")
    rows.sort(reverse=True)
    print(f"\n{'id':>4} {'depth pts':>10} {'billboards':>11} {'after':>8} {'normal moved':>13}")
    for dn, bn, an, ang, pid in rows[:12]:
        print(f"{pid:>4} {dn:>10,} {bn:>11,} {an:>8,} {ang:>12.2f}°")
    moved = np.array([r[3] for r in rows])
    print(f"\nplane normals moved: median {np.median(moved):.2f}°, p90 {np.percentile(moved,90):.2f}°")
    print(f"assigned {int((new>=0).sum()):,} of {len(new):,} depth points "
          f"({100*(new>=0).mean():.1f}%), was {100*(lab>=0).mean():.1f}%")

    arr = np.zeros(len(dxyz), dtype=[('x','f4'),('y','f4'),('z','f4'),
                                     ('nx','f4'),('ny','f4'),('nz','f4'),('plane_id','i4')])
    arr['x'], arr['y'], arr['z'] = dxyz[:,0], dxyz[:,1], dxyz[:,2]
    nn = d["normal"].astype(np.float32)
    arr['nx'], arr['ny'], arr['nz'] = nn[:,0], nn[:,1], nn[:,2]
    arr['plane_id'] = new
    PlyData([PlyElement.describe(arr, 'vertex')]).write(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
