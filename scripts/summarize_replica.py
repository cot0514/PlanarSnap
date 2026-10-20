"""
Collect the per-scene JSON written by run_replica_scene.sh into the tables the
write-up needs.

  python scripts/summarize_replica.py [--scenes room0 room1 ...] [--csv out.csv]
"""
import argparse, json, os, glob
import numpy as np

ARMS = ["bb_ransac", "bb_pca", "depth_ransac", "depth_pca", "hybrid", "gtdepth"]
LABEL = {"bb_ransac": "billboards + RANSAC (original)",
         "bb_pca":    "billboards + normal RANSAC",
         "depth_ransac": "depth fusion + RANSAC",
         "depth_pca": "depth fusion + normal RANSAC",
         "hybrid":    "depth planes, billboard refit",
         "gtdepth":   "GT depth (oracle)"}


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if a.scenes:
        scenes = a.scenes
    else:
        scenes = sorted(os.path.basename(d).replace("replica_", "").replace(f"_{a.tag}", "")
                        for d in glob.glob(os.path.join(a.root, f"replica_*_{a.tag}")))
    print(f"scenes: {', '.join(scenes)}\n")

    # ---- rendering quality -------------------------------------------------
    print("=" * 74)
    print("RENDERING")
    print(f"{'scene':<10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    for s in scenes:
        r = load(f"{a.root}/replica_{s}_{a.tag}/results.json")
        if not r:
            print(f"{s:<10} {'-':>8}"); continue
        k = sorted(r)[-1]
        print(f"{s:<10} {r[k]['PSNR']:>8.2f} {r[k]['SSIM']:>8.4f} {r[k]['LPIPS']:>8.4f}")

    # ---- normals -----------------------------------------------------------
    print()
    print("=" * 74)
    print("NORMAL ACCURACY vs GT MESH  (mean angular error, degrees)")
    print(f"{'scene':<10} {'billboard':>10} {'local PCA':>10} {'depth map':>10}   "
          f"{'bb horiz':>9} {'bb vert':>9}")
    for s in scenes:
        n = load(f"{a.root}/replica_{s}_{a.tag}/eval_normals.json")
        nd = load(f"{a.root}/replica_{s}_{a.tag}/eval_normals_depth.json")
        if not n:
            print(f"{s:<10} {'-':>10}"); continue
        R = n["results"]
        g = lambda d, k, sub="all": d["results"][k][sub]["mean"] if d and k in d["results"] else float("nan")
        print(f"{s:<10} {g(n,'own_quat'):>9.1f}° {g(n,'local_pca'):>9.1f}° "
              f"{g(nd,'depth_map'):>9.1f}°   {g(n,'own_quat','horizontal'):>8.1f}° "
              f"{g(n,'own_quat','vertical'):>8.1f}°")

    # ---- walls -------------------------------------------------------------
    rows = []
    print()
    print("=" * 74)
    print("WALL RECOVERY   (GT = Replica semantic instances)")
    print(f"{'arm':<32} {'scene':<9} {'found':>6} {'inst':>7} {'IoU':>7} "
          f"{'matched':>8} {'angle':>7} {'offset':>8}")
    for arm in ARMS:
        agg = {"inst": [0, 0], "iou": [], "matched": [0, 0], "ang": [], "off": []}
        for s in scenes:
            d = f"{a.root}/replica_{s}_{a.tag}"
            w, e, gm = load(f"{d}/walls_{arm}.json"), load(f"{d}/eval_walls_{arm}.json"), load(f"{d}/geom_{arm}.json")
            if not (w and e):
                continue
            iou = e["wall"]["iou"]
            inst, ninst = e["instances"]["recovered"], e["instances"]["gt_wall_instances"]
            if gm:
                mt, nm = gm["matched"], gm["gt_walls"]
                ok = [x for x in gm["walls"] if x.get("plane") is not None]
                ang = np.median([x["angle_deg"] for x in ok]) if ok else float("nan")
                off = np.median([x["offset_cm"] for x in ok]) if ok else float("nan")
            else:
                mt = nm = 0; ok = []; ang = off = float("nan")
            print(f"{LABEL[arm]:<32} {s:<9} {w['walls']:>6} {inst:>3}/{ninst:<3} "
                  f"{100*iou:>6.1f}% {mt:>4}/{nm:<3} {ang:>6.2f}° {off:>7.1f}cm")
            agg["inst"][0] += inst; agg["inst"][1] += ninst
            agg["iou"].append(iou)
            agg["matched"][0] += mt; agg["matched"][1] += nm
            if ok:
                agg["ang"] += [x["angle_deg"] for x in ok]
                agg["off"] += [x["offset_cm"] for x in ok]
            rows.append(dict(arm=arm, scene=s, walls=w["walls"], recovered=inst,
                             gt_inst=ninst, iou=iou, matched=mt, gt_walls=nm,
                             angle=ang, offset=off))
        if agg["inst"][1]:
            print(f"{'  -> all scenes':<32} {'':<9} {'':>6} "
                  f"{agg['inst'][0]:>3}/{agg['inst'][1]:<3} "
                  f"{100*np.mean(agg['iou']):>6.1f}% "
                  f"{agg['matched'][0]:>4}/{agg['matched'][1]:<3} "
                  f"{np.median(agg['ang']) if agg['ang'] else float('nan'):>6.2f}° "
                  f"{np.median(agg['off']) if agg['off'] else float('nan'):>7.1f}cm")
            print()

    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
