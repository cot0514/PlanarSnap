"""
Figures for the floorplan results on Replica (static PNGs for the write-up).

  python scripts/plot_floorplans.py --out docs/figures

  floorplans.png    -- the final floorplan of every scene over the GT footprint
  comparison.png    -- billboards alone / hybrid / hybrid + billboard refinement
                       on room0 and room1, polygon edges split into detected-wall
                       and virtual edges
  wall_elevation.png -- one scene's GT walls unrolled along the room outline,
                       coloured by whether they are rebuilt: observed wall
                       points only vs the extruded walls

Reads the room_*.json / floorplan_*.json that run_replica_scene.sh writes.
"""
import argparse, json, os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_walls_gt import load_semantic
from eval_footprint_gt import raster_tris
from eval_floorplan_gt import sample_boundary, sample_triangles
from analyze_walls import load_labeled

ROOT = "/home/cot0514/projects/3DGS"
SCENES = ["room0", "room1", "room2", "office0", "office1"]
FINAL = "hybrid_cons_ref"
LABEL = {"bb_ransac": "Billboards alone",
         "hybrid_cons": "Hybrid (depth walls)",
         "hybrid_cons_ref": "Hybrid + billboard refit"}

# reference palette, light mode: categorical slots 1-2, text and surface tokens
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GT_FILL, GT_EDGE, PTS = "#e4e3df", "#52514e", "#b9b8b2"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def semantic_dir(scene):
    return f"{ROOT}/data/replica/official/semantic/{scene[:-1]}_{scene[-1]}/habitat"


def load_room(scene, arm):
    o = f"{ROOT}/BBSplat/output/replica_{scene}_v1"
    return (json.load(open(f"{o}/room_{arm}.json")),
            json.load(open(f"{o}/floorplan_{arm}.json")),
            json.load(open(f"{o}/footprint_{arm}.json")))


class Frame:
    """2-D room frame rotated so the room's dominant direction is the x axis."""
    def __init__(self, r):
        self.e1, self.e2 = np.asarray(r["e1"]), np.asarray(r["e2"])
        d0 = np.asarray(r["dirs"][0])
        t = -np.arctan2(d0[1], d0[0])
        self.R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        self.r = r

    def from3(self, X):
        return np.stack([X @ self.e1, X @ self.e2], -1) @ self.R.T

    def from2(self, P):
        return np.asarray(P) @ self.R.T


def gt_outline(r, verts, tris, cls):
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    shape = tuple(r["raster"]["shape"])
    e1, e2 = np.asarray(r["e1"]), np.asarray(r["e2"])
    t = verts[tris[cls == "ceiling"]]
    m = raster_tris(np.stack([t @ e1, t @ e2], -1), lo, res, shape)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(c, 0.02 / res, True)[:, 0, :] * res + lo


def edge_kinds(r, dist=0.10):
    """For each polygon edge: True if it lies on a detected wall line."""
    poly = np.asarray(r["polygon"])
    kinds = []
    for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
        L = np.linalg.norm(p1 - p0)
        en = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / max(L, 1e-9)
        mid = 0.5 * (p0 + p1)
        best = None
        for ln in r["lines"]:
            n = np.asarray(ln["n"])
            if abs(float(n @ en)) < np.cos(np.radians(10)):
                continue
            d = abs(float((mid - np.asarray(ln["p"])) @ n))
            if d < dist and (best is None or d < best[0]):
                best = (d, ln["wall_id"] is not None)
        kinds.append(bool(best and best[1]))
    return kinds


def draw_room(ax, r, gt, pts=None, split_edges=False):
    F = Frame(r)
    g = F.from2(gt)
    ax.fill(g[:, 0], g[:, 1], color=GT_FILL, zorder=1, lw=0)
    ax.plot(*np.vstack([g, g[:1]]).T, color=GT_EDGE, lw=1.0, zorder=2)
    if pts is not None:
        p = F.from3(pts)
        ax.scatter(p[:, 0], p[:, 1], s=0.15, color=PTS, zorder=1.5, lw=0, rasterized=True)
    poly = F.from2(r["polygon"])
    if split_edges:
        for (a, b), wall in zip(zip(poly, np.roll(poly, -1, axis=0)), edge_kinds(r)):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE if wall else ORANGE,
                    lw=2.2, ls="-" if wall else (0, (3, 1.6)), zorder=3,
                    solid_capstyle="round")
    else:
        ax.plot(*np.vstack([poly, poly[:1]]).T, color=BLUE, lw=2.2, zorder=3)
    allp = np.vstack([g, poly])
    c = 0.5 * (allp.min(0) + allp.max(0))
    return c, allp.max(0) - allp.min(0)


def frame_axes(ax, c, half):
    """half = (half-width, half-height); one scale shared across panels."""
    hx, hy = half
    ax.set_xlim(c[0] - hx, c[0] + hx)
    ax.set_ylim(c[1] - hy - 0.35, c[1] + hy)              # room for the scale bar
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scale_bar(ax, c, half):
    hx, hy = half
    x0, y0 = c[0] - hx + 0.15, c[1] - hy - 0.28
    ax.plot([x0, x0 + 1.0], [y0, y0], color=INK2, lw=1.5)
    ax.text(x0 + 0.5, y0 + 0.06, "1 m", ha="center", va="bottom", color=INK2, fontsize=8)


def panel_title(ax, title, stats):
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", fontweight="bold", pad=16)
    ax.text(0.0, 1.01, stats, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def extents(rooms):
    """Shared half-extents (x, y) over rooms, in the rotated room frame."""
    spans = [np.ptp(Frame(r).from2(r["polygon"]), axis=0) for r in rooms]
    return tuple(0.5 * np.max(spans, axis=0) + 0.45)


def fig_floorplans(out, sem):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    rooms = []
    for s in SCENES:
        r, f, fp = load_room(s, FINAL)
        xyz, lab = load_labeled(r["labeled"])
        h = xyz @ np.asarray(r["up"])
        keep = (h > r["floor_height"] + 0.05) & (h < r["ceiling_height"] - 0.05)
        sel = np.where(keep)[0]
        sel = sel[np.random.default_rng(0).permutation(len(sel))[:40000]]
        rooms.append((s, r, f, fp, xyz[sel]))
    half = extents([r for _, r, _, _, _ in rooms])
    for ax, (s, r, f, fp, pts) in zip(axes.ravel(), rooms):
        c, _ = draw_room(ax, r, gt_outline(r, *sem[s]), pts)
        frame_axes(ax, c, half)
        scale_bar(ax, c, half)
        panel_title(ax, s, f"IoU {100 * fp['iou']:.1f}%   boundary {f['chamfer_cm']:.1f} cm   "
                    f"walls rebuilt {100 * f['wall_recall_3d']:.0f}%")
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=GT_FILL, edgecolor=GT_EDGE, label="GT footprint (ceiling)"),
                       Line2D([], [], color=BLUE, lw=2.2, label="Reconstructed floorplan"),
                       Line2D([], [], color=PTS, marker="o", ls="", ms=3, label="Depth points (top view)")],
              loc="center left", frameon=False, fontsize=10, labelcolor=INK)
    ax.text(0.0, 0.18, "Hybrid planes + wall consolidation\n+ billboard refit of the wall lines.\n"
            "All panels share one scale.", transform=ax.transAxes, color=INK2, fontsize=9)
    fig.suptitle("Reconstructed floorplans, Replica", x=0.02, ha="left", color=INK, fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_comparison(out, sem, scenes=("room0", "room1")):
    arms = ["bb_ransac", "hybrid_cons", "hybrid_cons_ref"]
    fig, axes = plt.subplots(len(scenes), 3, figsize=(10.5, 2.9 * len(scenes) + 0.9))
    for row, s in zip(np.atleast_2d(axes), scenes):
        data = [load_room(s, a) for a in arms]
        half = extents([r for r, _, _ in data])
        for ax, a, (r, f, fp) in zip(row, arms, data):
            c, _ = draw_room(ax, r, gt_outline(r, *sem[s]), split_edges=True)
            frame_axes(ax, c, half)
            panel_title(ax, f"{s} - {LABEL[a]}",
                        f"boundary {f['chamfer_cm']:.1f} cm   walls rebuilt "
                        f"{100 * f['wall_recall_3d']:.0f}%   wall edges {100 * f['wall_backed']:.0f}%")
    fig.legend(handles=[Patch(facecolor=GT_FILL, edgecolor=GT_EDGE, label="GT footprint"),
                        Line2D([], [], color=BLUE, lw=2.2, label="Edge on a detected wall"),
                        Line2D([], [], color=ORANGE, lw=2.2, ls=(0, (3, 1.6)),
                               label="Virtual edge (extent of the points)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Where the polygon edges come from", x=0.02, ha="left", color=INK,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_elevation(out, sem, scene="room0", arm=FINAL, dist=0.10):
    r, f, _ = load_room(scene, arm)
    verts, tris, oid, cls = sem[scene]
    up, e1, e2 = (np.asarray(r[k]) for k in ("up", "e1", "e2"))
    floor, ceil = r["floor_height"], r["ceiling_height"]
    S, _ = sample_triangles(verts[tris[cls == "wall"]], 2500.0, np.random.default_rng(0))
    S2 = np.stack([S @ e1, S @ e2], 1)
    hs = S @ up - floor

    # unroll along the GT outline: arc length of the nearest outline point
    gt = gt_outline(r, verts, tris, cls)
    B = sample_boundary(gt, 0.01)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
    u = arc[cKDTree(B).query(S2)[1]]

    # rebuilt by the extruded walls (as in eval_floorplan_gt) ...
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    contour = np.round((np.asarray(r["polygon"]) - lo) / res).astype(np.int32).reshape(-1, 1, 2)
    px = (S2 - lo) / res
    d2 = np.array([abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True)) for x, y in px]) * res
    dh = np.maximum(0, np.maximum(-hs, hs - (ceil - floor)))
    ext = np.hypot(d2, dh) < dist
    # ... and by the observed wall-plane points alone
    xyz, lab = load_labeled(r["labeled"])
    W = xyz[np.isin(lab, json.load(open(r["walls"]))["wall_ids"])]
    obs = cKDTree(W).query(S, distance_upper_bound=dist)[0] < dist

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.2), sharex=True)
    for ax, hit, name in ((axes[0], obs, "Observed wall points only (segmentation)"),
                          (axes[1], ext, "Extruded walls, floor to ceiling (floorplan)")):
        o = np.random.default_rng(1).permutation(len(u))
        ax.scatter(u[o], hs[o], s=0.6, lw=0, c=np.where(hit[o], BLUE, ORANGE), rasterized=True)
        ax.set_title(f"{name}:  {100 * hit.mean():.0f}% of the GT wall area within {int(dist * 100)} cm",
                     color=INK, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylabel("height above floor (m)")
        ax.set_ylim(-0.1, (ceil - floor) + 0.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e9e8e4", lw=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("position along the room outline (m)")
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="s", ls="", ms=7, label="Rebuilt"),
                        Line2D([], [], color=ORANGE, marker="s", ls="", ms=7, label="Missed")],
               loc="upper right", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(f"{scene}: GT walls unrolled along the room outline ({LABEL[arm]})",
                 x=0.02, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--elevation_scene", default="room0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sem = {}
    for s in SCENES:
        v, t, o, c = load_semantic(semantic_dir(s))
        sem[s] = (v, t, o, c)
    fig_floorplans(os.path.join(a.out, "floorplans.png"), {s: (v, t, c) for s, (v, t, o, c) in sem.items()})
    fig_comparison(os.path.join(a.out, "comparison.png"), {s: (v, t, c) for s, (v, t, o, c) in sem.items()})
    fig_elevation(os.path.join(a.out, "wall_elevation.png"), sem, a.elevation_scene)
    print(f"wrote {a.out}/floorplans.png, comparison.png, wall_elevation.png")


if __name__ == "__main__":
    main()
