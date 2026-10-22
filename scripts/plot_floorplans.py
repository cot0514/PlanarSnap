"""
Supplementary figures for the PlanarSnap write-up (few words; the write-up
carries the explanation).

  python scripts/plot_floorplans.py --out docs/figures

  billboards_vs_depth.png -- room0 at wall height from above: billboard
                             centres vs back-projected rendered depth
  segmentation.png        -- how the planes become a room, billboards (top
                             row) vs rendered depth (bottom): segmented planes,
                             walls after consolidation, extruded floorplan
  floorplans.png          -- the five reconstructed floorplans over the GT

Reads the planes_*.ply / walls_*.json / room_*.json that
run_replica_scene.sh writes.
"""
import argparse, json, os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_walls_gt import load_semantic
from eval_footprint_gt import raster_tris
from analyze_walls import load_labeled

ROOT = "/home/cot0514/projects/3DGS"
OUT = f"{ROOT}/BBSplat/output"
SCENES = ["room0", "room1", "room2", "office0", "office1"]

# reference palette, light mode: categorical slot 1 + text/surface tokens
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE = "#2a78d6"
GT_FILL, GT_EDGE = "#e4e3df", "#52514e"
OTHER = "#c9c8c2"          # planes that are not walls

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def semantic_dir(scene):
    return f"{ROOT}/data/replica/official/semantic/{scene[:-1]}_{scene[-1]}/habitat"


def J(scene, name):
    return json.load(open(f"{OUT}/replica_{scene}_v1/{name}.json"))


class Frame:
    """Room frame: horizontal basis rotated so the dominant wall direction is x."""
    def __init__(self, r):
        self.e1, self.e2 = np.asarray(r["e1"]), np.asarray(r["e2"])
        self.up, self.floor = np.asarray(r["up"]), float(r["floor_height"])
        d0 = np.asarray(r["dirs"][0])
        t = -np.arctan2(d0[1], d0[0])
        self.R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def from3(self, X):
        return np.stack([X @ self.e1, X @ self.e2], -1) @ self.R.T

    def from2(self, P):
        return np.asarray(P) @ self.R.T

    def height(self, X):
        return X @ self.up - self.floor


def gt_outline(r, verts, tris, cls):
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    e1, e2 = np.asarray(r["e1"]), np.asarray(r["e2"])
    t = verts[tris[cls == "ceiling"]]
    m = raster_tris(np.stack([t @ e1, t @ e2], -1), lo, res, tuple(r["raster"]["shape"]))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.02 / res, True)[:, 0, :] * res + lo


def off_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def label(ax, title, sub=None, three_d=False):
    put = ax.text2D if three_d else ax.text
    x, y = (0.04, 0.93) if three_d else (0.0, 1.10)
    kw = dict(transform=ax.transAxes, ha="left", va="bottom")
    put(x, y, title, fontsize=13, fontweight="bold", color=INK, **kw)
    if sub:
        put(x, y - (0.065 if three_d else 0.10), sub, fontsize=11, color=INK2, **kw)


# ------------------------------------------------------------------------
def fig_points(out, scene="room0"):
    r = J(scene, "room_hybrid_cons_ref"); F = Frame(r)
    pc = f"{OUT}/replica_{scene}_v1/point_cloud/iteration_32000"
    v, t, _, c = load_semantic(semantic_dir(scene))
    gt = F.from2(gt_outline(r, v, t, c))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (f, name) in zip(axes, [("planes_bb_ransac", "Billboards"),
                                    ("planes_depth_pca", "Rendered depth")]):
        X = load_labeled(f"{pc}/{f}.ply")[0]
        h = F.height(X)
        q = F.from3(X[(h > 0.3) & (h < 2.3)])
        ax.plot(*np.vstack([gt, gt[:1]]).T, color=GT_EDGE, lw=1.0, ls=(0, (4, 3)))
        ax.scatter(q[:, 0], q[:, 1], s=0.2, color=BLUE, alpha=0.35, lw=0, rasterized=True)
        c = 0.5 * (gt.min(0) + gt.max(0)); half = 0.5 * np.ptp(gt, axis=0) + 0.4
        ax.set_xlim(c[0] - half[0], c[0] + half[0]); ax.set_ylim(c[1] - half[1], c[1] + half[1])
        ax.set_aspect("equal"); off_axes(ax)
        label(ax, name, f"{len(X):,} points, wall height, top view")
    fig.legend(handles=[Line2D([], [], color=GT_EDGE, lw=1, ls=(0, (4, 3)), label="GT room outline")],
               loc="lower right", frameon=False, fontsize=11, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out, dpi=180); plt.close(fig)


# ------------------------------------------------------------------------
def room_points(ax, X, colors, F, ceil_h, lo, hi, n=45000, seed=0):
    # mplot3d does not clip to the axis limits, so cut the floaters behind the
    # walls and the ceiling (to see inside) here
    h = F.height(X)
    P = F.from3(X)
    keep = np.where((h > -0.05) & (h < ceil_h - 0.15)
                    & np.all((P > lo) & (P < hi), axis=1))[0]
    keep = keep[np.random.default_rng(seed).permutation(len(keep))[:n]]
    p = P[keep]
    c = colors[keep] if isinstance(colors, np.ndarray) else colors
    ax.scatter(p[:, 0], p[:, 1], h[keep], s=0.5, c=c, lw=0, depthshade=False, rasterized=True)


def frame3d(ax, lo, hi, H, azim, elev):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(0, H)
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], H))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def extruded(ax, poly, H, azim):
    """Back walls filled, front walls as a floor outline only (dollhouse view)."""
    v = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        outward = np.array([e[1], -e[0]]) * (1 if area > 0 else -1)
        if outward @ v < 0:
            ax.add_collection3d(Poly3DCollection([[[a[0], a[1], 0], [b[0], b[1], 0],
                                                   [b[0], b[1], H], [a[0], a[1], H]]],
                                                 facecolor=BLUE, alpha=0.22, edgecolor=BLUE, lw=1.0))
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], [0, 0], color=BLUE, lw=1.8)
    ax.add_collection3d(Poly3DCollection([[[x, y, 0] for x, y in poly]],
                                         facecolor=BLUE, alpha=0.10, edgecolor=BLUE, lw=1.8))


def fig_segmentation(out, scene="room0", azim=-58, elev=28):
    pc = f"{OUT}/replica_{scene}_v1/point_cloud/iteration_32000"
    rows = [("Billboards", "bb_ransac", "bb_ransac_cons", "room_bb_ransac"),
            ("Rendered depth", "hybrid", "hybrid_cons", "room_hybrid_cons_ref")]
    ref = J(scene, "room_hybrid_cons_ref")
    F = Frame(ref)
    H = float(ref["ceiling_height"]) - F.floor
    gpoly = F.from2(ref["polygon"])
    lo, hi = gpoly.min(0) - 0.1, gpoly.max(0) + 0.1
    wall_rgb, other_rgb = matplotlib.colors.to_rgb(BLUE), matplotlib.colors.to_rgb(OTHER)

    fig = plt.figure(figsize=(14, 7.6))
    for row, (name, seg, cons, room) in enumerate(rows):
        for col, arm in enumerate([seg, cons]):
            ax = fig.add_subplot(2, 3, 3 * row + col + 1, projection="3d")
            X, lab = load_labeled(f"{pc}/planes_{arm}.ply")
            walls = J(scene, f"walls_{arm}")["wall_ids"]
            on = lab >= 0
            colors = np.where(np.isin(lab, walls)[:, None], wall_rgb, other_rgb)
            room_points(ax, X[on], colors[on], F, H, lo, hi, seed=col)
            frame3d(ax, lo, hi, H, azim, elev)
            step = ["planes", "walls merged"][col]
            label(ax, f"{name}: {step}", f"{len(walls)} wall planes", three_d=True)
        ax = fig.add_subplot(2, 3, 3 * row + 3, projection="3d")
        r = J(scene, room)
        d = J(scene, room.replace("room_", "floorplan_"))
        X, _ = load_labeled(f"{pc}/planes_{seg}.ply")
        room_points(ax, X, OTHER, F, H, lo, hi, n=25000)
        extruded(ax, F.from2(r["polygon"]), H, azim)
        frame3d(ax, lo, hi, H, azim, elev)
        label(ax, f"{name}: floorplan",
              f"boundary {d['chamfer_cm']:.1f} cm, GT wall area {100 * d['wall_recall_3d']:.0f}%",
              three_d=True)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", ms=6, label="wall planes"),
                        Line2D([], [], color=OTHER, marker="o", ls="", ms=6, label="other planes"),
                        Patch(facecolor=BLUE, alpha=0.3, edgecolor=BLUE, label="reconstructed walls and floor")],
               loc="lower center", ncol=3, frameon=False, fontsize=11, labelcolor=INK2)
    fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0.06, wspace=0.0, hspace=0.02)
    fig.savefig(out, dpi=170); plt.close(fig)


# ------------------------------------------------------------------------
def fig_floorplans(out):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    rooms = []
    for s in SCENES:
        r = J(s, "room_hybrid_cons_ref")
        v, t, _, c = load_semantic(semantic_dir(s))
        rooms.append((s, r, Frame(r), gt_outline(r, v, t, c), J(s, "footprint_hybrid_cons_ref")))
    half = 0.5 * np.max([np.ptp(F.from2(g), axis=0) for _, _, F, g, _ in rooms], axis=0) + 0.3
    for ax, (s, r, F, g, fp) in zip(axes, rooms):
        g = F.from2(g); p = F.from2(r["polygon"])
        ax.fill(g[:, 0], g[:, 1], color=GT_FILL, lw=0)
        ax.plot(*np.vstack([g, g[:1]]).T, color=GT_EDGE, lw=1.0)
        ax.plot(*np.vstack([p, p[:1]]).T, color=BLUE, lw=2.0)
        c = 0.5 * (g.min(0) + g.max(0))
        ax.set_xlim(c[0] - half[0], c[0] + half[0]); ax.set_ylim(c[1] - half[1] - 0.3, c[1] + half[1])
        ax.set_aspect("equal"); off_axes(ax)
        label(ax, s, f"IoU {100 * fp['iou']:.0f}%")
    ax = axes[0]
    x0, y0 = ax.get_xlim()[0] + 0.1, ax.get_ylim()[0] + 0.05
    ax.plot([x0, x0 + 1], [y0, y0], color=INK2, lw=1.5)
    ax.text(x0 + 0.5, y0 + 0.06, "1 m", ha="center", va="bottom", fontsize=10, color=INK2)
    fig.legend(handles=[Patch(facecolor=GT_FILL, edgecolor=GT_EDGE, label="GT footprint"),
                        Line2D([], [], color=BLUE, lw=2, label="reconstructed")],
               loc="lower center", ncol=2, frameon=False, fontsize=11, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    fig.savefig(out, dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in [("billboards_vs_depth", fig_points), ("segmentation", fig_segmentation),
                     ("floorplans", fig_floorplans)]:
        fn(os.path.join(a.out, name + ".png"))
        print("wrote", name)


if __name__ == "__main__":
    main()
