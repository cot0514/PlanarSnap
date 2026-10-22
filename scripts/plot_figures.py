"""
Supplementary figures for the PlanarSnap write-up (few words; the write-up
carries the explanation).

  python scripts/plot_figures.py --out docs/figures [--only name,name]

  coverage_by_class.png   -- billboard coverage of each GT class (2.1)
  billboards_vs_depth.png -- room0 at wall height from above: billboard
                             centres vs back-projected rendered depth (2.2)
  depth_accuracy.png      -- distance to the GT surface, cumulative: billboards,
                             rendered depth, GT depth (3.2)
  wall_detection.png      -- walls found, and one plane per wall before/after
                             consolidation (5.3, 10.3)
  up_correction.png       -- GT floor heights along the camera up vs the
                             refined up (5.7)
  segmentation.png        -- how the planes become a room, billboards (top
                             row) vs hybrid (bottom): segmented planes, walls
                             after consolidation, extruded floorplan (10)
  floorplans.png          -- the five reconstructed floorplans over the GT (10.4)
  floorplan_accuracy.png  -- boundary error and rebuilt wall area (10.5)
  wall_elevation.png      -- room0's GT walls unrolled: observed wall points vs
                             extruded walls (10.5)

Reads the planes_*.ply / walls_*.json / room_*.json that
run_replica_scene.sh writes.
"""
import argparse, json, os, sys
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
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
from analyze_walls import load_labeled, up_from_cameras
from eval_floorplan_gt import sample_boundary, sample_triangles

ROOT = "/home/cot0514/projects/3DGS"
OUT = f"{ROOT}/BBSplat/output"
SCENES = ["room0", "room1", "room2", "office0", "office1"]

# reference palette, light mode: categorical slot 1 + text/surface tokens
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e9e8e4"
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


# ------------------------------------------------------------------------
def bar_axes(ax):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0, colors=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig_coverage(out):
    # eval_coverage_gt.py, pooled over the five scenes (write-up table 2.1)
    rows = [("ceiling", 8.9), ("wall", 27.5), ("door", 28.9), ("lamp", 30.7), ("floor", 30.9),
            ("sofa", 44.8), ("chair", 49.9), ("cabinet", 60.1), ("blinds", 60.9),
            ("rug", 70.3), ("table", 76.1), ("pillow", 86.0)]
    arch = {"ceiling", "wall", "floor"}
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(rows))[::-1]
    for yi, (n, v) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=BLUE if n in arch else OTHER, zorder=2)
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=11,
                color=INK if n in arch else INK2, fontweight="bold" if n in arch else "normal")
    ax.set_yticks(y, [n for n, _ in rows])
    for t, (n, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(INK if n in arch else INK2); t.set_fontweight("bold" if n in arch else "normal")
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    bar_axes(ax)
    ax.set_title("Billboard coverage of each GT class (share of the surface with a billboard within 5 cm)",
                 loc="left", fontsize=11, color=INK2, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180); plt.close(fig)


def mesh_scene(path):
    m = o3d.io.read_triangle_mesh(path)
    t = o3d.t.geometry.TriangleMesh.from_legacy(m)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(t)
    return sc


def fig_depth_accuracy(out):
    sets = {"Billboard centres": [], "Rendered depth": [], "GT depth (oracle)": []}
    for s in SCENES:
        sc = mesh_scene(f"{ROOT}/data/replica/cull_replica_mesh/{s}.ply")
        o = f"{OUT}/replica_{s}_v1"
        d = PlyData.read(f"{o}/point_cloud/iteration_32000/point_cloud.ply").elements[0].data
        clouds = {"Billboard centres": np.stack([d["x"], d["y"], d["z"]], 1),
                  "Rendered depth": np.load(f"{o}/depth_cloud_v05.npz")["xyz"],
                  "GT depth (oracle)": np.load(f"{o}/gtdepth.npz")["xyz"]}
        for k, X in clouds.items():
            X = X.astype(np.float32)
            cp = sc.compute_closest_points(o3d.core.Tensor(X))["points"].numpy()
            sets[k].append(np.linalg.norm(X - cp, axis=1))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.linspace(0, 20, 401)
    for (k, v), c in zip(sets.items(), [BLUE, ORANGE, MUTED]):
        d = np.sort(np.concatenate(v)) * 100
        cdf = np.searchsorted(d, xs, side="right") / len(d)
        at10 = 100 * np.searchsorted(d, 10.0, side="right") / len(d)
        ax.plot(xs, 100 * cdf, color=c, lw=2.2, ls="-" if c != MUTED else (0, (4, 2)),
                label=f"{k}: {at10:.0f}% within 10 cm, median {np.median(d):.1f} cm")
    ax.axvline(10, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(0, 20); ax.set_ylim(0, 102)
    ax.set_xlabel("distance to the GT surface (cm)", color=INK2)
    ax.set_ylabel("share of points within (%)", color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=180); plt.close(fig)


def fig_wall_detection(out):
    groups = [("Billboards", "bb_ransac", 4, 6), ("Rendered depth", "depth_pca", 6, 13),
              ("Hybrid", "hybrid", 2, 13), ("GT depth (oracle)", "gtdepth", 17, 18)]
    # one plane covering >= 50% of a GT wall, before / after consolidate_walls.py
    # (eval_walls_gt.py --min_overlap 0.5)
    found = [sum(J(s, f"geom_{a}")["matched"] for s in SCENES) for _, a, _, _ in groups]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(groups)); w = 0.26
    series = [("wall plane found (20 deg, 30 cm)", found, BLUE),
              ("one plane covers >= 50% of the wall", [g[2] for g in groups], ORANGE),
              ("... after merging wall fragments", [g[3] for g in groups], AQUA)]
    for k, (name, vals, c) in enumerate(series):
        b = ax.bar(x + (k - 1) * (w + 0.02), vals, w, color=c, label=name, zorder=2)
        for bi, v in zip(b, vals):
            ax.text(bi.get_x() + bi.get_width() / 2, v + 0.3, str(v), ha="center", va="bottom",
                    fontsize=10, color=INK)
    ax.axhline(22, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(len(groups) - 0.55, 22.3, "22 GT walls", ha="right", va="bottom", fontsize=10, color=INK2)
    ax.set_xticks(x, [g[0] for g in groups]); ax.set_ylim(0, 25)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, colors=INK2); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, fontsize=10, labelcolor=INK, ncol=3,
              bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    fig.savefig(out, dpi=180); plt.close(fig)


def fig_up_correction(out, scene="room0"):
    v, t, oid, c = load_semantic(semantic_dir(scene))
    T = v[t]; area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    fm = c == "floor"
    ids, inv = np.unique(oid[fm], return_inverse=True)
    keep = fm & (oid == ids[np.bincount(inv, weights=area[fm]).argmax()])
    cen, a = T[keep].mean(1), area[keep]
    cam, _ = up_from_cameras(f"{OUT}/replica_{scene}_v1/cameras.json")
    ref = np.asarray(J(scene, "walls_hybrid_cons")["up"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    hs = {}
    for up in (cam, ref):
        h = (cen @ up) * 100
        hs[id(up)] = h - np.median(h)
    allh = np.concatenate(list(hs.values()))
    bins = np.arange(np.floor(allh.min()) - 2, np.ceil(allh.max()) + 3, 1.0)
    for up, name, col in ((cam, "camera up (5.5 deg off gravity)", OTHER), (ref, "refined from the planes", BLUE)):
        h = hs[id(up)]
        ax.hist(h, bins=bins, weights=a, color=col, alpha=0.9 if col == BLUE else 1.0,
                label=f"{name}: 5-95% spread {np.percentile(h, 95) - np.percentile(h, 5):.0f} cm", zorder=2)
    ax.set_xlabel("height of the GT floor relative to its median (cm)", color=INK2)
    ax.set_ylabel("floor area (m$^2$)", color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)                 # room for the legend
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper right")
    ax.set_title(f"{scene}: one flat floor, measured along two up directions", loc="left",
                 fontsize=11, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=180); plt.close(fig)


def fig_floorplan_accuracy(out):
    arms = [("Billboards", "bb_ransac", OTHER), ("Hybrid + merged walls", "hybrid_cons", OTHER),
            ("... + billboard refit", "hybrid_cons_ref", BLUE), ("GT depth (oracle)", "gtdepth_cons", "#e1e0db")]
    ch = [np.mean([J(s, f"floorplan_{a}")["chamfer_cm"] for s in SCENES]) for _, a, _ in arms]
    rc = [100 * np.mean([J(s, f"floorplan_{a}")["wall_recall_3d"] for s in SCENES]) for _, a, _ in arms]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, vals, name, unit in ((axes[0], ch, "boundary error (lower is better)", " cm"),
                                 (axes[1], rc, "GT wall area rebuilt within 10 cm", "%")):
        ax.barh(y, vals, height=0.6, color=[c for _, _, c in arms], zorder=2)
        for yi, val, (_, a, _) in zip(y, vals, arms):
            ax.text(val + max(vals) * 0.02, yi, f"{val:.1f}{unit}", va="center", fontsize=10,
                    color=INK if a == "hybrid_cons_ref" else INK2,
                    fontweight="bold" if a == "hybrid_cons_ref" else "normal")
        ax.set_xlim(0, max(vals) * 1.3); ax.set_xticks([])
        bar_axes(ax); ax.grid(False)
        ax.set_title(name, loc="left", fontsize=11, color=INK, fontweight="bold")
    axes[0].set_yticks(y, [n for n, _, _ in arms])
    fig.tight_layout()
    fig.savefig(out, dpi=180); plt.close(fig)


def fig_elevation(out, scene="room0", dist=0.10):
    r = J(scene, "room_hybrid_cons_ref")
    verts, tris, oid, cls = load_semantic(semantic_dir(scene))
    up, e1, e2 = (np.asarray(r[k]) for k in ("up", "e1", "e2"))
    floor, ceil = r["floor_height"], r["ceiling_height"]
    S, _ = sample_triangles(verts[tris[cls == "wall"]], 2500.0, np.random.default_rng(0))
    S2 = np.stack([S @ e1, S @ e2], 1); hs = S @ up - floor
    B = sample_boundary(gt_outline(r, verts, tris, cls), 0.01)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
    u = arc[cKDTree(B).query(S2)[1]]
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    contour = np.round((np.asarray(r["polygon"]) - lo) / res).astype(np.int32).reshape(-1, 1, 2)
    px = (S2 - lo) / res
    d2 = np.array([abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True)) for x, y in px]) * res
    ext = np.hypot(d2, np.maximum(0, np.maximum(-hs, hs - (ceil - floor)))) < dist
    X, lab = load_labeled(r["labeled"])
    W = X[np.isin(lab, json.load(open(r["walls"]))["wall_ids"])]
    obs = cKDTree(W).query(S, distance_upper_bound=dist)[0] < dist
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    for ax, hit, name in ((axes[0], obs, "observed wall points"),
                          (axes[1], ext, "floorplan walls, floor to ceiling")):
        o = np.random.default_rng(1).permutation(len(u))
        ax.scatter(u[o], hs[o], s=0.6, lw=0, c=np.where(hit[o], BLUE, ORANGE), rasterized=True)
        ax.set_title(f"{name}: {100 * hit.mean():.0f}% of the GT wall area", loc="left",
                     fontsize=11, color=INK, fontweight="bold")
        ax.set_ylabel("height (m)", color=INK2); ax.set_yticks([0, 1, 2])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2)
    axes[1].set_xlabel(f"{scene}: position along the room outline (m)", color=INK2)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="s", ls="", ms=6, label="rebuilt (within 10 cm)"),
                        Line2D([], [], color=ORANGE, marker="s", ls="", ms=6, label="missed")],
               loc="upper right", ncol=2, frameon=False, fontsize=10, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [("coverage_by_class", fig_coverage), ("billboards_vs_depth", fig_points),
            ("depth_accuracy", fig_depth_accuracy), ("wall_detection", fig_wall_detection),
            ("up_correction", fig_up_correction), ("segmentation", fig_segmentation),
            ("floorplans", fig_floorplans), ("floorplan_accuracy", fig_floorplan_accuracy),
            ("wall_elevation", fig_elevation)]
    want = set(a.only.split(",")) if a.only else None
    for name, fn in jobs:
        if want and name not in want:
            continue
        fn(os.path.join(a.out, name + ".png"))
        print("wrote", name)


if __name__ == "__main__":
    main()
