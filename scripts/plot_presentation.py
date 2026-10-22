"""
Slide figures for the PlanarSnap talk: Korean labels, 16:9, large type, one
message per figure, in the order the talk tells it.

  python scripts/plot_presentation.py --out docs/figures/slides \
      --font ~/.local/share/fonts/NanumGothic.ttf

  01_coverage        problem   -- billboards are sparsest where planes are needed
  02_points          cause     -- room0 at wall height: billboards vs rendered depth
  03_walls           result 1  -- wall planes found / one plane per wall
  04_pipeline        method    -- segmented planes -> merged walls -> cells -> floorplan
  05_floorplans      result 2  -- all five scenes over the GT footprint
  06_virtual_edges   caveat    -- billboards' polygons rest on virtual edges
  07_accuracy        result 3  -- boundary error and rebuilt wall area
  08_room3d          showcase  -- room0 rebuilt in 3-D, front walls cut away
  09_elevation       result 4  -- hidden lower walls rebuilt by extrusion
"""
import argparse, json, os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_floorplans as pf
from plot_floorplans import (Frame, gt_outline, edge_kinds, load_room, semantic_dir,
                             SURFACE, INK, INK2, MUTED, BLUE, ORANGE, GT_FILL, GT_EDGE, PTS)
from eval_walls_gt import load_semantic
from eval_floorplan_gt import sample_boundary, sample_triangles
from analyze_walls import load_labeled

OUT = f"{pf.ROOT}/BBSplat/output"
SCENES = pf.SCENES
GRAY = "#b4b3ad"          # de-emphasised bars: the comparison, not the point
W, H = 13.33, 7.5         # 16:9 inches

LABEL = {"bb_ransac": "빌보드 (기존)", "hybrid_cons": "혼합", "hybrid_cons_ref": "혼합 + 빌보드 보정",
         "gtdepth_cons": "GT 깊이 (오라클)"}


def setup(font):
    for f in (font, font.replace("NanumGothic.ttf", "NanumGothicBold.ttf")):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
    name = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.family": name, "font.size": 16, "axes.unicode_minus": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})


def headline(fig, title, sub=None):
    fig.text(0.035, 0.94, title, color=INK, fontsize=26, fontweight="bold", va="top")
    if sub:
        fig.text(0.035, 0.865, sub, color=INK2, fontsize=16, va="top")


def clean(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def J(scene, name):
    return json.load(open(f"{OUT}/replica_{scene}_v1/{name}.json"))


def total(name, key, scenes=SCENES):
    return sum(J(s, name)[key] if "." not in key else
               J(s, name)[key.split(".")[0]][key.split(".")[1]] for s in scenes)


# ---------------------------------------------------------------- 01
def fig_coverage(out):
    rows = [("천장", 8.9), ("벽", 27.5), ("문", 28.9), ("조명", 30.7), ("바닥", 30.9),
            ("소파", 44.8), ("의자", 49.9), ("수납장", 60.1), ("블라인드", 60.9),
            ("카펫", 70.3), ("테이블", 76.1), ("쿠션", 86.0)]
    arch = {"천장", "벽", "바닥"}
    fig = plt.figure(figsize=(W, H))
    headline(fig, "빌보드는 평면 분할에 필요한 표면에 가장 적다",
             "GT 표면 중 5 cm 안에 빌보드가 있는 비율 (Replica 5씬) — MCMC는 렌더링 오차가 큰 곳, 즉 무늬가 많은 곳에 빌보드를 놓는다")
    ax = fig.add_axes([0.14, 0.09, 0.80, 0.70])
    y = np.arange(len(rows))[::-1]
    for yi, (n, v) in zip(y, rows):
        c = BLUE if n in arch else GRAY
        ax.barh(yi, v, height=0.62, color=c, zorder=2)
        ax.text(v + 1.0, yi, f"{v:.1f}%", va="center", fontsize=15,
                color=INK if n in arch else INK2, fontweight="bold" if n in arch else "normal")
    ax.set_yticks(y, [n for n, _ in rows], fontsize=16)
    for t, (n, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(INK if n in arch else INK2)
        t.set_fontweight("bold" if n in arch else "normal")
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color="#e9e8e4", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    clean(ax, keep=())
    ax.text(0.99, 0.97, "건축 표면(파랑)이 가장 비어 있고,\n가구·소품이 가장 조밀하다\n\n쿠션의 커버리지가 천장의 약 10배",
            transform=ax.transAxes, ha="right", va="top", fontsize=16, color=INK2, linespacing=1.4)
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 02
def fig_points(out, scene="room0"):
    r = J(scene, "room_hybrid_cons_ref")
    F = Frame(r)
    up = np.asarray(r["up"]); fl = r["floor_height"]
    pc = f"{OUT}/replica_{scene}_v1/point_cloud/iteration_32000"
    bb = load_labeled(f"{pc}/planes_bb_ransac.ply")[0]
    dp = load_labeled(f"{pc}/planes_depth_pca.ply")[0]
    verts, tris, oid, cls = load_semantic(semantic_dir(scene))
    gt = F.from2(gt_outline(r, verts, tris, cls))
    fig = plt.figure(figsize=(W, H))
    headline(fig, "렌더링 깊이는 빌보드가 비워 둔 민벽을 채운다",
             f"{scene}, 벽 높이(바닥 위 0.3~2.3 m)의 점을 위에서 본 모습 — 방의 외곽선이 곧 벽이다")
    for k, (pts, name, cov) in enumerate([(bb, "빌보드 중심", "벽 커버리지 12.2%"),
                                          (dp, "렌더링 깊이 역투영", "벽 커버리지 45.6%")]):
        ax = fig.add_axes([0.04 + 0.48 * k, 0.05, 0.44, 0.70])
        h = pts @ up - fl
        q = F.from3(pts[(h > 0.3) & (h < 2.3)])
        ax.plot(*np.vstack([gt, gt[:1]]).T, color=GT_EDGE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.scatter(q[:, 0], q[:, 1], s=0.25, color=BLUE, alpha=0.35, lw=0, rasterized=True, zorder=2)
        c = 0.5 * (gt.min(0) + gt.max(0)); hx, hy = 0.5 * np.ptp(gt, axis=0) + 0.35
        ax.set_xlim(c[0] - hx, c[0] + hx); ax.set_ylim(c[1] - hy, c[1] + hy)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); clean(ax, keep=())
        ax.set_title(f"{name}  ·  {len(pts):,}점", loc="left", fontsize=19, fontweight="bold",
                     color=INK, pad=28)
        ax.text(0, 1.015, cov, transform=ax.transAxes, fontsize=16, color=INK2, va="bottom")
    fig.legend(handles=[Line2D([], [], color=GT_EDGE, lw=1.2, ls=(0, (4, 3)), label="GT 방 외곽선")],
               loc="lower right", frameon=False, fontsize=14, labelcolor=INK2)
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 03
def fig_walls(out):
    groups = [("빌보드 (기존)", "bb_ransac"), ("렌더링 깊이", "depth_pca"),
              ("깊이 + 벽 통합", "depth_pca_cons"), ("GT 깊이 (오라클)", "gtdepth_cons")]
    found = [total(f"geom_{a}", "matched") for _, a in groups]
    # single-plane 50% overlap (eval_walls_gt.py --min_overlap 0.5); the stored
    # eval_walls_*.json use the 25% default, so these come from that rerun
    one = [4, 6, 13, 18]
    fig = plt.figure(figsize=(W, H))
    headline(fig, "깊이로 벽을 찾고, 조각을 합쳐 한 벽을 한 평면으로",
             "GT 벽 22개 기준 (Replica 5씬)")
    ax = fig.add_axes([0.07, 0.10, 0.88, 0.62])
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, found, w, color=BLUE, zorder=2)
    b2 = ax.bar(x + w / 2 + 0.01, one, w, color=ORANGE, zorder=2)
    for bars, vals in ((b1, found), (b2, one)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}", ha="center", va="bottom",
                    fontsize=18, color=INK, fontweight="bold")
    ax.axhline(22, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(len(groups) - 0.5, 22.2, "GT 벽 22개", ha="right", va="bottom", fontsize=14, color=INK2)
    ax.set_xticks(x, [g for g, _ in groups], fontsize=17)
    ax.set_ylim(0, 24.5); ax.set_yticks([0, 5, 10, 15, 20])
    ax.grid(axis="y", color="#e9e8e4", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0); clean(ax, keep=("bottom",))
    fig.legend(handles=[Patch(color=BLUE, label="벽 평면을 찾음 (각도 20°·거리 30 cm 이내)"),
                        Patch(color=ORANGE, label="한 평면이 벽의 절반 이상을 덮음")],
               loc="upper left", bbox_to_anchor=(0.03, 0.83), ncol=2, frameon=False,
               fontsize=15, labelcolor=INK)
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 04
def seg_of(q, F, up):
    """2-D segment (fit line, 2.5-97.5 pct extent) of a vertical plane's points."""
    p = F.from3(q)
    c = p.mean(0)
    d = np.linalg.svd(p - c, full_matrices=False)[2][0]
    t = (p - c) @ d
    return c + np.percentile(t, 2.5) * d, c + np.percentile(t, 97.5) * d


def fig_pipeline(out, scene="room0"):
    r = J(scene, "room_hybrid_cons_ref")
    F = Frame(r); up = np.asarray(r["up"])
    pc = f"{OUT}/replica_{scene}_v1/point_cloud/iteration_32000"
    verts, tris, oid, cls = load_semantic(semantic_dir(scene))
    gt = F.from2(gt_outline(r, verts, tris, cls))
    c = 0.5 * (gt.min(0) + gt.max(0)); hx, hy = 0.5 * np.ptp(gt, axis=0) + 0.45

    fig = plt.figure(figsize=(W, H))
    headline(fig, "벽 평면에서 평면도까지", f"{scene}, 위에서 본 모습 — 점선: GT 방 외곽선")
    pos = [[0.10 + 0.44 * (k % 2), 0.45 - 0.395 * (k // 2), 0.38, 0.28] for k in range(4)]
    titles = []

    for k, (lab_file, walls_file) in enumerate([("planes_hybrid", "walls_hybrid"),
                                                ("planes_hybrid_cons", "walls_hybrid_cons")]):
        ax = fig.add_axes(pos[k])
        xyz, lab = load_labeled(f"{pc}/{lab_file}.ply")
        ids = J(scene, walls_file)["wall_ids"]
        ax.plot(*np.vstack([gt, gt[:1]]).T, color=GT_EDGE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for i in ids:
            a, b = seg_of(xyz[lab == i], F, up)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=2.4, alpha=0.75,
                    solid_capstyle="butt", zorder=2)
            ax.plot([a[0], b[0]], [a[1], b[1]], ls="", marker="|", ms=9, color=BLUE, zorder=3)
        titles.append((ax, ["1. 분할된 벽 평면", "2. 같은 벽의 조각 통합"][k], f"벽 판정 평면 {len(ids)}개"))

    # cells
    ax = fig.add_axes(pos[2])
    mask = np.load(f"{OUT}/replica_{scene}_v1/room_hybrid_cons_ref_mask.npy")
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cs:
        p = F.from2(cnt[:, 0, :] * res + lo)
        ax.fill(p[:, 0], p[:, 1], color=BLUE, alpha=0.14, lw=0, zorder=1)
    for ln in r["lines"]:
        n = F.from2(np.asarray(ln["n"])[None])[0]; p = F.from2(np.asarray(ln["p"])[None])[0]
        d = np.array([-n[1], n[0]])
        ax.plot(*np.stack([p - 9 * d, p + 9 * d]).T, color=BLUE if ln["wall_id"] is not None else ORANGE,
                lw=1.2, ls="-" if ln["wall_id"] is not None else (0, (3, 2)), zorder=2)
    titles.append((ax, "3. 벽을 연장해 셀로 나누고", "관측된 셀(채움)만 방으로"))

    ax = fig.add_axes(pos[3])
    g = gt
    ax.fill(g[:, 0], g[:, 1], color=GT_FILL, lw=0, zorder=1)
    ax.plot(*np.vstack([g, g[:1]]).T, color=GT_EDGE, lw=1.0, zorder=2)
    poly = F.from2(r["polygon"])
    ax.plot(*np.vstack([poly, poly[:1]]).T, color=BLUE, lw=2.6, zorder=3)
    f = J(scene, "floorplan_hybrid_cons_ref")
    titles.append((ax, "4. 평면도", f"GT와 경계 오차 {f['chamfer_cm']:.1f} cm"))

    for ax, t, s in titles:
        ax.set_xlim(c[0] - hx, c[0] + hx); ax.set_ylim(c[1] - hy, c[1] + hy)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); clean(ax, keep=())
        ax.set_title(t, loc="left", fontsize=17, fontweight="bold", color=INK, pad=28)
        ax.text(0, 1.02, s, transform=ax.transAxes, fontsize=14, color=INK2, va="bottom")
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 05 / 06
def fig_floorplans(out, sem):
    fig = plt.figure(figsize=(W, H))
    headline(fig, "재구성한 평면도 — 가려진 바닥까지", "혼합 평면 + 벽 통합 + 빌보드 보정, 모든 패널 같은 축척")
    rooms = [(s,) + load_room(s, "hybrid_cons_ref") for s in SCENES]
    half = pf.extents([r for _, r, _, _ in rooms])
    for k, (s, r, f, fp) in enumerate(rooms):
        ax = fig.add_axes([0.02 + 0.33 * (k % 3), 0.45 - 0.40 * (k // 3), 0.31, 0.32])
        cc, _ = pf.draw_room(ax, r, gt_outline(r, *sem[s]))
        pf.frame_axes(ax, cc, half)
        ax.set_title(s, loc="left", fontsize=18, fontweight="bold", color=INK, pad=6)
        ax.text(1.0, 1.02, f"IoU {100 * fp['iou']:.0f}%   경계 {f['chamfer_cm']:.1f} cm",
                transform=ax.transAxes, fontsize=14, color=INK2, va="bottom", ha="right")
    m = np.mean([J(s, "footprint_hybrid_cons_ref")["iou"] for s in SCENES])
    v = np.mean([J(s, "footprint_hybrid_cons_ref")["visible_floor_iou"] for s in SCENES])
    fig.text(0.70, 0.30, f"평균 IoU {100 * m:.1f}%", fontsize=24, color=INK, fontweight="bold")
    fig.text(0.70, 0.24, f"보이는 바닥만 쓰면 {100 * v:.0f}%", fontsize=17, color=INK2)
    fig.legend(handles=[Patch(facecolor=GT_FILL, edgecolor=GT_EDGE, label="GT 평면도 (천장 투영)"),
                        Line2D([], [], color=BLUE, lw=2.4, label="재구성한 평면도")],
               loc="upper left", frameon=False, fontsize=15, labelcolor=INK2,
               bbox_to_anchor=(0.69, 0.20))
    fig.savefig(out, dpi=150); plt.close(fig)


def fig_virtual(out, sem, scene="room0"):
    arms = ["bb_ransac", "hybrid_cons", "hybrid_cons_ref"]
    data = [load_room(scene, a) for a in arms]
    half = pf.extents([r for r, _, _ in data])
    fig = plt.figure(figsize=(W, H))
    headline(fig, "빌보드의 평면도는 벽이 아니라 점의 범위로 만들어진다",
             f"{scene} — 직사각형 방이라 가상 벽만으로도 맞지만, 실제로 찾은 벽은 없다")
    for k, (a, (r, f, fp)) in enumerate(zip(arms, data)):
        ax = fig.add_axes([0.03 + 0.325 * k, 0.15, 0.30, 0.56])
        cc, _ = pf.draw_room(ax, r, gt_outline(r, *sem[scene]), split_edges=True)
        pf.frame_axes(ax, cc, half)
        ax.set_title(LABEL[a], loc="left", fontsize=19, fontweight="bold", color=INK, pad=50)
        ax.text(0, 1.02, f"검출된 벽 위의 변 {100 * f['wall_backed']:.0f}%\n경계 오차 {f['chamfer_cm']:.1f} cm",
                transform=ax.transAxes, fontsize=15, color=INK2, va="bottom", linespacing=1.3)
    fig.legend(handles=[Patch(facecolor=GT_FILL, edgecolor=GT_EDGE, label="GT 평면도"),
                        Line2D([], [], color=BLUE, lw=2.4, label="검출된 벽 위의 변"),
                        Line2D([], [], color=ORANGE, lw=2.4, ls=(0, (3, 1.6)), label="가상 벽 (점의 범위)")],
               loc="lower center", ncol=3, frameon=False, fontsize=15, labelcolor=INK2)
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 07
def fig_accuracy(out):
    arms = ["bb_ransac", "hybrid_cons", "hybrid_cons_ref", "gtdepth_cons"]
    ch = [np.mean([J(s, f"floorplan_{a}")["chamfer_cm"] for s in SCENES]) for a in arms]
    rc = [100 * np.mean([J(s, f"floorplan_{a}")["wall_recall_3d"] for s in SCENES]) for a in arms]
    fig = plt.figure(figsize=(W, H))
    headline(fig, "깊이로 찾고 빌보드로 맞추면 둘 다를 이긴다",
             "Replica 5씬 평균 — 오라클(GT 깊이)은 참고용")
    for k, (vals, name, unit, better) in enumerate([(ch, "평면도 경계 오차", "cm", "낮을수록 좋음"),
                                                   (rc, "3D 벽 면적 복원 (10 cm 이내)", "%", "높을수록 좋음")]):
        ax = fig.add_axes([0.22 + 0.40 * k, 0.12, 0.34, 0.58])
        cols = [GRAY, GRAY, BLUE, "#d9d8d3"]
        y = np.arange(len(arms))[::-1]
        ax.barh(y, vals, height=0.6, color=cols, zorder=2)
        for yi, v, a in zip(y, vals, arms):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.1f} {unit}" if unit == "cm" else f"{v:.1f}{unit}",
                    va="center", fontsize=17,
                    color=INK if a == "hybrid_cons_ref" else INK2,
                    fontweight="bold" if a == "hybrid_cons_ref" else "normal")
        ax.set_yticks(y, [LABEL[a] for a in arms] if k == 0 else [""] * len(arms), fontsize=16)
        if k == 0:
            for t, a in zip(ax.get_yticklabels(), arms):
                t.set_color(INK if a == "hybrid_cons_ref" else INK2)
                t.set_fontweight("bold" if a == "hybrid_cons_ref" else "normal")
        ax.set_xlim(0, max(vals) * 1.25); ax.set_xticks([])
        ax.tick_params(length=0); clean(ax, keep=())
        ax.set_title(name, loc="left", fontsize=19, fontweight="bold", color=INK, pad=30)
        ax.text(0, 1.02, better, transform=ax.transAxes, fontsize=14, color=INK2, va="bottom")
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 08
def fig_room3d(out, scene="room0", azim=-58, elev=32):
    r = J(scene, "room_hybrid_cons_ref")
    F = Frame(r); up = np.asarray(r["up"]); fl, ce = r["floor_height"], r["ceiling_height"]
    d = PlyData.read(f"{OUT}/replica_{scene}_v1/point_cloud/iteration_32000/point_cloud.ply").elements[0].data
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    rgb = np.clip(0.5 + 0.28209479 * np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1), 0, 1)
    h = xyz @ up - fl
    poly = F.from2(r["polygon"])
    inside = cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (0, 0), False)
    p2 = F.from3(xyz)
    keep = (h > -0.05) & (h < (ce - fl) - 0.2)
    keep &= np.array([cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2),
                                           (float(x), float(y)), True) > -0.05 for x, y in p2])
    idx = np.where(keep)[0]
    idx = idx[np.random.default_rng(0).permutation(len(idx))[:90000]]

    fig = plt.figure(figsize=(W, H))
    headline(fig, "3D로 복원한 방 — 벽은 바닥부터 천장까지", f"{scene}: 학습된 빌보드(색) + 복원한 벽면과 바닥(파랑), 앞쪽 벽은 잘라 냄")
    ax = fig.add_axes([0.0, -0.06, 1.0, 0.98], projection="3d")
    ax.scatter(p2[idx, 0], p2[idx, 1], h[idx], s=0.6, c=rgb[idx], lw=0, depthshade=False, rasterized=True)
    # back walls filled, front walls only outlined (view direction from azimuth)
    v = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    Hh = ce - fl
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        outward = np.array([e[1], -e[0]]) * (1 if area > 0 else -1)
        quad = [[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], Hh], [a[0], a[1], Hh]]
        if outward @ v < 0:        # faces away from the viewer: a back wall
            ax.add_collection3d(Poly3DCollection([quad], facecolor=BLUE, alpha=0.18, edgecolor=BLUE, lw=1.2))
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], [0, 0], color=BLUE, lw=2.0)
    ax.add_collection3d(Poly3DCollection([[[x, y, 0] for x, y in poly]], facecolor=BLUE, alpha=0.08,
                                         edgecolor=BLUE, lw=2.0))
    lo, hi = poly.min(0), poly.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(0, Hh)
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], Hh))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 09
def fig_elevation(out, sem_full, scene="room0", dist=0.10):
    r, f, _ = load_room(scene, "hybrid_cons_ref")
    verts, tris, oid, cls = sem_full[scene]
    up, e1, e2 = (np.asarray(r[k]) for k in ("up", "e1", "e2"))
    floor, ceil = r["floor_height"], r["ceiling_height"]
    S, _ = sample_triangles(verts[tris[cls == "wall"]], 2500.0, np.random.default_rng(0))
    S2 = np.stack([S @ e1, S @ e2], 1); hs = S @ up - floor
    gt = gt_outline(r, verts, tris, cls)
    B = sample_boundary(gt, 0.01)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
    u = arc[cKDTree(B).query(S2)[1]]
    lo = np.asarray(r["raster"]["lo"]); res = r["raster"]["res"]
    contour = np.round((np.asarray(r["polygon"]) - lo) / res).astype(np.int32).reshape(-1, 1, 2)
    px = (S2 - lo) / res
    d2 = np.array([abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True)) for x, y in px]) * res
    dh = np.maximum(0, np.maximum(-hs, hs - (ceil - floor)))
    ext = np.hypot(d2, dh) < dist
    xyz, lab = load_labeled(r["labeled"])
    Wp = xyz[np.isin(lab, json.load(open(r["walls"]))["wall_ids"])]
    obs = cKDTree(Wp).query(S, distance_upper_bound=dist)[0] < dist

    fig = plt.figure(figsize=(W, H))
    headline(fig, "가구에 가려진 벽 아랫부분까지 복원", f"{scene}의 GT 벽을 방 둘레를 따라 펼친 그림 — 파랑: 복원(10 cm 이내), 주황: 놓침")
    for k, (hit, name) in enumerate([(obs, "분할만 (관측된 벽 점)"), (ext, "평면도의 벽을 바닥~천장으로 세움")]):
        ax = fig.add_axes([0.07, 0.47 - 0.37 * k, 0.90, 0.27])
        o = np.random.default_rng(1).permutation(len(u))
        ax.scatter(u[o], hs[o], s=0.9, lw=0, c=np.where(hit[o], BLUE, ORANGE), rasterized=True)
        ax.set_title(f"{name}:  GT 벽 면적의 {100 * hit.mean():.0f}%", loc="left", fontsize=18,
                     fontweight="bold", color=INK, pad=8)
        ax.set_ylim(-0.1, (ceil - floor) + 0.1); ax.set_yticks([0, 1, 2])
        ax.set_ylabel("높이 (m)", fontsize=14)
        clean(ax); ax.tick_params(labelsize=13)
        if k == 1:
            ax.set_xlabel("방 둘레를 따른 위치 (m)", fontsize=14)
        else:
            ax.set_xticklabels([])
    fig.axes[-1].text(1.0, 1.04, "5~12 m: 창이 있는 벽이 블라인드 면(벽 14 cm 앞)으로 잡혀 놓침",
                      transform=fig.axes[-1].transAxes, ha="right", va="bottom", fontsize=13, color=INK2)
    fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/slides")
    ap.add_argument("--font", default=os.path.expanduser("~/.local/share/fonts/NanumGothic.ttf"))
    ap.add_argument("--only", default=None, help="comma-separated figure numbers, e.g. 01,08")
    a = ap.parse_args()
    setup(a.font)
    os.makedirs(a.out, exist_ok=True)
    want = set(a.only.split(",")) if a.only else None
    sem_full = {s: load_semantic(semantic_dir(s)) for s in SCENES}
    sem = {s: (v, t, c) for s, (v, t, o, c) in sem_full.items()}
    jobs = [("01", "01_coverage", lambda p: fig_coverage(p)),
            ("02", "02_points", lambda p: fig_points(p)),
            ("03", "03_walls", lambda p: fig_walls(p)),
            ("04", "04_pipeline", lambda p: fig_pipeline(p)),
            ("05", "05_floorplans", lambda p: fig_floorplans(p, sem)),
            ("06", "06_virtual_edges", lambda p: fig_virtual(p, sem)),
            ("07", "07_accuracy", lambda p: fig_accuracy(p)),
            ("08", "08_room3d", lambda p: fig_room3d(p)),
            ("09", "09_elevation", lambda p: fig_elevation(p, sem_full))]
    for k, name, fn in jobs:
        if want and k not in want:
            continue
        fn(os.path.join(a.out, name + ".png"))
        print("wrote", name)


if __name__ == "__main__":
    main()
