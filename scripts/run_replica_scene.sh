#!/usr/bin/env bash
# End-to-end PlanarSnap pipeline for one Replica scene.
#   scripts/run_replica_scene.sh room0 [tag]
# Steps run in order and each is skipped if its output already exists, so the
# script can be re-run after an interruption.
set -eu
SCENE=${1:?scene name, e.g. room0}
TAG=${2:-v1}
PY=/home/cot0514/anaconda3/envs/bbsplat/bin/python
ROOT=/home/cot0514/projects/3DGS
REPO=$ROOT/BBSplat
DATA=$ROOT/data/replica_colmap/$SCENE
MESH=$ROOT/data/replica/cull_replica_mesh/$SCENE.ply
UNSEEN=$ROOT/data/replica/cull_replica_mesh/${SCENE}_pc_unseen.npy
OUT=$REPO/output/replica_${SCENE}_${TAG}
cd "$REPO"

say() { echo; echo "=== [$SCENE] $* ($(date +%H:%M:%S))"; }

if [ ! -f "$DATA/sparse/0/points3D.bin" ]; then
  say "convert"
  $PY scripts/replica_to_colmap.py \
      --replica_root $ROOT/data/replica/Replica \
      --out_root $ROOT/data/replica_colmap --scenes "$SCENE" --stride 10
fi

if [ ! -f "$OUT/point_cloud/iteration_32000/point_cloud.ply" ]; then
  say "train"
  # no --add_sky_box: Replica rooms are closed, every ray hits a surface
  $PY train.py -s "$DATA" --model_path="$OUT" \
      --cap_max=200_000 --max_read_points=180_000 --eval \
      --densify_until_iter 10000 \
      --rotation_init pca --normal_from_iter 3000 --lambda_normal 0.05
fi

if [ ! -d "$OUT/test" ]; then
  say "render + metrics"
  $PY render.py -m "$OUT" -s "$DATA" --skip_mesh
  $PY metrics.py -m "$OUT" || true
fi

if [ ! -f "$OUT/depth_cloud_v05.npz" ]; then
  say "depth fusion (5 cm)"
  $PY scripts/depth_fusion.py -m "$OUT" -s "$DATA" \
      --voxel 0.05 --pixel_stride 2 --clip_billboards 0.3 \
      --out "$OUT/depth_cloud_v05.ply"
fi

DIST=$($PY -c "
import sys; sys.path.insert(0,'$REPO')
from segment_planes import scene_extent, load_billboards
x,_ = load_billboards('$OUT/point_cloud/iteration_32000/point_cloud.ply')
print(round(scene_extent(x)*0.005, 4))")

NB=$($PY -c "from plyfile import PlyData;print(len(PlyData.read('$OUT/point_cloud/iteration_32000/point_cloud.ply').elements[0].data))")
ND=$($PY -c "import numpy;print(len(numpy.load('$OUT/depth_cloud_v05.npz')['xyz']))")
# ask both point sets for planes of the same physical size
MPS=$($PY -c "print(max(200, round(200*$ND/$NB)))")
echo "    shared plane-fit threshold $DIST | billboards $NB | depth $ND (--min_plane_samples $MPS)"

# 2x2: point set (billboards vs depth fusion) x algorithm (position-only RANSAC
# vs normal-constrained RANSAC).  bb_ransac is the method all the earlier
# results were reported with; the grid separates "did the extra points help"
# from "did the depth-derived normals help".
# --no_skybox_filter: Replica rooms are closed and were trained without a
# sky-box, and that filter's radius is measured from the coordinate origin,
# which Replica puts at a room corner.
PC="$OUT/point_cloud/iteration_32000"
for arm in bb_ransac bb_pca depth_ransac depth_pca; do
  case $arm in
    bb_*)    SRC=""                                        ; N=200   ;;
    depth_*) SRC="--depth_cloud $OUT/depth_cloud_v05.npz"  ; N=$MPS  ;;
  esac
  case $arm in
    *_ransac) METH=ransac ;;
    *_pca)    METH=pca    ;;
  esac
  say "segment: $arm"
  $PY segment_planes.py -m "$OUT" $SRC --method $METH \
      --dist "$DIST" --min_plane_samples "$N" --max_planes 80 \
      --no_skybox_filter --no_color_split --no_vis --save
  mv "$PC/planes.ply" "$PC/planes_$arm.ply"
done

# --- hybrid: depth finds the planes, billboards refine the plane equations ---
say "segment: hybrid"
$PY scripts/hybrid_planes.py -m "$OUT" --labeled "$PC/planes_depth_pca.ply" \
    --depth_cloud "$OUT/depth_cloud_v05.npz" --dist "$DIST" \
    --out "$PC/planes_hybrid.ply" | tail -3

# --- oracle: the same pipeline on Replica's own depth maps, to separate
# "the segmentation cannot do it" from "the model's depth is not good enough" ---
if [ ! -f "$OUT/gtdepth.npz" ]; then
  say "GT-depth oracle fusion"
  $PY scripts/gt_depth_fusion.py --scene "$SCENE" --voxel 0.05 --pixel_stride 2 \
      --out "$OUT/gtdepth.ply" | tail -2
fi
say "segment: gtdepth"
$PY segment_planes.py -m "$OUT" --depth_cloud "$OUT/gtdepth.npz" --method pca \
    --dist "$DIST" --min_plane_samples 200 --max_planes 80 \
    --no_skybox_filter --no_color_split --no_vis --save 2>&1 | grep -E "Planes detected|Assigned "
mv "$PC/planes.ply" "$PC/planes_gtdepth.ply"

BASE="bb_ransac bb_pca depth_ransac depth_pca hybrid gtdepth"

# --- floorplan step 1: merge each wall's fragments into one plane.  Applied
# to every arm, oracle included, so the *_cons arms compare like for like ---
say "consolidate walls"
for arm in $BASE; do
  $PY scripts/consolidate_walls.py --labeled "$PC/planes_$arm.ply" \
      --cameras "$OUT/cameras.json" --out "$PC/planes_${arm}_cons.ply" | grep -E "^merge"
done

ARMS="$BASE $(for x in $BASE; do printf '%s_cons ' $x; done)"

say "walls"
for arm in $ARMS; do
  echo "--- $arm"
  $PY scripts/analyze_walls.py --labeled "$PC/planes_$arm.ply" \
      --cameras "$OUT/cameras.json" --json "$OUT/walls_$arm.json" | tail -5
done

say "GT normal accuracy"
$PY scripts/eval_normals_gt.py -m "$OUT" --mesh "$MESH" --unseen "$UNSEEN" \
    --pca --json "$OUT/eval_normals.json" | tail -8
$PY scripts/eval_normals_gt.py -m "$OUT" --mesh "$MESH" --unseen "$UNSEEN" \
    --depth_cloud "$OUT/depth_cloud_v05.npz" --json "$OUT/eval_normals_depth.json" | tail -7

SEM=$ROOT/data/replica/official/semantic/$(echo "$SCENE" | sed -E 's/([a-z]+)([0-9]+)/\1_\2/')/habitat
if [ -f "$SEM/mesh_semantic.ply" ]; then
  say "GT up / floor height"
  for arm in $ARMS; do
    echo "--- $arm"
    $PY scripts/eval_floor_gt.py --walls "$OUT/walls_$arm.json" --semantic "$SEM" \
        --json "$OUT/floor_$arm.json" | head -2
  done
  say "room polygon + GT footprint"
  for arm in $ARMS; do
    echo "--- $arm"
    $PY scripts/room_polygon.py --labeled "$PC/planes_$arm.ply" --walls "$OUT/walls_$arm.json" \
        --json "$OUT/room_$arm.json" --png "$OUT/room_$arm.png" | grep -E "wall lines"
    $PY scripts/eval_footprint_gt.py --room "$OUT/room_$arm.json" --semantic "$SEM" \
        --json "$OUT/footprint_$arm.json" | head -1
    $PY scripts/eval_floorplan_gt.py --room "$OUT/room_$arm.json" --semantic "$SEM" \
        --json "$OUT/floorplan_$arm.json"
  done
  # depth walls re-placed with the positionally accurate billboards
  for arm in depth_ransac depth_pca hybrid depth_ransac_cons depth_pca_cons hybrid_cons; do
    echo "--- ${arm}_ref"
    $PY scripts/room_polygon.py --labeled "$PC/planes_$arm.ply" --walls "$OUT/walls_$arm.json" \
        --refine_cloud "$PC/planes_bb_ransac.ply" \
        --json "$OUT/room_${arm}_ref.json" --png "$OUT/room_${arm}_ref.png" | grep -E "wall lines"
    $PY scripts/eval_footprint_gt.py --room "$OUT/room_${arm}_ref.json" --semantic "$SEM" \
        --json "$OUT/footprint_${arm}_ref.json" | head -1
    $PY scripts/eval_floorplan_gt.py --room "$OUT/room_${arm}_ref.json" --semantic "$SEM" \
        --json "$OUT/floorplan_${arm}_ref.json"
  done
  say "GT wall detection (point level)"
  for arm in $ARMS; do
    echo "--- $arm"
    $PY scripts/eval_walls_gt.py --labeled "$PC/planes_$arm.ply" \
        --walls "$OUT/walls_$arm.json" --semantic "$SEM" \
        --json "$OUT/eval_walls_$arm.json" | grep -E "^      wall|instances recovered|majority GT"
  done
  say "GT wall geometry (plane level)"
  for arm in $ARMS; do
    echo "--- $arm"
    $PY scripts/eval_wall_geometry_gt.py --labeled "$PC/planes_$arm.ply" \
        --walls "$OUT/walls_$arm.json" --semantic "$SEM" \
        --json "$OUT/geom_$arm.json" | tail -4
  done
else
  echo "no semantic labels at $SEM - skipping the GT wall metrics"
fi

say "done"
