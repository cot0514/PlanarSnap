#!/usr/bin/env bash
# Extra arm: normal-constrained RANSAC on the billboards, but using the
# billboard's own stored quaternion normal instead of a local k-NN PCA fit.
# Against the Replica GT mesh the stored normal is 20.7-25.6 deg off and local
# PCA is 31.4-44.1, so the pipeline has been using the worse of the two.
set -u
cd /home/cot0514/projects/3DGS/BBSplat
PY=/home/cot0514/anaconda3/envs/bbsplat/bin/python
ROOT=/home/cot0514/projects/3DGS
for SCENE in room0 room1 room2 office0 office1; do
  OUT=output/replica_${SCENE}_v1
  PC=$OUT/point_cloud/iteration_32000
  SEM=$ROOT/data/replica/official/semantic/$(echo "$SCENE" | sed -E 's/([a-z]+)([0-9]+)/\1_\2/')/habitat
  DIST=$($PY -c "
import sys; sys.path.insert(0,'.')
from segment_planes import scene_extent, load_billboards
x,_ = load_billboards('$PC/point_cloud.ply'); print(round(scene_extent(x)*0.005,4))")
  echo; echo "######## $SCENE (dist $DIST)"
  $PY segment_planes.py -m "$OUT" --method pca --normal_source own \
      --dist "$DIST" --min_plane_samples 200 --max_planes 80 \
      --no_skybox_filter --no_color_split --no_vis --save 2>&1 \
      | grep -E "normal source|Planes detected|Assigned "
  mv "$PC/planes.ply" "$PC/planes_bb_own.ply"
  $PY scripts/analyze_walls.py --labeled "$PC/planes_bb_own.ply" \
      --cameras "$OUT/cameras.json" --json "$OUT/walls_bb_own.json" \
      | grep -E "architectural|distinct orient"
  $PY scripts/eval_walls_gt.py --labeled "$PC/planes_bb_own.ply" \
      --walls "$OUT/walls_bb_own.json" --semantic "$SEM" \
      --json "$OUT/eval_walls_bb_own.json" | grep -E "^      wall|instances recovered"
  $PY scripts/eval_wall_geometry_gt.py --labeled "$PC/planes_bb_own.ply" \
      --walls "$OUT/walls_bb_own.json" --semantic "$SEM" \
      --json "$OUT/geom_bb_own.json" | tail -3
done
echo "######## OWN-NORMAL ARM DONE"
