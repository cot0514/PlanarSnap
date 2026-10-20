#!/usr/bin/env bash
# Re-initialise and retrain the scenes whose COLMAP cloud was too sparse.
#
# MCMC has to grow the model from the triangulated points up to cap_max.
# Scenes that started from 34k-47k points trained fine; those that started
# from 8k-18k had to be cloned 11-25x and diverged (room2 fell to PSNR 11.4 by
# iter 7000, office1's texture alpha collapsed to zero). Replica's office
# scenes are mostly plain paint, so SIFT at its default peak threshold finds
# very few keypoints. Lowering it to 0.002 and raising the feature cap takes
# office1 from 7,860 to 50,344 points at the same 200 frames, with a better
# reprojection error (0.349 px vs 0.497).
set -u
cd /home/cot0514/projects/3DGS/BBSplat
PY=/home/cot0514/anaconda3/envs/bbsplat/bin/python
ROOT=/home/cot0514/projects/3DGS
for s in office1 office0 room2; do
  echo; echo "################ RECONVERT $s $(date +%H:%M:%S) ################"
  rm -rf $ROOT/data/replica_colmap/$s
  $PY scripts/replica_to_colmap.py --replica_root $ROOT/data/replica/Replica \
      --out_root $ROOT/data/replica_colmap --scenes "$s" --stride 10 \
      --peak_threshold 0.002 --max_features 16384 2>&1 | grep -avE "^I2026|^W2026"
  echo; echo "################ TRAIN $s $(date +%H:%M:%S) ################"
  rm -rf output/replica_${s}_v1
  bash scripts/train_resilient.sh "$s" v1 32000 || echo "!!!! $s training gave up"
done
echo "################ THREE SCENES TRAINED $(date) ################"
for s in room2 office0 office1; do
  echo; echo "################ EVAL $s $(date +%H:%M:%S) ################"
  bash scripts/run_replica_scene.sh "$s" v1 || echo "!!!! $s eval FAILED, continuing"
done
echo "################ ALL DONE $(date) ################"
