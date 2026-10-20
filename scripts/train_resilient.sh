#!/usr/bin/env bash
# Train one Replica scene, resuming after the WSL2 GPU faults.
#
# On this machine training dies around iter 7000-10000 with
#   RuntimeError: CUDA driver error: device not ready
# and the kernel log shows `dxgkio_make_resident: Ioctl failed: -12` (ENOMEM):
# WSL2's GPU paging cannot keep the working set resident once MCMC has grown
# the model to cap_max primitives. It killed room2, office0 and office1 in a
# row. The fault is transient, so checkpoint often and restart from the last
# one instead of from scratch.
set -u
SCENE=${1:?scene}
TAG=${2:-v1}
TARGET=${3:-32000}
PY=/home/cot0514/anaconda3/envs/bbsplat/bin/python
ROOT=/home/cot0514/projects/3DGS
OUT=$ROOT/BBSplat/output/replica_${SCENE}_${TAG}
DATA=$ROOT/data/replica_colmap/$SCENE
cd $ROOT/BBSplat

CKPTS="2000 4000 6000 7000 8000 9000 10000 14000 18000 22000 26000 30000 $TARGET"
DONE=$OUT/point_cloud/iteration_$TARGET/point_cloud.ply

for attempt in $(seq 1 40); do
  [ -f "$DONE" ] && { echo "[$SCENE] reached $TARGET"; break; }
  LAST=$(ls -1 $OUT/chkpnt*.pth 2>/dev/null | sed 's/.*chkpnt//; s/\.pth//' | sort -n | tail -1)
  if [ -n "${LAST:-}" ]; then
    RESUME="--start_checkpoint $OUT/chkpnt${LAST}.pth"
    echo "[$SCENE] attempt $attempt: resuming from iter $LAST ($(date +%H:%M:%S))"
  else
    RESUME=""
    echo "[$SCENE] attempt $attempt: starting from scratch ($(date +%H:%M:%S))"
  fi
  $PY train.py -s "$DATA" --model_path="$OUT" \
      --cap_max=200_000 --max_read_points=180_000 --eval \
      --densify_until_iter 10000 --iterations $TARGET \
      --rotation_init pca --normal_from_iter 3000 --lambda_normal 0.05 \
      --checkpoint_iterations $CKPTS $RESUME
  rc=$?
  echo "[$SCENE] train.py exited with $rc"
  # keep only the two newest checkpoints; each is a few GB
  ls -1t $OUT/chkpnt*.pth 2>/dev/null | tail -n +3 | xargs -r rm -f
  [ -f "$DONE" ] && { echo "[$SCENE] reached $TARGET"; break; }
  sleep 30
done
[ -f "$DONE" ] || { echo "[$SCENE] GAVE UP after $attempt attempts"; exit 1; }
rm -f $OUT/chkpnt*.pth
echo "[$SCENE] training complete $(date +%H:%M:%S)"
