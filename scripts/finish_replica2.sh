#!/usr/bin/env bash
# Train the three scenes that the WSL2 GPU fault killed, resuming across
# crashes, then run the full evaluation for all five.
cd /home/cot0514/projects/3DGS/BBSplat
for s in room2 office0 office1; do
  echo; echo "################ TRAIN $s $(date +%H:%M:%S) ################"
  bash scripts/train_resilient.sh "$s" v1 32000 || echo "!!!! $s training gave up"
done
for s in room2 office0 office1 room0 room1; do
  echo; echo "################ EVAL $s $(date +%H:%M:%S) ################"
  bash scripts/run_replica_scene.sh "$s" v1 || echo "!!!! $s eval FAILED, continuing"
done
echo "################ ALL DONE $(date) ################"
