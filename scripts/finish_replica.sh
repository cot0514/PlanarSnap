#!/usr/bin/env bash
# Finish the five-scene Replica evaluation.
# Order: evaluate the scenes that are already trained (fast, gives a partial
# table soon), then retrain the two that stopped at iter 7000 and evaluate
# those.  run_replica_scene.sh skips any step whose output already exists.
cd /home/cot0514/projects/3DGS/BBSplat
for s in office1 room1 room0 room2 office0; do
  echo; echo "################ $s $(date +%H:%M:%S) ################"
  bash scripts/run_replica_scene.sh "$s" v1 || echo "!!!! $s FAILED, continuing"
done
echo "################ ALL FIVE SCENES DONE $(date) ################"
