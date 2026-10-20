#!/usr/bin/env bash
# Wait for the room0 training already in flight, then run the full pipeline for
# the five evaluation scenes.  run_replica_scene.sh skips any step whose output
# exists, so room0 picks up at rendering.
cd /home/cot0514/projects/3DGS/BBSplat
PLY=output/replica_room0_v1/point_cloud/iteration_32000/point_cloud.ply
echo "waiting for room0 training to finish ($(date +%H:%M:%S))"
for i in $(seq 1 480); do
  if [ -f "$PLY" ]; then
    a=$(stat -c %s "$PLY"); sleep 20; b=$(stat -c %s "$PLY")
    [ "$a" = "$b" ] && break           # file size settled = writing finished
  fi
  sleep 20
done
[ -f "$PLY" ] || { echo "room0 checkpoint never appeared; aborting"; exit 1; }
echo "room0 checkpoint present ($(date +%H:%M:%S))"
for s in room0 room1 room2 office0 office1; do
  echo; echo "################ $s ################"
  bash scripts/run_replica_scene.sh "$s" v1 || echo "!!!! $s FAILED, continuing"
done
echo "################ all scenes done $(date) ################"
