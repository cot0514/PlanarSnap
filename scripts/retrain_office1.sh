#!/usr/bin/env bash
# office1's texture alpha collapsed to zero between iter 1000 and 7000, so the
# model renders nothing (PSNR 15.9, SSIM 0.086) and every later checkpoint is
# dead. Retrain it once the main queue is free.
cd /home/cot0514/projects/3DGS/BBSplat
while pgrep -f "[f]inish_replica.sh" >/dev/null; do sleep 120; done
echo "################ office1 retrain $(date +%H:%M:%S) ################"
bash scripts/run_replica_scene.sh office1 v1
echo "################ office1 done $(date) ################"
