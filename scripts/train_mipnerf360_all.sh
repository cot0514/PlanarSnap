#!/bin/bash
# Mip-NeRF 360 전체 씬 순차 학습 (room 제외 — 이미 v15로 완료)
# Indoor: cap_max=200k / Outdoor: cap_max=300k
# - 이미 results.json이 있으면 해당 씬 건너뜀
# - 한 씬 실패해도 다음 씬 계속 진행

DATA=/home/cot0514/projects/3DGS/data
OUT=/home/cot0514/projects/3DGS/BBSplat/output
CD=/home/cot0514/projects/3DGS/BBSplat

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_scene() {
    local scene=$1
    local cap=$2
    local read_pts=$3
    local out_dir=$OUT/${scene}_v15

    if [ -f "$out_dir/results.json" ]; then
        log "=== SKIP (already done): $scene ==="
        return 0
    fi

    log "=== START: $scene (cap_max=$cap) ==="
    python $CD/train.py \
        -s $DATA/$scene \
        --model_path=$out_dir \
        --cap_max=$cap \
        --max_read_points=$read_pts \
        --add_sky_box \
        --eval \
        --densify_until_iter 10000 || { log "=== TRAIN FAILED: $scene ==="; return 1; }

    log "=== TRAIN DONE: $scene, rendering... ==="
    python $CD/render.py -m $out_dir -s $DATA/$scene --skip_mesh || { log "=== RENDER FAILED: $scene ==="; return 1; }

    log "=== RENDER DONE: $scene, computing metrics... ==="
    python $CD/metrics.py -m $out_dir || { log "=== METRICS FAILED: $scene ==="; return 1; }

    log "=== COMPLETE: $scene ==="
}

cd $CD

# Indoor scenes (cap_max=200k)
run_scene counter  200000 180000 || true
run_scene bonsai   200000 180000 || true
run_scene kitchen  200000 180000 || true

# Outdoor scenes (cap_max=300k)
run_scene garden   300000 290000 || true
run_scene bicycle  300000 290000 || true
run_scene stump    300000 290000 || true

log "=== ALL SCENES DONE ==="
