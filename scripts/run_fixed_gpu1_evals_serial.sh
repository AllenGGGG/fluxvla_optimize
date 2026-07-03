#!/usr/bin/env bash
set -euo pipefail

cd /home/guohao/FluxVLA

PYTHON=/home/guohao/miniconda3/envs/fluxvla/bin/python
COMPARE=scripts/compare_pi05_task0_with_l_pixshuffle.py
ROOT_OUT=work_dirs/fixed_compare
ROOT_LOG=logs/fixed_gpu1_serial.log

mkdir -p logs "$ROOT_OUT"

run_task() {
  local name="$1"
  shift

  echo "[$(date '+%F %T')] START ${name}"
  "$@"
  echo "[$(date '+%F %T')] DONE  ${name}"
}

{
  echo "[$(date '+%F %T')] Fixed GPU1 serial evals started"
  echo "[$(date '+%F %T')] CUDA target: GPU 1, LIBERO task id: 4"

  run_task baseline_pixshuffle_mean \
    "$PYTHON" "$COMPARE" \
      --tag baseline_pixshuffle_mean \
      --output-dir "$ROOT_OUT/baseline_pixshuffle_mean" \
      --with-l-ckpt work_dirs/pi05_libero10_task0_with_l/checkpoints/latest-checkpoint.safetensors \
      --pixshuffle-ckpt work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/latest-checkpoint.safetensors \
      --task-id 4 \
      --success-gpus 1 \
      --success-nproc-per-node 1 \
      --speed-single-gpus 1 \
      --skip-speed-multi \
      --success-trials-per-task 50 \
      --success-seeds 7,8 \
      --speed-warmup-iters 10 \
      --speed-bench-iters 100 \
      --master-port 29680

  run_task pixshuffle_mlp \
    "$PYTHON" "$COMPARE" \
      --tag pixshuffle_mlp \
      --output-dir "$ROOT_OUT/pixshuffle_mlp" \
      --variant pixshuffle_mlp configs/pi05/pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py work_dirs/pi05_libero10_task0_with_l_pixshuffle_mlp_v1/checkpoints/latest-checkpoint.safetensors \
      --task-id 4 \
      --success-gpus 1 \
      --success-nproc-per-node 1 \
      --speed-single-gpus 1 \
      --skip-speed-multi \
      --success-trials-per-task 50 \
      --success-seeds 7,8 \
      --speed-warmup-iters 10 \
      --speed-bench-iters 100 \
      --master-port 29680

  run_task outline_vsta \
    "$PYTHON" "$COMPARE" \
      --tag outline_vsta \
      --output-dir "$ROOT_OUT/outline_vsta" \
      --variant outline configs/pi05/pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py work_dirs/pi05_libero10_task0_tempovla_pixshuffle_mlp_vsta_videos/checkpoints/latest-checkpoint.safetensors \
      --variant vsta configs/pi05/pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py work_dirs/vsta2_tempovla/checkpoints/latest-checkpoint.safetensors \
      --eval-speeds 0.5,1.0,1.25,1.5,2.0 \
      --task-id 4 \
      --success-gpus 1 \
      --success-nproc-per-node 1 \
      --speed-single-gpus 1 \
      --skip-speed-multi \
      --success-trials-per-task 50 \
      --success-seeds 7,8 \
      --speed-warmup-iters 10 \
      --speed-bench-iters 100 \
      --master-port 29680

  echo "[$(date '+%F %T')] All fixed GPU1 serial evals completed"
} 2>&1 | tee -a "$ROOT_LOG"
