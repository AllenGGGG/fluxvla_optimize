#!/bin/bash
# 对比标准版 vs 优化版（Ultra Fusion）

setsid /home/guohao/miniconda3/envs/fluxvla/bin/python \
scripts/compare_pi05_task0_with_l_pixshuffle.py \
--tag standard_vs_ultra \
--variant standard_baseline \
configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py \
work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--variant ultra_optimized \
configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--base-weights work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--eval-speeds 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
--task-id 4 \
--success-trials-per-task 50 \
--success-seeds 7 \
--success-gpus 2,3,4,5,6,7 \
--success-nproc-per-node 6 \
--speed-single-gpus 2 \
--speed-multi-gpus 2,3,4,5,6,7 \
--speed-warmup-iters 10 \
--speed-bench-iters 50 \
> logs/standard_vs_ultra_compare.log 2>&1 < /dev/null &

echo "测试启动！日志: logs/standard_vs_ultra_compare.log"
echo ""
echo "对比内容:"
echo "  标准版: Triton + CUDA Graph + Speed 缓存 (无 Ultra Fusion)"
echo "  优化版: 标准版 + Ultra Fusion Kernels"
echo ""
echo "查看日志: tail -f logs/standard_vs_ultra_compare.log"
