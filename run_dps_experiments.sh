#!/bin/bash

# DPS experiments across 8 GPUs
# Each experiment runs on a different GPU in parallel
# Note: DPS uses 1000 sampling steps and sigma_y = 0.05 (noisy measurements)

# DPS parameters
SIGMA_Y=0.00
ZETA_DPS=1.0
TIMESTEPS=1000

pids=()
cleanup() {
    echo "Interrupt received, killing children..."
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

echo "Starting DPS experiments with sigma_y = $SIGMA_Y, zeta_dps = $ZETA_DPS"
echo "Tasks: SR×4, SR×8, 80% random inpainting, 128×128 box inpainting"
echo "Datasets: CelebA_HQ and ImageNet"
echo ""

# CelebA_HQ experiments
echo "Starting CelebA_HQ experiments..."

CUDA_VISIBLE_DEVICES=0 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset CelebA_HQ \
    --degradation SR \
    --scale_factor 4 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_celeba_sr4.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=1 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset CelebA_HQ \
    --degradation SR \
    --scale_factor 8 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_celeba_sr8.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=2 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset CelebA_HQ \
    --degradation Inpainting \
    --mask_type random \
    --random_amount 0.8 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_celeba_random.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=3 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset CelebA_HQ \
    --degradation Inpainting \
    --mask_type box \
    --box_indices 64 64 128 128 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_celeba_box.log 2>&1 &
pids+=($!)

# ImageNet experiments
echo "Starting ImageNet experiments..."

CUDA_VISIBLE_DEVICES=4 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset ImageNet \
    --degradation SR \
    --scale_factor 4 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_imagenet_sr4.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=5 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset ImageNet \
    --degradation SR \
    --scale_factor 8 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_imagenet_sr8.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=6 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset ImageNet \
    --degradation Inpainting \
    --mask_type random \
    --random_amount 0.8 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_imagenet_random.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=7 python -u hw4_step1_main.py \
    --ps_type DPS \
    --dataset ImageNet \
    --degradation Inpainting \
    --mask_type box \
    --box_indices 64 64 128 128 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$TIMESTEPS" \
    --zeta_dps "$ZETA_DPS" \
    > /tmp/dps_imagenet_box.log 2>&1 &
pids+=($!)

# Wait for all experiments to complete
echo ""
echo "All experiments started. Waiting for completion..."
echo "Logs are being written to /tmp/dps_*.log"
wait

echo ""
echo "=========================================="
echo "All DPS experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  results/step1_results/CelebA_HQ/DPS/"
echo "  results/step1_results/ImageNet/DPS/"
echo ""
echo "To view logs:"
echo "  cat /tmp/dps_celeba_sr4.log"
echo "  cat /tmp/dps_celeba_sr8.log"
echo "  cat /tmp/dps_celeba_random.log"
echo "  cat /tmp/dps_celeba_box.log"
echo "  cat /tmp/dps_imagenet_sr4.log"
echo "  cat /tmp/dps_imagenet_sr8.log"
echo "  cat /tmp/dps_imagenet_random.log"
echo "  cat /tmp/dps_imagenet_box.log"
