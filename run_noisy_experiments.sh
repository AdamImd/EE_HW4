#!/bin/bash

# Noisy experiments (sigma_y = 0.05) for ILVR, MCG, and DDNM
# Tasks: SR×4 and box inpainting
# Dataset: CelebA_HQ
# This script runs experiments to analyze performance degradation with measurement noise

SIGMA_Y=0.05
DATASET="CelebA_HQ"

# Zeta parameters (use same as noiseless experiments or tune if needed)
ZETA_ILVR=0.8
ZETA_MCG=0.5
ZETA_DDNM=1.0

# DDNM uses 100 timesteps, ILVR and MCG use 1000
DDNM_TIMESTEPS=100
ILVR_MCG_TIMESTEPS=1000

pids=()
cleanup() {
    echo "Interrupt received, killing children..."
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

echo "Starting noisy experiments with sigma_y = $SIGMA_Y on $DATASET..."
echo "Tasks: SR×4 and Box Inpainting"
echo "Methods: ILVR, MCG, DDNM"
echo ""

# ==================== ILVR Experiments ====================
echo "Starting ILVR experiments..."

# ILVR - SR×4
CUDA_VISIBLE_DEVICES=0 python -u hw4_step1_main.py \
    --ps_type ILVR \
    --dataset "$DATASET" \
    --degradation SR \
    --scale_factor 4 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$ILVR_MCG_TIMESTEPS" \
    --zeta_ilvr "$ZETA_ILVR" \
    > /tmp/noisy_ilvr_sr4.log 2>&1 &
pids+=($!)

# ILVR - Box Inpainting
CUDA_VISIBLE_DEVICES=1 python -u hw4_step1_main.py \
    --ps_type ILVR \
    --dataset "$DATASET" \
    --degradation Inpainting \
    --mask_type box \
    --box_indices 64 64 128 128 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$ILVR_MCG_TIMESTEPS" \
    --zeta_ilvr "$ZETA_ILVR" \
    > /tmp/noisy_ilvr_box.log 2>&1 &
pids+=($!)

# ==================== MCG Experiments ====================
echo "Starting MCG experiments..."

# MCG - SR×4
CUDA_VISIBLE_DEVICES=2 python -u hw4_step1_main.py \
    --ps_type MCG \
    --dataset "$DATASET" \
    --degradation SR \
    --scale_factor 4 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$ILVR_MCG_TIMESTEPS" \
    --zeta_mcg "$ZETA_MCG" \
    > /tmp/noisy_mcg_sr4.log 2>&1 &
pids+=($!)

# MCG - Box Inpainting
CUDA_VISIBLE_DEVICES=3 python -u hw4_step1_main.py \
    --ps_type MCG \
    --dataset "$DATASET" \
    --degradation Inpainting \
    --mask_type box \
    --box_indices 64 64 128 128 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$ILVR_MCG_TIMESTEPS" \
    --zeta_mcg "$ZETA_MCG" \
    > /tmp/noisy_mcg_box.log 2>&1 &
pids+=($!)

# ==================== DDNM Experiments ====================
echo "Starting DDNM experiments..."

# DDNM - SR×4
CUDA_VISIBLE_DEVICES=4 python -u hw4_step1_main.py \
    --ps_type DDNM \
    --dataset "$DATASET" \
    --degradation SR \
    --scale_factor 4 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$DDNM_TIMESTEPS" \
    --zeta_ddnm "$ZETA_DDNM" \
    > /tmp/noisy_ddnm_sr4.log 2>&1 &
pids+=($!)

# DDNM - Box Inpainting
CUDA_VISIBLE_DEVICES=5 python -u hw4_step1_main.py \
    --ps_type DDNM \
    --dataset "$DATASET" \
    --degradation Inpainting \
    --mask_type box \
    --box_indices 64 64 128 128 \
    --sigma_y "$SIGMA_Y" \
    --desired_timesteps "$DDNM_TIMESTEPS" \
    --zeta_ddnm "$ZETA_DDNM" \
    > /tmp/noisy_ddnm_box.log 2>&1 &
pids+=($!)

# Wait for all experiments to complete
echo ""
echo "All experiments started. Waiting for completion..."
echo "Logs are being written to /tmp/noisy_*.log"
wait

echo ""
echo "=========================================="
echo "All noisy experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: results/step1_results/$DATASET/"
echo ""
echo "To view logs:"
echo "  cat /tmp/noisy_ilvr_sr4.log"
echo "  cat /tmp/noisy_ilvr_box.log"
echo "  cat /tmp/noisy_mcg_sr4.log"
echo "  cat /tmp/noisy_mcg_box.log"
echo "  cat /tmp/noisy_ddnm_sr4.log"
echo "  cat /tmp/noisy_ddnm_box.log"
