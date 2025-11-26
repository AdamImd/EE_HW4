#!/bin/bash

# MCG experiments across 8 GPUs
# Each experiment runs on a different GPU in parallel

# CelebA_HQ experiments
# MCG variable
PS_TYPE="MCG"
ZETA_MCG=0.5

pids=()
cleanup() {
    echo "Interrupt received, killing children..."
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

CUDA_VISIBLE_DEVICES=0 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset CelebA_HQ --degradation SR --scale_factor 4 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu0.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=1 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset CelebA_HQ --degradation SR --scale_factor 8 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu1.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=2 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset CelebA_HQ --degradation Inpainting --mask_type random --random_amount 0.8 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu2.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=3 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset CelebA_HQ --degradation Inpainting --mask_type box --box_indices 64 64 128 128 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu3.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=4 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset ImageNet --degradation SR --scale_factor 4 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu4.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=5 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset ImageNet --degradation SR --scale_factor 8 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu5.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=6 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset ImageNet --degradation Inpainting --mask_type random --random_amount 0.8 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu6.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=7 python -u hw4_step1_main.py --ps_type "$PS_TYPE" --dataset ImageNet --degradation Inpainting --mask_type box --box_indices 64 64 128 128 --zeta_mcg "$ZETA_MCG" > /tmp/mcg_gpu7.log 2>&1 &
pids+=($!)

wait

echo "All MCG experiments completed!"
