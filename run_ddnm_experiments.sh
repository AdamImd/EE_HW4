#!/bin/bash

# DDNM experiments across 8 GPUs
# Each experiment runs on a different GPU in parallel
# Note: DDNM uses 100 sampling steps (not 1000) as per the homework requirement

# CelebA_HQ experiments
# ZETA_DDNM=1.0
ZETA_DDNM=1

CUDA_VISIBLE_DEVICES=0 python hw4_step1_main.py --ps_type DDNM --dataset CelebA_HQ --degradation SR --scale_factor 4 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &
CUDA_VISIBLE_DEVICES=1 python hw4_step1_main.py --ps_type DDNM --dataset CelebA_HQ --degradation SR --scale_factor 8 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &
CUDA_VISIBLE_DEVICES=2 python hw4_step1_main.py --ps_type DDNM --dataset CelebA_HQ --degradation Inpainting --mask_type random --random_amount 0.8 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &
CUDA_VISIBLE_DEVICES=3 python hw4_step1_main.py --ps_type DDNM --dataset CelebA_HQ --degradation Inpainting --mask_type box --box_indices 64 64 128 128 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &

# ImageNet experiments
CUDA_VISIBLE_DEVICES=4 python hw4_step1_main.py --ps_type DDNM --dataset ImageNet --degradation SR --scale_factor 4 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &
CUDA_VISIBLE_DEVICES=5 python hw4_step1_main.py --ps_type DDNM --dataset ImageNet --degradation SR --scale_factor 8 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &
CUDA_VISIBLE_DEVICES=6 python hw4_step1_main.py --ps_type DDNM --dataset ImageNet --degradation Inpainting --mask_type random --random_amount 0.8 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &
CUDA_VISIBLE_DEVICES=7 python hw4_step1_main.py --ps_type DDNM --dataset ImageNet --degradation Inpainting --mask_type box --box_indices 64 64 128 128 --desired_timesteps 100 --zeta_ddnm "$ZETA_DDNM" &

# Wait for all background jobs to complete
wait

echo "All DDNM experiments completed!"
