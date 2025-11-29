#!/bin/bash

# Script to collect results for Q2: Conditioning Diffusion
# Tests both Classifier Guidance (CG) and Classifier-Free Guidance (CFG)
# with guidance scales: 0.0, 1.0, 3.0, 5.0, 10.0

echo "=========================================="
echo "Running Step 2 Experiments"
echo "Classifier Guidance (CG) and Classifier-Free Guidance (CFG)"
echo "=========================================="
echo ""

# Create results directory if it doesn't exist
mkdir -p results/step2_results

# Define guidance scales to test
SCALES=(0.0 1.0 3.0 5.0 10.0)

# Number of sampling steps
SAMPLE_STEPS=100

# Run experiments for each guidance scale
for scale in "${SCALES[@]}"; do
    echo "----------------------------------------"
    echo "Running with guidance scale: $scale"
    echo "----------------------------------------"
    
    # Run the experiment
    python hw4_step2_main.py \
        --cg_scale $scale \
        --cfg_scale $scale \
        --sample_steps $SAMPLE_STEPS
    
    echo ""
    echo "Completed scale $scale"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in results/step2_results/"
echo "=========================================="
