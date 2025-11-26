#!/bin/bash
# Run Stable Diffusion experiments for HW4 Step 3

# Exit on error
set -e

echo "========================================"
echo "HW4 Step 3: Stable Diffusion Experiments"
echo "========================================"

# Check if required packages are installed
python -c "import diffusers; import transformers" 2>/dev/null || {
    echo "Installing required packages..."
    pip install -r requirements_step3.txt
}

# Default settings
MODEL="runwayml/stable-diffusion-v1-5"
CLIP_MODEL="openai/clip-vit-base-patch32"
SEED=42
RESULTS_DIR="results/step3_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --parts)
            PARTS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

PARTS="${PARTS:-abcd}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  CLIP Model: $CLIP_MODEL"
echo "  Seed: $SEED"
echo "  Parts: $PARTS"
echo "  Results: $RESULTS_DIR"
echo ""

# Run main experiments
python hw4_step3_main.py \
    --model "$MODEL" \
    --clip_model "$CLIP_MODEL" \
    --seed "$SEED" \
    --parts "$PARTS" \
    --results_dir "$RESULTS_DIR"

# Generate visualizations
echo ""
echo "Generating visualizations..."
python step3_visualize.py --results_dir "$RESULTS_DIR"

echo ""
echo "========================================"
echo "Experiments completed!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
