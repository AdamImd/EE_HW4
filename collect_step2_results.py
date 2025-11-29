#!/usr/bin/env python3
"""
Script to systematically collect results for Q2: Conditioning Diffusion
Runs both CG and CFG sampling with multiple guidance scales and collects metrics.
"""

import subprocess
import json
import os
from pathlib import Path

def run_experiment(cg_scale, cfg_scale, sample_steps=100):
    """Run a single experiment with specified parameters."""
    cmd = [
        "python", "hw4_step2_main.py",
        "--cg_scale", str(cg_scale),
        "--cfg_scale", str(cfg_scale),
        "--sample_steps", str(sample_steps)
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Parse metrics from output
    metrics = {}
    for line in result.stdout.split('\n'):
        if "Classifier Acc on CG" in line:
            parts = line.split('|')
            acc_part = parts[0].split(':')[1].strip()
            div_part = parts[1].split(':')[1].strip()
            metrics['cg_accuracy'] = float(acc_part)
            metrics['cg_diversity'] = float(div_part)
        elif "Classifier Acc on CFG" in line:
            parts = line.split('|')
            acc_part = parts[0].split(':')[1].strip()
            div_part = parts[1].split(':')[1].strip()
            metrics['cfg_accuracy'] = float(acc_part)
            metrics['cfg_diversity'] = float(div_part)
    
    return metrics

def main():
    # Create results directory
    results_dir = Path("results/step2_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define guidance scales to test (from Q2.tex)
    guidance_scales = [0.0, 1.0, 3.0, 5.0, 10.0]
    sample_steps = 100
    
    print("=" * 60)
    print("Step 2: Conditioning Diffusion - Results Collection")
    print("=" * 60)
    print(f"Guidance scales: {guidance_scales}")
    print(f"Sample steps: {sample_steps}")
    print("=" * 60)
    
    # Collect all results
    all_results = []
    
    for scale in guidance_scales:
        print(f"\n{'=' * 60}")
        print(f"Testing guidance scale: {scale}")
        print(f"{'=' * 60}")
        
        metrics = run_experiment(
            cg_scale=scale,
            cfg_scale=scale,
            sample_steps=sample_steps
        )
        
        result_entry = {
            'guidance_scale': scale,
            'sample_steps': sample_steps,
            'cg_accuracy': metrics.get('cg_accuracy', None),
            'cg_diversity': metrics.get('cg_diversity', None),
            'cfg_accuracy': metrics.get('cfg_accuracy', None),
            'cfg_diversity': metrics.get('cfg_diversity', None),
            'cg_image_path': f"results/step2_results/mnist_cg_{scale}.png",
            'cfg_image_path': f"results/step2_results/mnist_cfg_{scale}.png"
        }
        
        all_results.append(result_entry)
        
        print(f"\nMetrics for scale {scale}:")
        print(f"  CG  - Accuracy: {result_entry['cg_accuracy']:.3f}, Diversity: {result_entry['cg_diversity']:.3f}")
        print(f"  CFG - Accuracy: {result_entry['cfg_accuracy']:.3f}, Diversity: {result_entry['cfg_diversity']:.3f}")
    
    # Save results to JSON
    results_file = results_dir / "step2_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    
    print(f"\n{'=' * 60}")
    print("All experiments completed!")
    print(f"Results saved to: {results_file}")
    print(f"Images saved to: {results_dir}/")
    print("=" * 60)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Scale':<8} {'CG Acc':<10} {'CG Div':<10} {'CFG Acc':<10} {'CFG Div':<10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['guidance_scale']:<8.1f} "
              f"{r['cg_accuracy']:<10.3f} "
              f"{r['cg_diversity']:<10.3f} "
              f"{r['cfg_accuracy']:<10.3f} "
              f"{r['cfg_diversity']:<10.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
