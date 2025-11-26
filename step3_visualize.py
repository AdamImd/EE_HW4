"""
Generate visualizations and report from Step 3 results.
Run this after hw4_step3_main.py has completed.
"""

import os
import json
import argparse
from PIL import Image

from step3_utils.visualization import (
    create_comparison_grid,
    create_topic_comparison,
    plot_cfg_analysis,
    plot_version_comparison,
    create_cfg_image_grid
)
from step3_utils.analysis import generate_report


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for Step 3")
    parser.add_argument("--results_dir", type=str, default="results/step3_results",
                        help="Directory containing results")
    args = parser.parse_args()
    
    results_dir = args.results_dir
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load results
    part_b_results = None
    part_c_analysis = None
    part_d_results = None
    
    # Part B results
    part_b_path = os.path.join(results_dir, "part_b", "part_b_results.json")
    if os.path.exists(part_b_path):
        print("Loading Part B results...")
        with open(part_b_path, "r") as f:
            part_b_results = json.load(f)
        
        # Create topic comparison grid
        create_topic_comparison(
            part_b_results,
            os.path.join(results_dir, "part_b"),
            os.path.join(viz_dir, "part_b_topic_comparison.png")
        )
    
    # Part C analysis
    part_c_path = os.path.join(results_dir, "part_c", "part_c_analysis.json")
    if os.path.exists(part_c_path):
        print("Loading Part C analysis...")
        with open(part_c_path, "r") as f:
            part_c_analysis = json.load(f)
        
        # Create version comparison plot
        plot_version_comparison(
            part_c_analysis,
            os.path.join(viz_dir, "part_c_version_comparison.png")
        )
    
    # Part D results
    part_d_path = os.path.join(results_dir, "part_d", "part_d_results.json")
    if os.path.exists(part_d_path):
        print("Loading Part D results...")
        with open(part_d_path, "r") as f:
            part_d_data = json.load(f)
            part_d_results = part_d_data['results']
        
        # Create CFG analysis plot
        plot_cfg_analysis(
            part_d_results,
            os.path.join(viz_dir, "part_d_cfg_analysis.png")
        )
        
        # Create CFG image grid
        create_cfg_image_grid(
            part_d_results,
            os.path.join(viz_dir, "part_d_cfg_image_grid.png")
        )
    
    # Generate comprehensive report
    report = generate_report(
        part_b_results=part_b_results,
        part_c_analysis=part_c_analysis,
        part_d_results=part_d_results,
        save_path=os.path.join(results_dir, "experiment_report.txt")
    )
    
    print("\n" + report)
    print(f"\nVisualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
