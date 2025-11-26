"""
Visualization utilities for Step 3 results.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def create_comparison_grid(
    images: List[Image.Image],
    titles: List[str],
    save_path: str,
    ncols: int = 3,
    figsize: Tuple[int, int] = None
):
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of PIL Images
        titles: List of titles for each image
        save_path: Path to save the figure
        ncols: Number of columns in the grid
        figsize: Figure size (width, height)
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to: {save_path}")


def create_topic_comparison(
    results: List[Dict],
    images_dir: str,
    save_path: str
):
    """
    Create a comparison grid organized by topic and prompt version.
    
    Args:
        results: List of result dictionaries from part b
        images_dir: Directory containing the generated images
        save_path: Path to save the figure
    """
    # Organize by topic
    topics = {}
    for r in results:
        topic = r['topic']
        if topic not in topics:
            topics[topic] = {}
        topics[topic][r['version']] = r
    
    n_topics = len(topics)
    fig, axes = plt.subplots(n_topics, 3, figsize=(15, n_topics * 5))
    
    version_order = ['simple', 'medium', 'detailed']
    
    for row, (topic, versions) in enumerate(topics.items()):
        for col, version in enumerate(version_order):
            if version in versions:
                r = versions[version]
                img = Image.open(r['image_path'])
                axes[row, col].imshow(img)
                title = f"{topic.replace('_', ' ').title()}\n{version.capitalize()}"
                title += f"\nCLIP: {r['clip_score']:.1f}%"
                axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Topic comparison saved to: {save_path}")


def plot_cfg_analysis(
    results: List[Dict],
    save_path: str
):
    """
    Plot CFG scale vs CLIP score analysis.
    
    Args:
        results: List of result dictionaries from part d
        save_path: Path to save the figure
    """
    cfg_scales = [r['cfg_scale'] for r in results]
    clip_scores = [r['clip_score'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Line plot
    ax1.plot(cfg_scales, clip_scores, 'bo-', linewidth=2, markersize=10)
    ax1.set_xlabel('CFG Scale (ω_CFG)', fontsize=12)
    ax1.set_ylabel('CLIP Similarity Score (%)', fontsize=12)
    ax1.set_title('CLIP Score vs CFG Scale', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cfg_scales)
    
    # Bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(cfg_scales)))
    bars = ax2.bar(range(len(cfg_scales)), clip_scores, color=colors)
    ax2.set_xlabel('CFG Scale', fontsize=12)
    ax2.set_ylabel('CLIP Similarity Score (%)', fontsize=12)
    ax2.set_title('CLIP Score by CFG Scale', fontsize=14)
    ax2.set_xticks(range(len(cfg_scales)))
    ax2.set_xticklabels([str(c) for c in cfg_scales])
    
    # Add value labels on bars
    for bar, score in zip(bars, clip_scores):
        height = bar.get_height()
        ax2.annotate(f'{score:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CFG analysis plot saved to: {save_path}")


def plot_version_comparison(
    analysis: Dict,
    save_path: str
):
    """
    Plot comparison of CLIP and manual scores by prompt version.
    
    Args:
        analysis: Analysis dictionary from part c
        save_path: Path to save the figure
    """
    versions = ['simple', 'medium', 'detailed']
    clip_avgs = [analysis['version_averages'][v]['avg_clip'] for v in versions]
    manual_avgs = [analysis['version_averages'][v]['avg_manual'] * 10 for v in versions]  # Scale to %
    
    x = np.arange(len(versions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, clip_avgs, width, label='CLIP Score', color='steelblue')
    bars2 = ax.bar(x + width/2, manual_avgs, width, label='Manual Score (×10)', color='coral')
    
    ax.set_xlabel('Prompt Version', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Average Scores by Prompt Complexity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([v.capitalize() for v in versions])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Version comparison plot saved to: {save_path}")


def create_cfg_image_grid(
    results: List[Dict],
    save_path: str
):
    """
    Create a grid showing images generated at different CFG scales.
    
    Args:
        results: List of result dictionaries from part d
        save_path: Path to save the figure
    """
    n_images = len(results)
    ncols = min(3, n_images)
    nrows = (n_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    
    if nrows == 1:
        axes = axes if ncols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for idx, r in enumerate(results):
        img = Image.open(r['image_path'])
        axes[idx].imshow(img)
        title = f"CFG = {r['cfg_scale']}\nCLIP: {r['clip_score']:.1f}%"
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Effect of CFG Scale on Image Generation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CFG image grid saved to: {save_path}")
