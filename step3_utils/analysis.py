"""
Analysis utilities for Step 3 results.
"""

import os
import json
import numpy as np
from typing import List, Dict


def analyze_results(results: List[Dict]) -> Dict:
    """
    Perform comprehensive analysis on generation results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {}
    
    # Basic statistics
    clip_scores = [r['clip_score'] for r in results]
    analysis['clip_stats'] = {
        'mean': float(np.mean(clip_scores)),
        'std': float(np.std(clip_scores)),
        'min': float(np.min(clip_scores)),
        'max': float(np.max(clip_scores)),
        'median': float(np.median(clip_scores))
    }
    
    # Analysis by version if available
    if 'version' in results[0]:
        version_scores = {}
        for r in results:
            v = r['version']
            if v not in version_scores:
                version_scores[v] = {'clip': [], 'manual': []}
            version_scores[v]['clip'].append(r['clip_score'])
            if 'manual_score' in r:
                version_scores[v]['manual'].append(r['manual_score'])
        
        analysis['by_version'] = {}
        for v, scores in version_scores.items():
            analysis['by_version'][v] = {
                'avg_clip': float(np.mean(scores['clip'])),
                'std_clip': float(np.std(scores['clip'])),
                'avg_manual': float(np.mean(scores['manual'])) if scores['manual'] else None
            }
    
    # Analysis by topic if available
    if 'topic' in results[0]:
        topic_scores = {}
        for r in results:
            t = r['topic']
            if t not in topic_scores:
                topic_scores[t] = []
            topic_scores[t].append(r['clip_score'])
        
        analysis['by_topic'] = {
            t: {'avg_clip': float(np.mean(scores)), 'std_clip': float(np.std(scores))}
            for t, scores in topic_scores.items()
        }
    
    # Correlation analysis if manual scores available
    if 'manual_score' in results[0]:
        manual_scores = [r['manual_score'] for r in results]
        correlation = np.corrcoef(clip_scores, manual_scores)[0, 1]
        analysis['correlation'] = float(correlation)
    
    return analysis


def analyze_cfg_effects(results: List[Dict]) -> Dict:
    """
    Analyze the effects of different CFG scales.
    
    Args:
        results: List of result dictionaries from part d
        
    Returns:
        Dictionary containing CFG analysis
    """
    cfg_scales = [r['cfg_scale'] for r in results]
    clip_scores = [r['clip_score'] for r in results]
    
    analysis = {
        'cfg_scales': cfg_scales,
        'clip_scores': clip_scores,
        'best_cfg': cfg_scales[np.argmax(clip_scores)],
        'best_score': float(np.max(clip_scores)),
        'worst_cfg': cfg_scales[np.argmin(clip_scores)],
        'worst_score': float(np.min(clip_scores)),
        'score_range': float(np.max(clip_scores) - np.min(clip_scores))
    }
    
    # Trend analysis
    if len(cfg_scales) > 2:
        # Simple linear regression to detect trend
        slope = np.polyfit(cfg_scales, clip_scores, 1)[0]
        analysis['trend'] = 'increasing' if slope > 0.1 else ('decreasing' if slope < -0.1 else 'stable')
        analysis['slope'] = float(slope)
    
    return analysis


def generate_report(
    part_b_results: List[Dict] = None,
    part_c_analysis: Dict = None,
    part_d_results: List[Dict] = None,
    save_path: str = None
) -> str:
    """
    Generate a comprehensive text report of all results.
    
    Args:
        part_b_results: Results from part b
        part_c_analysis: Analysis from part c
        part_d_results: Results from part d
        save_path: Optional path to save the report
        
    Returns:
        Report as a string
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("STABLE DIFFUSION EXPERIMENT REPORT")
    report_lines.append("=" * 70)
    
    # Part B Summary
    if part_b_results:
        report_lines.append("\n" + "-" * 50)
        report_lines.append("PART B: 15 Prompts Generation Results")
        report_lines.append("-" * 50)
        
        # Organize by topic
        topics = {}
        for r in part_b_results:
            topic = r['topic']
            if topic not in topics:
                topics[topic] = {}
            topics[topic][r['version']] = r
        
        for topic, versions in topics.items():
            report_lines.append(f"\n{topic.replace('_', ' ').title()}:")
            for version in ['simple', 'medium', 'detailed']:
                if version in versions:
                    r = versions[version]
                    report_lines.append(f"  {version.capitalize():10} | CLIP: {r['clip_score']:5.1f}% | "
                                       f"Manual: {r.get('manual_score', 'N/A')}/10")
    
    # Part C Summary
    if part_c_analysis:
        report_lines.append("\n" + "-" * 50)
        report_lines.append("PART C: CLIP vs Manual Score Analysis")
        report_lines.append("-" * 50)
        
        if 'version_averages' in part_c_analysis:
            report_lines.append("\nAverage Scores by Prompt Complexity:")
            for version in ['simple', 'medium', 'detailed']:
                if version in part_c_analysis['version_averages']:
                    v = part_c_analysis['version_averages'][version]
                    manual_str = f"{v['avg_manual']:.1f}/10" if v.get('avg_manual') else "N/A"
                    report_lines.append(f"  {version.capitalize():10} | CLIP: {v['avg_clip']:5.1f}% | "
                                       f"Manual: {manual_str}")
        
        if 'correlation' in part_c_analysis:
            report_lines.append(f"\nCorrelation (CLIP vs Manual): {part_c_analysis['correlation']:.3f}")
    
    # Part D Summary
    if part_d_results:
        report_lines.append("\n" + "-" * 50)
        report_lines.append("PART D: CFG Scale Analysis")
        report_lines.append("-" * 50)
        
        report_lines.append("\nCFG Scale vs CLIP Score:")
        for r in part_d_results:
            report_lines.append(f"  CFG {r['cfg_scale']:5.1f} | CLIP: {r['clip_score']:5.1f}%")
        
        cfg_analysis = analyze_cfg_effects(part_d_results)
        report_lines.append(f"\nBest CFG: {cfg_analysis['best_cfg']} (Score: {cfg_analysis['best_score']:.1f}%)")
        report_lines.append(f"Worst CFG: {cfg_analysis['worst_cfg']} (Score: {cfg_analysis['worst_score']:.1f}%)")
        report_lines.append(f"Score Range: {cfg_analysis['score_range']:.1f}%")
        if 'trend' in cfg_analysis:
            report_lines.append(f"Overall Trend: {cfg_analysis['trend']}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {save_path}")
    
    return report


def compute_prompt_statistics(prompts: Dict) -> Dict:
    """
    Compute statistics about prompt lengths.
    
    Args:
        prompts: Dictionary of prompts organized by topic
        
    Returns:
        Dictionary with prompt statistics
    """
    stats = {'simple': [], 'medium': [], 'detailed': []}
    
    for topic, versions in prompts.items():
        for version, prompt in versions.items():
            word_count = len(prompt.split())
            char_count = len(prompt)
            stats[version].append({
                'topic': topic,
                'word_count': word_count,
                'char_count': char_count
            })
    
    summary = {}
    for version, data in stats.items():
        word_counts = [d['word_count'] for d in data]
        char_counts = [d['char_count'] for d in data]
        summary[version] = {
            'avg_words': float(np.mean(word_counts)),
            'avg_chars': float(np.mean(char_counts)),
            'min_words': int(np.min(word_counts)),
            'max_words': int(np.max(word_counts))
        }
    
    return summary
