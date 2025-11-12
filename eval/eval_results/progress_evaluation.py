#!/usr/bin/env python3
"""
Progress Estimation Evaluation Module

This module provides comprehensive evaluation functions for progress estimation tasks,
including support for N/A values, false positive detection, and trajectory order consistency (VOC).

Author: Generated from ProgressLM evaluation pipeline
Date: 2025-01
"""

import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Union


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def calculate_false_positives(
    predicted_ref: Union[int, str, None],
    predicted_score: Union[float, str, None],
    gt_ref: Optional[int],
    gt_score: Optional[float]
) -> Tuple[bool, bool]:
    """
    Calculate false positive rates for ref and score predictions.

    False positive occurs when:
    - GT is numeric but prediction is "n/a"
    - GT is "n/a" (None) but prediction is numeric

    Correct cases:
    - Both GT and prediction are "n/a"
    - Both GT and prediction are numeric (use error calculation instead)

    Args:
        predicted_ref: Predicted reference (int, "n/a", or None)
        predicted_score: Predicted score (float, "n/a", or None)
        gt_ref: Ground truth reference (int or None for n/a)
        gt_score: Ground truth score (float or None for n/a)

    Returns:
        Tuple of (is_ref_false_positive, is_score_false_positive)
    """
    # Check ref false positive
    gt_ref_is_na = (gt_ref is None)
    pred_ref_is_na = (
        predicted_ref == "n/a" or
        predicted_ref == "" or
        predicted_ref is None or
        not isinstance(predicted_ref, int)
    )
    ref_fp = gt_ref_is_na != pred_ref_is_na

    # Check score false positive
    gt_score_is_na = (gt_score is None)
    pred_score_is_na = (
        predicted_score == "n/a" or
        predicted_score is None or
        not isinstance(predicted_score, (int, float))
    )
    score_fp = gt_score_is_na != pred_score_is_na

    return ref_fp, score_fp


def calculate_evaluation_score(
    predicted: Optional[float],
    ground_truth: Optional[float]
) -> float:
    """
    Calculate relative error: |ground_truth - predicted| / ground_truth

    Uses pure relative error metric. Lower is better (0.0 = perfect prediction).
    Only calculates when both values are numeric.

    Args:
        predicted: Predicted progress score (0-1) or None
        ground_truth: Ground truth progress score (0-1) or None

    Returns:
        Relative error (0.0 = perfect, higher = worse), or inf if either is None or GT is 0
    """
    if predicted is None or ground_truth is None:
        return float('inf')

    if not isinstance(predicted, (int, float)) or not isinstance(ground_truth, (int, float)):
        return float('inf')

    # Avoid division by zero
    if ground_truth == 0.0:
        return 0.0 if predicted == 0.0 else float('inf')

    relative_error = abs(ground_truth - predicted) / ground_truth
    return relative_error


def calculate_ref_error(
    predicted_ref: Optional[int],
    ground_truth_ref: Optional[int]
) -> float:
    """
    Calculate reference index absolute error: |ground_truth_ref - predicted_ref|

    Only calculates when both values are numeric integers.

    Args:
        predicted_ref: Predicted reference index (1-based) or None
        ground_truth_ref: Ground truth reference index (1-based) or None

    Returns:
        Absolute error (0.0 = perfect, higher = worse), or inf if either is None
    """
    if predicted_ref is None or ground_truth_ref is None:
        return float('inf')

    if not isinstance(predicted_ref, int) or not isinstance(ground_truth_ref, int):
        return float('inf')

    absolute_error = abs(ground_truth_ref - predicted_ref)
    return float(absolute_error)


def calculate_voc_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate VOC (Visual/Trajectory Order Consistency) using Spearman correlation.

    Process:
    1. Group samples by trajectory ID
    2. Filter: only keep trajectories where GT closest_idx and progress_score are both numeric
    3. For each trajectory:
       - Sort by GT progress_score to get true order
       - Sort by predicted score (n/a → 0.0) to get predicted order
       - Calculate Spearman correlation between rankings
    4. Return mean and std of all valid VOCs

    Args:
        results: List of result dictionaries with meta_data containing:
                 - id: trajectory identifier
                 - closest_idx: GT reference (int or None)
                 - progress_score: GT score (float or None)

    Returns:
        Dictionary with VOC statistics:
        {
            'voc_mean': float or None,
            'voc_std': float or None,
            'voc_median': float or None,
            'voc_count': int,  # number of trajectories with VOC
            'voc_values': List[float]  # individual VOC values
        }
    """
    # Group by trajectory ID
    trajectories = defaultdict(list)

    for res in results:
        meta = res.get('meta_data', {})
        traj_id = meta.get('id', '')

        # Only include if GT has numeric values
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        if gt_ref is not None and gt_score is not None:
            # GT is numeric
            pred_score = res.get('score')

            # Convert n/a to 0.0 for ranking
            if pred_score == "n/a" or pred_score is None:
                pred_score_numeric = 0.0
            else:
                # Parse percentage string if needed
                if isinstance(pred_score, str):
                    pred_score = pred_score.strip().replace('%', '')
                    try:
                        pred_score_numeric = float(pred_score) / 100.0 if float(pred_score) > 1.0 else float(pred_score)
                    except:
                        pred_score_numeric = 0.0
                else:
                    pred_score_numeric = float(pred_score) if isinstance(pred_score, (int, float)) else 0.0

            trajectories[traj_id].append({
                'gt_score': gt_score,
                'pred_score': pred_score_numeric,
                'result': res
            })

    # Calculate VOC for each trajectory
    voc_values = []

    for traj_id, samples in trajectories.items():
        if len(samples) <= 1:
            # Cannot calculate correlation for single sample
            continue

        # Sort by GT score to get true ranking
        samples_sorted_by_gt = sorted(samples, key=lambda x: x['gt_score'])
        true_order = list(range(len(samples_sorted_by_gt)))

        # Sort by predicted score to get predicted ranking
        samples_sorted_by_pred = sorted(samples, key=lambda x: x['pred_score'])

        # Map each sample to its predicted rank
        pred_rank_map = {id(s['result']): rank for rank, s in enumerate(samples_sorted_by_pred)}
        pred_order = [pred_rank_map[id(s['result'])] for s in samples_sorted_by_gt]

        # Calculate Spearman correlation
        if len(set(true_order)) > 1 and len(set(pred_order)) > 1:
            try:
                correlation, _ = spearmanr(true_order, pred_order)
                if not np.isnan(correlation):
                    voc_values.append(correlation)
            except:
                continue

    # Calculate statistics
    if len(voc_values) > 0:
        return {
            'voc_mean': float(np.mean(voc_values)),
            'voc_std': float(np.std(voc_values)),
            'voc_median': float(np.median(voc_values)),
            'voc_count': len(voc_values),
            'voc_values': voc_values
        }
    else:
        return {
            'voc_mean': None,
            'voc_std': None,
            'voc_median': None,
            'voc_count': 0,
            'voc_values': []
        }


# ============================================================================
# Data Loading and Filtering
# ============================================================================

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load results from a JSONL file.

    Args:
        file_path: Path to the JSONL results file

    Returns:
        List of result dictionaries
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    return results


def filter_valid_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter results into valid and error samples.

    Args:
        results: List of result dictionaries

    Returns:
        Tuple of (valid_results, error_results)
    """
    valid_results = []
    error_results = []

    for result in results:
        status = result.get('meta_data', {}).get('status', 'unknown')
        if status == 'success':
            valid_results.append(result)
        else:
            error_results.append(result)

    return valid_results, error_results


def calculate_gt_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate ground truth distribution (numeric vs n/a).

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with counts:
        {
            'gt_numeric_count': int,  # Both ref and score are numeric
            'gt_na_count': int,       # Either ref or score is n/a
            'gt_ref_numeric': int,
            'gt_ref_na': int,
            'gt_score_numeric': int,
            'gt_score_na': int
        }
    """
    gt_numeric_count = 0
    gt_na_count = 0
    gt_ref_numeric = 0
    gt_ref_na = 0
    gt_score_numeric = 0
    gt_score_na = 0

    for result in results:
        meta = result.get('meta_data', {})
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        # Count ref
        if gt_ref is not None:
            gt_ref_numeric += 1
        else:
            gt_ref_na += 1

        # Count score
        if gt_score is not None:
            gt_score_numeric += 1
        else:
            gt_score_na += 1

        # Count combined (both numeric)
        if gt_ref is not None and gt_score is not None:
            gt_numeric_count += 1
        else:
            gt_na_count += 1

    return {
        'gt_numeric_count': gt_numeric_count,
        'gt_na_count': gt_na_count,
        'gt_ref_numeric': gt_ref_numeric,
        'gt_ref_na': gt_ref_na,
        'gt_score_numeric': gt_score_numeric,
        'gt_score_na': gt_score_na
    }


# ============================================================================
# Main Analysis Functions
# ============================================================================

def analyze_results(
    results: List[Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze evaluation results comprehensively.

    Args:
        results: List of result dictionaries from JSONL file
        verbose: Whether to print detailed statistics

    Returns:
        Dictionary containing all evaluation metrics
    """
    if not results:
        return {
            'error': 'No results to analyze',
            'total_samples': 0
        }

    # Filter valid and error samples
    valid_results, error_results = filter_valid_results(results)

    # Initialize counters
    total_samples = len(results)
    error_count = len(error_results)
    valid_count = len(valid_results)

    # Calculate errors and false positives
    score_errors = []
    ref_errors = []
    ref_fp_count = 0
    score_fp_count = 0

    for result in results:
        meta = result.get('meta_data', {})
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        # Parse predicted values
        pred_ref_str = result.get('ref')
        pred_score_str = result.get('score')

        # Convert predicted ref
        if pred_ref_str == "n/a":
            pred_ref = "n/a"
        elif pred_ref_str is not None and isinstance(pred_ref_str, str) and pred_ref_str.isdigit():
            pred_ref = int(pred_ref_str)
        elif isinstance(pred_ref_str, int):
            pred_ref = pred_ref_str
        else:
            pred_ref = None

        # Convert predicted score
        if pred_score_str == "n/a":
            pred_score = "n/a"
        elif pred_score_str is not None:
            try:
                if isinstance(pred_score_str, str):
                    pred_score_str_clean = pred_score_str.strip().replace('%', '')
                    pred_score_val = float(pred_score_str_clean)
                    pred_score = pred_score_val / 100.0 if pred_score_val > 1.0 else pred_score_val
                else:
                    pred_score = float(pred_score_str)
            except:
                pred_score = None
        else:
            pred_score = None

        # Calculate false positives
        ref_fp, score_fp = calculate_false_positives(pred_ref, pred_score, gt_ref, gt_score)
        if ref_fp:
            ref_fp_count += 1
        if score_fp:
            score_fp_count += 1

        # Calculate errors (only for numeric pairs)
        if gt_score is not None and isinstance(pred_score, (int, float)):
            score_error = calculate_evaluation_score(pred_score, gt_score)
            if score_error != float('inf'):
                score_errors.append(score_error)

        if gt_ref is not None and isinstance(pred_ref, int):
            ref_error = calculate_ref_error(pred_ref, gt_ref)
            if ref_error != float('inf'):
                ref_errors.append(ref_error)

    # Calculate VOC metrics
    voc_metrics = calculate_voc_metrics(results)

    # Calculate GT distribution
    gt_dist = calculate_gt_distribution(results)

    # Compile statistics
    stats = {
        'total_samples': total_samples,
        'valid_samples': valid_count,
        'error_samples': error_count,
        'error_rate': error_count / total_samples if total_samples > 0 else 0.0,

        # Score error statistics
        'score_error_mean': float(np.mean(score_errors)) if score_errors else None,
        'score_error_median': float(np.median(score_errors)) if score_errors else None,
        'score_error_std': float(np.std(score_errors)) if score_errors else None,
        'score_error_count': len(score_errors),

        # Ref error statistics
        'ref_error_mean': float(np.mean(ref_errors)) if ref_errors else None,
        'ref_error_median': float(np.median(ref_errors)) if ref_errors else None,
        'ref_error_std': float(np.std(ref_errors)) if ref_errors else None,
        'ref_error_count': len(ref_errors),

        # False positive statistics
        'ref_fp_count': ref_fp_count,
        'ref_fp_rate': ref_fp_count / total_samples if total_samples > 0 else 0.0,
        'score_fp_count': score_fp_count,
        'score_fp_rate': score_fp_count / total_samples if total_samples > 0 else 0.0,

        # VOC statistics
        'voc_mean': voc_metrics['voc_mean'],
        'voc_std': voc_metrics['voc_std'],
        'voc_median': voc_metrics['voc_median'],
        'voc_count': voc_metrics['voc_count'],

        # GT distribution
        **gt_dist
    }

    # Print detailed report if verbose
    if verbose:
        print("=" * 80)
        print("PROGRESS ESTIMATION EVALUATION REPORT")
        print("=" * 80)

        print(f"\n📊 Basic Statistics:")
        print(f"  Total samples:     {stats['total_samples']}")
        print(f"  Valid samples:     {stats['valid_samples']}")
        print(f"  Error samples:     {stats['error_samples']} ({stats['error_rate']*100:.2f}%)")

        print(f"\n📈 Score Error Metrics:")
        if stats['score_error_mean'] is not None:
            print(f"  Mean error:        {stats['score_error_mean']:.4f}")
            print(f"  Median error:      {stats['score_error_median']:.4f}")
            print(f"  Std error:         {stats['score_error_std']:.4f}")
            print(f"  Valid samples:     {stats['score_error_count']}/{stats['total_samples']}")
        else:
            print(f"  No valid error calculations")

        print(f"\n📍 Ref Error Metrics:")
        if stats['ref_error_mean'] is not None:
            print(f"  Mean error:        {stats['ref_error_mean']:.4f}")
            print(f"  Median error:      {stats['ref_error_median']:.4f}")
            print(f"  Std error:         {stats['ref_error_std']:.4f}")
            print(f"  Valid samples:     {stats['ref_error_count']}/{stats['total_samples']}")
        else:
            print(f"  No valid error calculations")

        print(f"\n⚠️  False Positive Rates:")
        print(f"  Ref FP rate:       {stats['ref_fp_rate']*100:.2f}% ({stats['ref_fp_count']}/{stats['total_samples']})")
        print(f"  Score FP rate:     {stats['score_fp_rate']*100:.2f}% ({stats['score_fp_count']}/{stats['total_samples']})")

        print(f"\n🔄 VOC (Trajectory Order Consistency):")
        if stats['voc_mean'] is not None:
            print(f"  Mean VOC:          {stats['voc_mean']:.4f}")
            print(f"  Median VOC:        {stats['voc_median']:.4f}")
            print(f"  Std VOC:           {stats['voc_std']:.4f}")
            print(f"  Valid trajectories: {stats['voc_count']}")
        else:
            print(f"  VOC: N/A (insufficient data)")

        print(f"\n📋 Ground Truth Distribution:")
        print(f"  Both numeric:      {stats['gt_numeric_count']} ({stats['gt_numeric_count']/stats['total_samples']*100:.1f}%)")
        print(f"  Contains N/A:      {stats['gt_na_count']} ({stats['gt_na_count']/stats['total_samples']*100:.1f}%)")
        print(f"    - Ref numeric:   {stats['gt_ref_numeric']}")
        print(f"    - Ref N/A:       {stats['gt_ref_na']}")
        print(f"    - Score numeric: {stats['gt_score_numeric']}")
        print(f"    - Score N/A:     {stats['gt_score_na']}")

        print("=" * 80)

    return stats


def generate_summary_report(file_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report from a JSONL results file.

    Args:
        file_path: Path to the JSONL results file
        output_file: Optional path to save summary JSON (default: same as input with _summary.json)

    Returns:
        Dictionary containing all evaluation metrics
    """
    print(f"\n📂 Loading results from: {file_path}")
    results = load_results(file_path)

    if not results:
        print("❌ No results found in file")
        return {'error': 'No results found'}

    print(f"✓ Loaded {len(results)} samples\n")

    # Analyze results
    stats = analyze_results(results, verbose=True)

    # Save summary if output file specified
    if output_file is None:
        output_file = file_path.replace('.jsonl', '_evaluation_summary.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary saved to: {output_file}\n")

    return stats


def compare_models(
    result_files: Dict[str, str],
    output_file: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare evaluation metrics across multiple models.

    Args:
        result_files: Dictionary mapping model names to their result file paths
                     Example: {'model_A': 'path/to/results_A.jsonl', 'model_B': 'path/to/results_B.jsonl'}
        output_file: Optional path to save comparison JSON

    Returns:
        Dictionary mapping model names to their evaluation statistics
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    all_stats = {}

    for model_name, file_path in result_files.items():
        print(f"\n🔍 Evaluating: {model_name}")
        print(f"   File: {file_path}")

        results = load_results(file_path)
        if not results:
            print(f"   ❌ No results found")
            continue

        stats = analyze_results(results, verbose=False)
        all_stats[model_name] = stats

        # Print brief summary
        print(f"   ✓ Samples: {stats['total_samples']}")
        print(f"   ✓ Score Error: {stats['score_error_mean']:.4f}" if stats['score_error_mean'] else "   ✓ Score Error: N/A")
        print(f"   ✓ VOC: {stats['voc_mean']:.4f}" if stats['voc_mean'] else "   ✓ VOC: N/A")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    metrics_to_compare = [
        ('Score Error (Mean)', 'score_error_mean'),
        ('Ref Error (Mean)', 'ref_error_mean'),
        ('Score FP Rate', 'score_fp_rate'),
        ('Ref FP Rate', 'ref_fp_rate'),
        ('VOC (Mean)', 'voc_mean'),
        ('Error Rate', 'error_rate')
    ]

    print(f"\n{'Metric':<25} " + " ".join([f"{name:>15}" for name in result_files.keys()]))
    print("-" * (25 + 16 * len(result_files)))

    for metric_name, metric_key in metrics_to_compare:
        values = []
        for model_name in result_files.keys():
            val = all_stats.get(model_name, {}).get(metric_key)
            if val is not None:
                if metric_key.endswith('_rate'):
                    values.append(f"{val*100:.2f}%")
                else:
                    values.append(f"{val:.4f}")
            else:
                values.append("N/A")
        print(f"{metric_name:<25} " + " ".join([f"{v:>15}" for v in values]))

    # Save comparison if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Comparison saved to: {output_file}")

    print("=" * 80 + "\n")

    return all_stats


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Progress Estimation Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single file
  python progress_evaluation.py results.jsonl

  # Evaluate and save summary
  python progress_evaluation.py results.jsonl --output summary.json

  # Compare multiple models
  python progress_evaluation.py --compare modelA:results_A.jsonl modelB:results_B.jsonl --output comparison.json
        """
    )

    parser.add_argument(
        'input_file',
        nargs='?',
        help='Path to JSONL results file'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path for summary JSON'
    )

    parser.add_argument(
        '--compare', '-c',
        nargs='+',
        metavar='NAME:FILE',
        help='Compare multiple models (format: name:filepath)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        result_files = {}
        for item in args.compare:
            try:
                name, filepath = item.split(':', 1)
                result_files[name] = filepath
            except ValueError:
                print(f"Error: Invalid format for --compare: {item}")
                print("Expected format: name:filepath")
                return 1

        compare_models(result_files, output_file=args.output)
        return 0

    # Single file evaluation mode
    if not args.input_file:
        parser.print_help()
        return 1

    generate_summary_report(args.input_file, output_file=args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
