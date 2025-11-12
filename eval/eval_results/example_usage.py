#!/usr/bin/env python3
"""
Example usage of the progress_evaluation module.

This script demonstrates various ways to use the evaluation functions.
"""

import json
from progress_evaluation import (
    load_results,
    analyze_results,
    generate_summary_report,
    compare_models,
    calculate_false_positives,
    calculate_evaluation_score,
    calculate_voc_metrics
)


def example_1_single_file_evaluation():
    """Example 1: Evaluate a single results file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single File Evaluation")
    print("=" * 80 + "\n")

    # Method 1: Simple one-liner
    print("Method 1: Using generate_summary_report (with verbose output)")
    stats = generate_summary_report(
        'path/to/results.jsonl',
        output_file='summary.json'
    )

    # Method 2: Load and analyze separately
    print("\nMethod 2: Load and analyze separately")
    results = load_results('path/to/results.jsonl')
    stats = analyze_results(results, verbose=True)

    # Access specific metrics
    print(f"\nKey Metrics:")
    print(f"  Score Error Mean: {stats['score_error_mean']:.4f}")
    print(f"  VOC Mean: {stats['voc_mean']:.4f}")
    print(f"  Score FP Rate: {stats['score_fp_rate']*100:.2f}%")


def example_2_model_comparison():
    """Example 2: Compare multiple models."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Model Comparison")
    print("=" * 80 + "\n")

    # Define models to compare
    models = {
        'Baseline': 'results/baseline.jsonl',
        'SFT-3B': 'results/sft_3b.jsonl',
        'SFT-7B': 'results/sft_7b.jsonl',
        'DPO-7B': 'results/dpo_7b.jsonl'
    }

    # Compare all models
    comparison = compare_models(models, output_file='model_comparison.json')

    # Find best model by VOC
    best_voc_model = max(
        comparison.items(),
        key=lambda x: x[1]['voc_mean'] if x[1]['voc_mean'] else -1
    )
    print(f"\n🏆 Best model by VOC: {best_voc_model[0]} (VOC: {best_voc_model[1]['voc_mean']:.4f})")

    # Find best model by score error
    best_error_model = min(
        comparison.items(),
        key=lambda x: x[1]['score_error_mean'] if x[1]['score_error_mean'] else float('inf')
    )
    print(f"🏆 Best model by Score Error: {best_error_model[0]} (Error: {best_error_model[1]['score_error_mean']:.4f})")


def example_3_custom_analysis():
    """Example 3: Custom analysis using core functions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Analysis")
    print("=" * 80 + "\n")

    # Load results
    results = load_results('path/to/results.jsonl')

    # Filter specific conditions
    high_error_samples = []
    false_positive_samples = []

    for result in results:
        # Get GT values
        meta = result.get('meta_data', {})
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        # Get predictions
        pred_ref_str = result.get('ref')
        pred_score_str = result.get('score')

        # Parse predictions
        if pred_ref_str and pred_ref_str.isdigit():
            pred_ref = int(pred_ref_str)
        else:
            pred_ref = pred_ref_str

        if pred_score_str and pred_score_str != "n/a":
            try:
                pred_score = float(pred_score_str.strip().replace('%', '')) / 100
            except:
                pred_score = None
        else:
            pred_score = pred_score_str

        # Calculate false positives
        ref_fp, score_fp = calculate_false_positives(pred_ref, pred_score, gt_ref, gt_score)

        if ref_fp or score_fp:
            false_positive_samples.append({
                'id': meta.get('id'),
                'ref_fp': ref_fp,
                'score_fp': score_fp,
                'gt_ref': gt_ref,
                'pred_ref': pred_ref,
                'gt_score': gt_score,
                'pred_score': pred_score
            })

        # Find high error samples
        if gt_score and isinstance(pred_score, (int, float)):
            error = calculate_evaluation_score(pred_score, gt_score)
            if error != float('inf') and error > 0.5:  # More than 50% error
                high_error_samples.append({
                    'id': meta.get('id'),
                    'error': error,
                    'gt_score': gt_score,
                    'pred_score': pred_score
                })

    print(f"Found {len(false_positive_samples)} false positive samples")
    print(f"Found {len(high_error_samples)} high error samples (>50%)")

    # Analyze false positives by type
    ref_fp_only = sum(1 for s in false_positive_samples if s['ref_fp'] and not s['score_fp'])
    score_fp_only = sum(1 for s in false_positive_samples if s['score_fp'] and not s['ref_fp'])
    both_fp = sum(1 for s in false_positive_samples if s['ref_fp'] and s['score_fp'])

    print(f"\nFalse Positive Breakdown:")
    print(f"  Ref FP only:   {ref_fp_only}")
    print(f"  Score FP only: {score_fp_only}")
    print(f"  Both FP:       {both_fp}")

    # Show a few examples
    if false_positive_samples:
        print(f"\nExample false positive cases (first 3):")
        for sample in false_positive_samples[:3]:
            print(f"  ID: {sample['id']}")
            print(f"    GT: ref={sample['gt_ref']}, score={sample['gt_score']}")
            print(f"    Pred: ref={sample['pred_ref']}, score={sample['pred_score']}")
            print(f"    FP: ref_fp={sample['ref_fp']}, score_fp={sample['score_fp']}")


def example_4_trajectory_analysis():
    """Example 4: Analyze specific trajectories."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Trajectory Analysis")
    print("=" * 80 + "\n")

    # Load results
    results = load_results('path/to/results.jsonl')

    # Group by trajectory
    from collections import defaultdict
    trajectories = defaultdict(list)

    for result in results:
        meta = result.get('meta_data', {})
        traj_id = meta.get('id', '')
        trajectories[traj_id].append(result)

    # Calculate VOC for all trajectories
    voc_metrics = calculate_voc_metrics(results)

    print(f"Total trajectories: {len(trajectories)}")
    print(f"Valid trajectories for VOC: {voc_metrics['voc_count']}")
    print(f"Mean VOC: {voc_metrics['voc_mean']:.4f}" if voc_metrics['voc_mean'] else "Mean VOC: N/A")

    # Find best and worst trajectories
    if voc_metrics['voc_values']:
        traj_vocs = []
        for idx, (traj_id, samples) in enumerate(trajectories.items()):
            if idx < len(voc_metrics['voc_values']):
                traj_vocs.append((traj_id, voc_metrics['voc_values'][idx]))

        traj_vocs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🏆 Top 3 trajectories by VOC:")
        for traj_id, voc in traj_vocs[:3]:
            print(f"  {traj_id}: {voc:.4f}")

        print(f"\n⚠️  Bottom 3 trajectories by VOC:")
        for traj_id, voc in traj_vocs[-3:]:
            print(f"  {traj_id}: {voc:.4f}")


def example_5_error_distribution():
    """Example 5: Analyze error distribution."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Error Distribution Analysis")
    print("=" * 80 + "\n")

    # Load results
    results = load_results('path/to/results.jsonl')

    # Collect all errors
    score_errors = []
    ref_errors = []

    for result in results:
        meta = result.get('meta_data', {})
        gt_ref = meta.get('closest_idx')
        gt_score = meta.get('progress_score')

        # Parse predictions
        pred_ref_str = result.get('ref')
        pred_score_str = result.get('score')

        # Calculate errors
        if gt_score and pred_score_str and pred_score_str != "n/a":
            try:
                pred_score = float(pred_score_str.strip().replace('%', '')) / 100
                error = calculate_evaluation_score(pred_score, gt_score)
                if error != float('inf'):
                    score_errors.append(error)
            except:
                pass

        if gt_ref and pred_ref_str and pred_ref_str.isdigit():
            from progress_evaluation import calculate_ref_error
            pred_ref = int(pred_ref_str)
            error = calculate_ref_error(pred_ref, gt_ref)
            if error != float('inf'):
                ref_errors.append(error)

    # Calculate percentiles
    if score_errors:
        import numpy as np
        print("Score Error Distribution:")
        print(f"  Min:    {np.min(score_errors):.4f}")
        print(f"  25th:   {np.percentile(score_errors, 25):.4f}")
        print(f"  Median: {np.median(score_errors):.4f}")
        print(f"  75th:   {np.percentile(score_errors, 75):.4f}")
        print(f"  Max:    {np.max(score_errors):.4f}")
        print(f"  Mean:   {np.mean(score_errors):.4f}")
        print(f"  Std:    {np.std(score_errors):.4f}")

        # Error ranges
        ranges = [
            ('0-10%', 0, 0.1),
            ('10-25%', 0.1, 0.25),
            ('25-50%', 0.25, 0.5),
            ('>50%', 0.5, float('inf'))
        ]

        print(f"\nError Range Distribution:")
        for name, low, high in ranges:
            count = sum(1 for e in score_errors if low <= e < high)
            pct = count / len(score_errors) * 100
            print(f"  {name:<10} {count:>4} ({pct:>5.1f}%)")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("PROGRESS EVALUATION MODULE - USAGE EXAMPLES")
    print("=" * 80)

    # Note: These examples use placeholder paths
    # Replace with your actual file paths when running

    print("\n⚠️  Note: These are example demonstrations.")
    print("   Replace file paths with your actual result files to run.\n")

    # Uncomment the examples you want to run:

    # example_1_single_file_evaluation()
    # example_2_model_comparison()
    # example_3_custom_analysis()
    # example_4_trajectory_analysis()
    # example_5_error_distribution()

    print("\n" + "=" * 80)
    print("To use these examples:")
    print("1. Replace 'path/to/results.jsonl' with your actual file path")
    print("2. Uncomment the example functions you want to run")
    print("3. Run: python example_usage.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
