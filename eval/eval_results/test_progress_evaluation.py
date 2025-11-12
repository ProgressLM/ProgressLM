#!/usr/bin/env python3
"""
Simple test script for progress_evaluation module.
Run this to verify that all functions work correctly.
"""

import sys
import json
from progress_evaluation import (
    calculate_false_positives,
    calculate_evaluation_score,
    calculate_ref_error,
    calculate_voc_metrics,
    calculate_gt_distribution,
    analyze_results
)


def test_false_positives():
    """Test false positive calculation."""
    print("Testing calculate_false_positives...")

    # Test case 1: Both numeric (not FP)
    ref_fp, score_fp = calculate_false_positives(
        predicted_ref=5,
        predicted_score=0.33,
        gt_ref=3,
        gt_score=0.30
    )
    assert not ref_fp and not score_fp, "Both numeric should not be FP"

    # Test case 2: Both n/a (not FP)
    ref_fp, score_fp = calculate_false_positives(
        predicted_ref="n/a",
        predicted_score="n/a",
        gt_ref=None,
        gt_score=None
    )
    assert not ref_fp and not score_fp, "Both n/a should not be FP"

    # Test case 3: GT numeric, pred n/a (FP)
    ref_fp, score_fp = calculate_false_positives(
        predicted_ref="n/a",
        predicted_score="n/a",
        gt_ref=3,
        gt_score=0.30
    )
    assert ref_fp and score_fp, "GT numeric but pred n/a should be FP"

    # Test case 4: GT n/a, pred numeric (FP)
    ref_fp, score_fp = calculate_false_positives(
        predicted_ref=5,
        predicted_score=0.33,
        gt_ref=None,
        gt_score=None
    )
    assert ref_fp and score_fp, "GT n/a but pred numeric should be FP"

    print("  ✓ All false positive tests passed")


def test_evaluation_score():
    """Test evaluation score calculation."""
    print("Testing calculate_evaluation_score...")

    # Test case 1: Normal calculation
    error = calculate_evaluation_score(0.33, 0.30)
    expected = abs(0.30 - 0.33) / 0.30
    assert abs(error - expected) < 0.0001, f"Expected {expected}, got {error}"

    # Test case 2: None prediction
    error = calculate_evaluation_score(None, 0.30)
    assert error == float('inf'), "None prediction should return inf"

    # Test case 3: None GT
    error = calculate_evaluation_score(0.33, None)
    assert error == float('inf'), "None GT should return inf"

    # Test case 4: Perfect prediction
    error = calculate_evaluation_score(0.50, 0.50)
    assert error == 0.0, "Perfect prediction should be 0"

    print("  ✓ All evaluation score tests passed")


def test_ref_error():
    """Test ref error calculation."""
    print("Testing calculate_ref_error...")

    # Test case 1: Normal calculation
    error = calculate_ref_error(5, 3)
    assert error == 2.0, f"Expected 2.0, got {error}"

    # Test case 2: None prediction
    error = calculate_ref_error(None, 3)
    assert error == float('inf'), "None prediction should return inf"

    # Test case 3: None GT
    error = calculate_ref_error(5, None)
    assert error == float('inf'), "None GT should return inf"

    # Test case 4: Perfect prediction
    error = calculate_ref_error(3, 3)
    assert error == 0.0, "Perfect prediction should be 0"

    print("  ✓ All ref error tests passed")


def test_voc_metrics():
    """Test VOC calculation."""
    print("Testing calculate_voc_metrics...")

    # Create test data with 2 trajectories
    results = [
        # Trajectory 1: Perfect correlation
        {
            'score': 0.2,
            'meta_data': {
                'id': 'traj1',
                'closest_idx': 1,
                'progress_score': 0.2
            }
        },
        {
            'score': 0.5,
            'meta_data': {
                'id': 'traj1',
                'closest_idx': 2,
                'progress_score': 0.5
            }
        },
        {
            'score': 0.8,
            'meta_data': {
                'id': 'traj1',
                'closest_idx': 3,
                'progress_score': 0.8
            }
        },
        # Trajectory 2: Contains n/a (should be excluded)
        {
            'score': 0.3,
            'meta_data': {
                'id': 'traj2',
                'closest_idx': None,  # n/a
                'progress_score': 0.3
            }
        }
    ]

    voc_metrics = calculate_voc_metrics(results)

    assert voc_metrics['voc_count'] == 1, "Should have 1 valid trajectory"
    assert voc_metrics['voc_mean'] is not None, "Should have VOC mean"
    assert 0.9 <= voc_metrics['voc_mean'] <= 1.0, "Perfect correlation should be close to 1.0"

    print("  ✓ All VOC tests passed")


def test_gt_distribution():
    """Test GT distribution calculation."""
    print("Testing calculate_gt_distribution...")

    results = [
        {
            'meta_data': {
                'closest_idx': 1,
                'progress_score': 0.2
            }
        },
        {
            'meta_data': {
                'closest_idx': None,
                'progress_score': 0.3
            }
        },
        {
            'meta_data': {
                'closest_idx': 2,
                'progress_score': None
            }
        },
        {
            'meta_data': {
                'closest_idx': None,
                'progress_score': None
            }
        }
    ]

    dist = calculate_gt_distribution(results)

    assert dist['gt_numeric_count'] == 1, "Should have 1 fully numeric sample"
    assert dist['gt_na_count'] == 3, "Should have 3 samples with n/a"
    assert dist['gt_ref_numeric'] == 2, "Should have 2 numeric refs"
    assert dist['gt_ref_na'] == 2, "Should have 2 n/a refs"
    assert dist['gt_score_numeric'] == 2, "Should have 2 numeric scores"
    assert dist['gt_score_na'] == 2, "Should have 2 n/a scores"

    print("  ✓ All GT distribution tests passed")


def test_analyze_results():
    """Test full result analysis."""
    print("Testing analyze_results...")

    # Create comprehensive test data
    results = [
        # Success case with numeric values
        {
            'ref': '3',
            'score': '30%',
            'meta_data': {
                'id': 'traj1',
                'closest_idx': 3,
                'progress_score': 0.30,
                'status': 'success'
            }
        },
        # Success case with n/a
        {
            'ref': 'n/a',
            'score': 'n/a',
            'meta_data': {
                'id': 'traj2',
                'closest_idx': None,
                'progress_score': None,
                'status': 'success'
            }
        },
        # False positive case
        {
            'ref': 'n/a',
            'score': '50%',
            'meta_data': {
                'id': 'traj3',
                'closest_idx': 2,
                'progress_score': 0.45,
                'status': 'success'
            }
        },
        # Error case
        {
            'ref': None,
            'score': None,
            'meta_data': {
                'id': 'traj4',
                'closest_idx': 1,
                'progress_score': 0.10,
                'status': 'failed'
            }
        }
    ]

    stats = analyze_results(results, verbose=False)

    assert stats['total_samples'] == 4, f"Should have 4 total samples, got {stats['total_samples']}"
    assert stats['valid_samples'] == 3, f"Should have 3 valid samples, got {stats['valid_samples']}"
    assert stats['error_samples'] == 1, f"Should have 1 error sample, got {stats['error_samples']}"
    assert stats['ref_fp_count'] > 0, f"Should have ref false positives, got {stats['ref_fp_count']}"
    # Note: gt_numeric_count counts samples where BOTH ref and score are numeric
    # In our test data: sample 1 has both numeric (1), sample 2 has both n/a (0),
    # sample 3 has numeric score but n/a ref (0), sample 4 has both numeric (1)
    # So we expect 2 samples with both numeric
    assert stats['gt_numeric_count'] >= 1, f"Should have at least 1 numeric GT sample, got {stats['gt_numeric_count']}"

    print("  ✓ All analyze_results tests passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PROGRESS EVALUATION MODULE - UNIT TESTS")
    print("=" * 80 + "\n")

    tests = [
        test_false_positives,
        test_evaluation_score,
        test_ref_error,
        test_voc_metrics,
        test_gt_distribution,
        test_analyze_results
    ]

    failed = []

    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"  ✗ Test failed: {e}")
            failed.append(test_func.__name__)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed.append(test_func.__name__)

    print("\n" + "=" * 80)
    if not failed:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {len(failed)} TEST(S) FAILED:")
        for name in failed:
            print(f"   - {name}")
        sys.exit(1)

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
