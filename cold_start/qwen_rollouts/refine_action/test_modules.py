#!/usr/bin/env python3
"""
Test script for text cleaning system modules.
Run this to verify all modules are working correctly.
"""

import os
import sys
import json
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_text_dataset import load_clean_text_dataset
from clean_text_prompt import build_clean_text_prompt, build_clean_text_prompt_from_item, CLEAN_TEXT_SYSTEM_PROMPT
from text_format_validator import is_sample_format_valid, validate_text_format


def test_dataset_loader():
    """Test dataset loading functionality."""
    print("=" * 70)
    print("Testing Dataset Loader...")
    print("=" * 70)

    # Create a temporary test dataset
    test_data = [
        {
            "id": "test_001",
            "text_demo": "Step 1: First step\nBy now, our progress is 0.5.\n\nStep 2: Second step\nBy now, our progress is 1.0.",
            "total_steps": "2",
            "progress_score": 0.5
        },
        {
            "id": "test_002",
            "text_demo": "Step 1: Only step\nBy now, our progress is 1.0.",
            "total_steps": 1
        }
    ]

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    try:
        # Test loading without expansion
        print("\n1. Testing basic loading (num_inferences=1)...")
        data = load_clean_text_dataset(temp_file, num_inferences=1)
        assert len(data) == 2, f"Expected 2 samples, got {len(data)}"
        assert data[0]['id'] == 'test_001'
        assert data[0]['total_steps'] == 2
        print("   ✓ Basic loading works")

        # Test loading with expansion
        print("\n2. Testing data expansion (num_inferences=3)...")
        data = load_clean_text_dataset(temp_file, num_inferences=3)
        assert len(data) == 6, f"Expected 6 samples (2×3), got {len(data)}"
        assert data[0]['_inference_idx'] == 0
        assert data[1]['_inference_idx'] == 1
        assert data[2]['_inference_idx'] == 2
        print("   ✓ Data expansion works")

        print("\n✓ Dataset loader tests passed!\n")

    finally:
        # Cleanup
        os.unlink(temp_file)


def test_prompt_builder():
    """Test prompt building functionality."""
    print("=" * 70)
    print("Testing Prompt Builder...")
    print("=" * 70)

    test_text = "Step 1: First step\nBy now, our progress is 0.5.\n\nStep 2: Second step\nBy now, our progress is 1.0."

    print("\n1. Testing build_clean_text_prompt()...")
    messages = build_clean_text_prompt(test_text)
    assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
    assert messages[0]['type'] == 'text'
    assert test_text in messages[0]['value']
    print("   ✓ Prompt building works")

    print("\n2. Testing build_clean_text_prompt_from_item()...")
    item = {'text_demo': test_text}
    messages = build_clean_text_prompt_from_item(item)
    assert len(messages) == 1
    assert messages[0]['type'] == 'text'
    print("   ✓ Prompt building from item works")

    print("\n3. Checking system prompt...")
    assert len(CLEAN_TEXT_SYSTEM_PROMPT) > 0
    assert "expert text editor" in CLEAN_TEXT_SYSTEM_PROMPT.lower()
    print("   ✓ System prompt is defined")

    print("\n✓ Prompt builder tests passed!\n")


def test_format_validator():
    """Test format validation functionality."""
    print("=" * 70)
    print("Testing Format Validator...")
    print("=" * 70)

    # Valid sample
    print("\n1. Testing valid format...")
    valid_sample = {
        'id': 'test_001',
        'text_demo': 'Step 1: First\nBy now, our progress is 0.5.\n\nStep 2: Second\nBy now, our progress is 1.0.',
        'total_steps': 2
    }
    is_valid, errors = is_sample_format_valid(valid_sample)
    assert is_valid, f"Valid sample marked as invalid: {errors}"
    assert len(errors) == 0
    print("   ✓ Valid format recognized")

    # Missing step
    print("\n2. Testing missing step...")
    missing_step_sample = {
        'id': 'test_002',
        'text_demo': 'Step 1: First\nBy now, our progress is 0.5.',
        'total_steps': 2
    }
    is_valid, errors = is_sample_format_valid(missing_step_sample)
    assert not is_valid, "Missing step not detected"
    assert len(errors) > 0
    print(f"   ✓ Missing step detected: {errors[0]}")

    # Wrong progress marker
    print("\n3. Testing wrong progress marker...")
    wrong_progress_sample = {
        'id': 'test_003',
        'text_demo': 'Step 1: First\nBy now, our progress is 0.8.\n\nStep 2: Second\nBy now, our progress is 1.0.',
        'total_steps': 2
    }
    is_valid, errors = is_sample_format_valid(wrong_progress_sample)
    assert not is_valid, "Wrong progress marker not detected"
    assert len(errors) > 0
    print(f"   ✓ Wrong progress marker detected")

    # Extra step
    print("\n4. Testing extra step...")
    extra_step_sample = {
        'id': 'test_004',
        'text_demo': 'Step 1: First\nBy now, our progress is 0.5.\n\nStep 2: Second\nBy now, our progress is 1.0.\n\nStep 3: Extra',
        'total_steps': 2
    }
    is_valid, errors = is_sample_format_valid(extra_step_sample)
    assert not is_valid, "Extra step not detected"
    assert any('多余步骤' in err or 'Step 3' in err for err in errors)
    print(f"   ✓ Extra step detected")

    # Test different progress formats
    print("\n5. Testing different progress formats...")
    formats_to_test = [
        ('0.33', 1, 3),  # 0.33
        ('0.3333333333333333', 1, 3),  # Long float
        ('1.0', 2, 2),  # 1.0
        ('1', 2, 2),  # 1 (integer)
    ]

    for progress_str, step_num, total_steps in formats_to_test:
        sample = {
            'id': 'test_format',
            'text_demo': f'Step {step_num}: Test\nBy now, our progress is {progress_str}.',
            'total_steps': total_steps
        }
        is_valid, errors = is_sample_format_valid(sample)
        # Note: Some formats might not match exactly, that's expected
        print(f"   - Progress '{progress_str}' (step {step_num}/{total_steps}): {'✓' if is_valid else '✗'}")

    print("\n✓ Format validator tests passed!\n")


def test_validate_text_format():
    """Test simplified validation function."""
    print("=" * 70)
    print("Testing Simplified Validator...")
    print("=" * 70)

    text_demo = 'Step 1: First\nBy now, our progress is 0.5.\n\nStep 2: Second\nBy now, our progress is 1.0.'
    result = validate_text_format(text_demo, 2, 'test_simple')
    assert result is True, "Valid format not recognized"
    print("   ✓ Simplified validator works")
    print("\n✓ Simplified validator tests passed!\n")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TEXT CLEANING SYSTEM - MODULE TESTS" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    try:
        test_dataset_loader()
        test_prompt_builder()
        test_format_validator()
        test_validate_text_format()

        print("=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\nThe text cleaning system is ready to use.")
        print("Run the following command to start processing:")
        print("\n  bash scripts/clean_text_comm.sh\n")

        return 0

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"\nError: {e}\n")
        return 1

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
