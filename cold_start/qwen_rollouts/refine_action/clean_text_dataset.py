import os
import json
from typing import Dict, Any, List


def load_clean_text_dataset(
    dataset_path: str,
    num_inferences: int = 1
) -> List[Dict[str, Any]]:
    """
    Load text cleaning dataset from JSONL file and expand each sample N times.

    Expected format for each line:
    {
        "id": "WikiHow_40810_1",
        "text_demo": "Back Up Messages...\n\nStep 1: ...",
        "total_steps": "8",
        "stage_to_estimate": "images/...",  # Optional, not used
        "closest_demo_idx": "1",            # Optional, not used
        "progress_score": 0.12,             # Optional, not used
        "data_source": "WikiHow"            # Optional, not used
    }

    Args:
        dataset_path: Path to the JSONL dataset file
        num_inferences: Number of times to replicate each sample (default: 1)

    Returns:
        List of expanded dataset items (length = original_length * num_inferences)
        Each item contains:
        - id: Sample ID
        - text_demo: Original text demonstration
        - total_steps: Total number of steps (converted to int)
        - _inference_idx: Inference index (0 to num_inferences-1)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load raw data
    raw_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)

                # Validate required fields
                required_fields = ['id', 'text_demo', 'total_steps']
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # Validate text_demo is a non-empty string
                if not isinstance(item['text_demo'], str) or len(item['text_demo'].strip()) == 0:
                    print(f"Warning: Line {line_num} has invalid text_demo (must be non-empty string), skipping")
                    continue

                # Validate and convert total_steps to integer
                try:
                    item['total_steps'] = int(item['total_steps'])
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid total_steps (must be convertible to int), skipping")
                    continue

                if item['total_steps'] <= 0:
                    print(f"Warning: Line {line_num} has total_steps <= 0, skipping")
                    continue

                # Keep only necessary fields (optional fields can be preserved if present)
                clean_item = {
                    'id': item['id'],
                    'text_demo': item['text_demo'],
                    'total_steps': item['total_steps']
                }

                # Preserve optional fields if they exist (for potential later use)
                for optional_field in ['stage_to_estimate', 'closest_demo_idx', 'progress_score', 'data_source']:
                    if optional_field in item:
                        clean_item[optional_field] = item[optional_field]

                raw_data.append(clean_item)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue

    print(f"Loaded {len(raw_data)} raw samples from {dataset_path}")

    # Expand data: replicate each sample num_inferences times
    expanded_data = []
    for item in raw_data:
        for inference_idx in range(num_inferences):
            expanded_item = item.copy()
            expanded_item['_inference_idx'] = inference_idx  # Internal marker
            expanded_data.append(expanded_item)

    if num_inferences > 1:
        print(f"Expanded to {len(expanded_data)} samples (×{num_inferences})")
    else:
        print(f"Total samples: {len(expanded_data)}")

    return expanded_data
