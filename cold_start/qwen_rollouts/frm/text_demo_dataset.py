import os
import json
from typing import Dict, Any, List, Tuple


def load_text_demo_dataset(
    dataset_path: str,
    num_inferences: int = 4,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load Text Demo dataset from JSONL file and expand each sample N times.

    Expected format for each line:
    {
        "id": "WikiHow_40810",
        "text_demo": "Back Up Messages...\n\nStep 1: ...\nBy now, our progress is 0.12.\n...",
        "stage_to_estimate": "images/comm/WikiHow/WikiHow_40810/231678.jpg",
        "progress_score": 0.25,
        "total_steps": "8",
        "closest_demo_idx": "2",
        "data_source": "WikiHow"
    }

    Args:
        dataset_path: Path to the JSONL dataset file
        num_inferences: Number of times to replicate each sample (default: 4)
        image_root: Optional root directory to prepend to relative image paths

    Returns:
        List of expanded dataset items (length = original_length * num_inferences)
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
                required_fields = ['id', 'text_demo', 'stage_to_estimate', 'progress_score']
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # Validate text_demo is a non-empty string
                if not isinstance(item['text_demo'], str) or len(item['text_demo'].strip()) == 0:
                    print(f"Warning: Line {line_num} has invalid text_demo (must be non-empty string), skipping")
                    continue

                # Normalize stage_to_estimate to string format
                if isinstance(item['stage_to_estimate'], str):
                    stage_img = item['stage_to_estimate']
                elif isinstance(item['stage_to_estimate'], list) and len(item['stage_to_estimate']) == 1:
                    stage_img = item['stage_to_estimate'][0]
                else:
                    print(f"Warning: Line {line_num} has invalid stage_to_estimate (must be string or list with 1 element), skipping")
                    continue

                item['stage_to_estimate'] = stage_img

                # Validate progress_score is numeric
                try:
                    item['progress_score'] = float(item['progress_score'])
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid progress_score (must be numeric), skipping")
                    continue

                # Prepend image_root to image path if provided
                if image_root and not os.path.isabs(item['stage_to_estimate']):
                    item['stage_to_estimate'] = os.path.join(image_root, item['stage_to_estimate'])

                raw_data.append(item)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue

    print(f"Loaded {len(raw_data)} raw samples from {dataset_path}")
    if image_root:
        print(f"Image root directory: {image_root}")

    # Expand data: replicate each sample num_inferences times
    expanded_data = []
    for item in raw_data:
        for inference_idx in range(num_inferences):
            expanded_item = item.copy()
            expanded_item['_inference_idx'] = inference_idx  # Internal marker
            expanded_data.append(expanded_item)

    print(f"Expanded to {len(expanded_data)} samples (×{num_inferences})")

    return expanded_data


def validate_image_path(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that the image path in stage_to_estimate exists.

    Args:
        item: Dataset item with 'stage_to_estimate' field

    Returns:
        (is_valid, error_message)
    """
    stage_img = item['stage_to_estimate']
    if not os.path.exists(stage_img):
        return False, f"stage_to_estimate image not found: {stage_img}"

    return True, ""


def get_text_and_image(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract text_demo and stage_to_estimate from Text Demo dataset item.

    Args:
        item: Dataset item with 'text_demo' and 'stage_to_estimate' fields

    Returns:
        (text_demo, stage_to_estimate_path)
    """
    text_demo = item['text_demo']
    stage_to_estimate_path = item['stage_to_estimate']

    return text_demo, stage_to_estimate_path
