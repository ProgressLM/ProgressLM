import os
import json
from typing import Dict, Any, List, Tuple


def load_visual_demo_dataset(
    dataset_path: str,
    num_inferences: int = 4,
    image_root: str = None
) -> List[Dict[str, Any]]:
    """
    Load Visual Demo dataset from JSONL file and expand each sample N times.

    Expected format for each line:
    {
        "id": "WikiHow_85639_1",
        "visual_demo": ["img1.jpg", "img2.jpg", ...],  // variable length
        "stage_to_estimate": ["current.jpg"],
        "progress_score": 0.2,
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
                required_fields = ['id', 'visual_demo', 'stage_to_estimate', 'progress_score']
                missing_fields = [f for f in required_fields if f not in item]
                if missing_fields:
                    print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                    continue

                # Validate visual_demo is a list
                if not isinstance(item['visual_demo'], list) or len(item['visual_demo']) == 0:
                    print(f"Warning: Line {line_num} has invalid visual_demo (must be non-empty list), skipping")
                    continue

                # Validate stage_to_estimate is a list with 1 element
                if not isinstance(item['stage_to_estimate'], list) or len(item['stage_to_estimate']) != 1:
                    print(f"Warning: Line {line_num} has invalid stage_to_estimate (must be list with 1 element), skipping")
                    continue

                # Validate progress_score is numeric
                try:
                    item['progress_score'] = float(item['progress_score'])
                except (ValueError, TypeError):
                    print(f"Warning: Line {line_num} has invalid progress_score (must be numeric), skipping")
                    continue

                # Prepend image_root to image paths if provided
                if image_root:
                    item['visual_demo'] = [
                        os.path.join(image_root, img) if not os.path.isabs(img) else img
                        for img in item['visual_demo']
                    ]
                    item['stage_to_estimate'] = [
                        os.path.join(image_root, img) if not os.path.isabs(img) else img
                        for img in item['stage_to_estimate']
                    ]

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


def validate_image_paths(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that all image paths in the item exist.

    Args:
        item: Dataset item with 'visual_demo' and 'stage_to_estimate' fields

    Returns:
        (is_valid, error_message)
    """
    # Check visual_demo images
    for img_path in item['visual_demo']:
        if not os.path.exists(img_path):
            return False, f"visual_demo image not found: {img_path}"

    # Check stage_to_estimate image
    stage_img = item['stage_to_estimate'][0]
    if not os.path.exists(stage_img):
        return False, f"stage_to_estimate image not found: {stage_img}"

    return True, ""


def get_image_paths(item: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Extract image paths from Visual Demo dataset item.

    Args:
        item: Dataset item with 'visual_demo' and 'stage_to_estimate' fields

    Returns:
        (visual_demo_paths, stage_to_estimate_path)
    """
    visual_demo_paths = item['visual_demo']
    stage_to_estimate_path = item['stage_to_estimate'][0]

    return visual_demo_paths, stage_to_estimate_path
