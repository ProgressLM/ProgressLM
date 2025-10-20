import os
import json
from typing import Dict, Any, List


def load_worldvlm_dataset(dataset_path: str, image_root: str = None) -> List[Dict[str, Any]]:
    """
    Load WorldVLM dataset from JSONL file.

    Expected format for each line:
    {
        "data_source": "",
        "start_img": "",
        "end_img": "",
        "task_goal": "",
        "step": "",
        "action": "",
        "source_id": ""
    }

    Args:
        dataset_path: Path to the JSONL dataset file
        image_root: Optional root directory to prepend to image paths

    Returns:
        List of dataset items
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Validate required fields
                required_fields = ['start_img', 'end_img', 'action']
                for field in required_fields:
                    if field not in item:
                        print(f"Warning: Line {line_num} missing required field '{field}', skipping")
                        continue

                # Prepend image_root to image paths if provided
                if image_root:
                    if 'start_img' in item and item['start_img']:
                        # Only prepend if path is not already absolute
                        if not os.path.isabs(item['start_img']):
                            item['start_img'] = os.path.join(image_root, item['start_img'])
                    if 'end_img' in item and item['end_img']:
                        # Only prepend if path is not already absolute
                        if not os.path.isabs(item['end_img']):
                            item['end_img'] = os.path.join(image_root, item['end_img'])

                # Add line number as index if source_id not present
                if 'source_id' not in item or not item['source_id']:
                    item['source_id'] = f"line_{line_num}"

                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue

    print(f"Loaded {len(data)} samples from {dataset_path}")
    if image_root:
        print(f"Image root directory: {image_root}")
    return data


def dump_worldvlm_images(item: Dict[str, Any]) -> List[str]:
    """
    Extract image paths from WorldVLM dataset item.
    Returns a list of [start_img_path, end_img_path]

    Args:
        item: Dataset item containing 'start_img' and 'end_img' fields

    Returns:
        List of image paths [start_img, end_img]
    """
    start_img = item.get('start_img', '')
    end_img = item.get('end_img', '')

    # Validate that images exist
    if not os.path.exists(start_img):
        raise FileNotFoundError(f"start_img not found: {start_img}")
    if not os.path.exists(end_img):
        raise FileNotFoundError(f"end_img not found: {end_img}")

    return [start_img, end_img]


def get_worldvlm_action(item: Dict[str, Any]) -> str:
    """
    Extract action text from WorldVLM dataset item.

    Args:
        item: Dataset item containing 'action' field

    Returns:
        Action text string
    """
    return item.get('action', '')
