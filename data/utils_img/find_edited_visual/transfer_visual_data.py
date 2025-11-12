#!/usr/bin/env python3
"""
Transfer visual data from edited_visual_nega.jsonl to visual_demo format.

This script:
1. Reads edited_visual_nega.jsonl data
2. Finds matching entries in visual_demo/*.jsonl by id and task_goal
3. Transforms the matched data with modified fields
4. Outputs to edited_visual_transfer_raw.jsonl
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# File paths
# EDITED_VISUAL_NEGA_PATH = "/projects/p32958/chengxuan/ProgressLM/data/negative/final_edited/edited_visual_nega.jsonl"
# VISUAL_DEMO_DIR = "/projects/p32958/chengxuan/ProgressLM/data/raw/visual_demo"
# OUTPUT_PATH = "/projects/p32958/chengxuan/ProgressLM/data/negative/final_edited/edited_visual_transfer_raw.jsonl"

EDITED_VISUAL_NEGA_PATH = "/projects/p32958/chengxuan/ProgressLM/data/raw/edit_imgs/labeled/human_annotated_2.jsonl"
VISUAL_DEMO_DIR = "/projects/p32958/chengxuan/ProgressLM/data/raw/visual_demo"
OUTPUT_PATH = "/projects/p32958/chengxuan/ProgressLM/data/raw/edit_imgs/labeled/edited_visual_nega_2.jsonl"



def load_visual_demo_data() -> Dict[Tuple[str, str], dict]:
    """
    Load all visual_demo jsonl files and build an index.

    Returns:
        Dictionary mapping (id, task_goal) -> first matching sample data
    """
    print(f"📂 Loading visual_demo data from: {VISUAL_DEMO_DIR}")

    index = {}
    jsonl_files = list(Path(VISUAL_DEMO_DIR).glob("*.jsonl"))

    print(f"   Found {len(jsonl_files)} JSONL files")

    for jsonl_file in tqdm(jsonl_files, desc="Loading visual_demo files"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    key = (data.get('id'), data.get('task_goal'))

                    # Only store the first occurrence
                    if key not in index:
                        index[key] = data
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing line in {jsonl_file}: {e}")
                    continue

    print(f"✅ Loaded {len(index)} unique (id, task_goal) pairs\n")
    return index


def load_edited_visual_nega() -> List[dict]:
    """
    Load all records from edited_visual_nega.jsonl.

    Returns:
        List of all records
    """
    print(f"📂 Loading edited visual data from: {EDITED_VISUAL_NEGA_PATH}")

    records = []
    with open(EDITED_VISUAL_NEGA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line: {e}")
                continue

    print(f"✅ Loaded {len(records)} records\n")
    return records


def modify_image_name(image_name: str) -> str:
    """
    Modify image name by adding '_edit' before the extension.

    Example: camera_top_0051.jpg -> camera_top_0051_edit.jpg
    """
    if '.' in image_name:
        name, ext = image_name.rsplit('.', 1)
        return f"{name}_edit.{ext}"
    else:
        return f"{image_name}_edit"


def process_record(record: dict, visual_demo_index: Dict[Tuple[str, str], dict]) -> Optional[dict]:
    """
    Process a single record from edited_visual_nega.

    Args:
        record: Record from edited_visual_nega.jsonl
        visual_demo_index: Index of visual_demo data

    Returns:
        Transformed record if match found, None otherwise
    """
    # Extract id and task_goal from meta_data
    meta_data = record.get('meta_data', {})
    record_id = meta_data.get('id')
    task_goal = meta_data.get('task_goal')
    image = meta_data.get('image')

    if not record_id or not task_goal:
        return None

    # Look up in index
    key = (record_id, task_goal)
    matched_data = visual_demo_index.get(key)

    if matched_data is None:
        return None

    # Create a copy of matched data
    result = matched_data.copy()

    # Modify fields
    result['closest_idx'] = "n/a"
    result['progress_score'] = "n/a"

    # Modify stage_to_estimate with edited image name
    if image:
        modified_image = modify_image_name(image)
        result['stage_to_estimate'] = [modified_image]
    else:
        result['stage_to_estimate'] = ["n/a"]

    return result


def main():
    """Main execution function."""
    print("=" * 70)
    print("Visual Data Transfer Tool")
    print("=" * 70)
    print()

    # Step 1: Load visual_demo data and build index
    visual_demo_index = load_visual_demo_data()

    # Step 2: Load edited_visual_nega records
    edited_records = load_edited_visual_nega()

    # Step 3: Process records with multithreading
    print("🔄 Processing records...")

    results = []
    unmatched = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_record = {
            executor.submit(process_record, record, visual_demo_index): record
            for record in edited_records
        }

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_record), total=len(edited_records), desc="Matching records"):
            record = future_to_record[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    unmatched.append(record)
            except Exception as e:
                print(f"\n⚠️  Error processing record: {e}")
                unmatched.append(record)

    print()

    # Step 4: Write results to output file
    print(f"💾 Writing results to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Step 5: Print statistics
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✅ Total records processed:  {len(edited_records)}")
    print(f"✅ Successfully matched:     {len(results)}")
    print(f"⚠️  Unmatched records:        {len(unmatched)}")
    print(f"📊 Match rate:               {len(results)/len(edited_records)*100:.2f}%")
    print()

    # Print unmatched records details
    if unmatched:
        print("Unmatched Records Details:")
        print("-" * 70)
        for i, record in enumerate(unmatched[:10], 1):  # Show first 10
            meta_data = record.get('meta_data', {})
            print(f"{i}. ID: {meta_data.get('id', 'N/A')}")
            print(f"   Task Goal: {meta_data.get('task_goal', 'N/A')}")
            print(f"   Image: {meta_data.get('image', 'N/A')}")
            print()

        if len(unmatched) > 10:
            print(f"... and {len(unmatched) - 10} more unmatched records")
            print()

        # Save unmatched records to a separate file
        unmatched_path = OUTPUT_PATH.replace('.jsonl', '_unmatched.jsonl')
        with open(unmatched_path, 'w', encoding='utf-8') as f:
            for record in unmatched:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"📝 Unmatched records saved to: {unmatched_path}")

    print()
    print("✅ Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
