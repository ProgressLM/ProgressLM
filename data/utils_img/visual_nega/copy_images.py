#!/usr/bin/env python3
"""
Script to copy images from stage_to_estimate with renamed format.
New naming format: {id_with_underscores}_{original_image_name}
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
JSONL_FILE = "/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img.jsonl"
SOURCE_IMAGE_DIR = "/projects/p32958/chengxuan/ProgressLM/data/images"
TARGET_DIR = "/projects/p32958/chengxuan/ProgressLM/data/images/visual_negative_replacement"

def main():
    # Create target directory if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Statistics
    total_images = 0
    copied_images = 0
    missing_images = 0
    errors = []

    print(f"Reading from: {JSONL_FILE}")
    print(f"Target directory: {TARGET_DIR}")
    print()

    # Read and process JSONL file
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} entries...")

    # Create progress bar
    pbar = tqdm(total=len(lines), desc="Processing", unit="entry")

    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())

            # Extract id and stage_to_estimate
            record_id = data.get('id', '')
            stage_images = data.get('stage_to_estimate', [])

            if not record_id:
                errors.append(f"Line {line_num}: Missing 'id' field")
                pbar.update(1)
                continue

            if not stage_images:
                pbar.update(1)
                continue  # Skip if no images to process

            # Convert id to filename prefix (replace / with _)
            id_prefix = record_id.replace('/', '_')

            # Process each image in stage_to_estimate
            for img_name in stage_images:
                total_images += 1

                # Construct source path
                source_path = os.path.join(SOURCE_IMAGE_DIR, record_id, img_name)

                # Construct new filename
                new_filename = f"{id_prefix}_{img_name}"
                target_path = os.path.join(TARGET_DIR, new_filename)

                # Check if source exists
                if not os.path.exists(source_path):
                    missing_images += 1
                    errors.append(f"Line {line_num}: Source not found: {source_path}")
                    continue

                # Copy the file
                try:
                    shutil.copy2(source_path, target_path)
                    copied_images += 1
                except Exception as e:
                    errors.append(f"Line {line_num}: Error copying {source_path} -> {target_path}: {str(e)}")

            # Update progress bar with current stats
            pbar.set_postfix({
                'Success': copied_images,
                'Missing': missing_images,
                'Errors': len(errors)
            })
            pbar.update(1)

        except json.JSONDecodeError as e:
            errors.append(f"Line {line_num}: JSON decode error: {str(e)}")
            pbar.update(1)
        except Exception as e:
            errors.append(f"Line {line_num}: Unexpected error: {str(e)}")
            pbar.update(1)

    pbar.close()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images to process: {total_images}")
    print(f"Successfully copied: {copied_images}")
    print(f"Missing source images: {missing_images}")
    print(f"Total errors: {len(errors)}")
    print()

    # Print errors if any
    if errors:
        print("ERRORS (showing first 20):")
        print("-"*60)
        for error in errors[:20]:
            print(error)

        if len(errors) > 20:
            print(f"\n... and {len(errors) - 20} more errors")

    print()
    print(f"Images copied to: {TARGET_DIR}")

if __name__ == "__main__":
    main()
