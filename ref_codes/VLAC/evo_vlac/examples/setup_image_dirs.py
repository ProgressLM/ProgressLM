#!/usr/bin/env python3
"""
Setup image directory structure for VLAC example dataset
Creates {id}/{filename} directory structure and copies images
"""

import json
import os
import shutil
from pathlib import Path

def setup_image_directories():
    """
    Read JSONL and create IMAGE_ROOT/{id}/{filename} directory structure
    Copy images from images/ref and images/test to the new structure
    """

    base_dir = Path(__file__).parent
    jsonl_file = base_dir / "vlac_example_visual_demo.jsonl"
    source_ref_dir = base_dir / "images" / "ref"
    source_test_dir = base_dir / "images" / "test"

    # Track all unique IDs and their required images
    id_images = {}  # {id: set of filenames}

    print("Reading JSONL to determine required directory structure...")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line.strip())
            record_id = record['id']

            # Collect all image paths for this ID
            all_paths = record['visual_demo'] + record['stage_to_estimate']

            for path in all_paths:
                # Path format: {id}/{filename}
                # Extract the filename
                filename = os.path.basename(path)

                if record_id not in id_images:
                    id_images[record_id] = set()
                id_images[record_id].add(filename)

    print(f"Found {len(id_images)} unique IDs")
    print(f"Creating directory structure...\n")

    total_dirs = 0
    total_files = 0

    for record_id, filenames in id_images.items():
        # Create directory for this ID
        target_dir = base_dir / record_id
        target_dir.mkdir(parents=True, exist_ok=True)
        total_dirs += 1

        # Copy images
        for filename in filenames:
            # Determine source directory (ref or test based on filename pattern)
            if filename.startswith('599-'):
                source_file = source_ref_dir / filename
            elif filename.startswith('595-'):
                source_file = source_test_dir / filename
            else:
                print(f"  Warning: Unknown pattern for {filename}, skipping")
                continue

            target_file = target_dir / filename

            if not source_file.exists():
                print(f"  Warning: Source file not found: {source_file}")
                continue

            # Copy file
            shutil.copy2(source_file, target_file)
            total_files += 1

        print(f"✓ {record_id}: {len(filenames)} images")

    print(f"\n{'='*70}")
    print(f"Directory structure setup complete!")
    print(f"{'='*70}")
    print(f"Created {total_dirs} directories")
    print(f"Copied {total_files} images")
    print(f"\nStructure:")
    print(f"  IMAGE_ROOT/")
    print(f"  └── vlac_example_scoop_rice/")
    print(f"      ├── camera_0/")
    print(f"      │   ├── test_00/")
    print(f"      │   │   ├── 599-0-521-0.jpg")
    print(f"      │   │   ├── 599-100-521-0.jpg")
    print(f"      │   │   ├── ...")
    print(f"      │   │   └── 595-6-565-0.jpg")
    print(f"      │   ├── test_01/")
    print(f"      │   └── ...")
    print(f"      ├── camera_1/")
    print(f"      └── camera_2/")
    print(f"\nYou can now run the evaluation script with:")
    print(f"  IMAGE_ROOT={base_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    setup_image_directories()
