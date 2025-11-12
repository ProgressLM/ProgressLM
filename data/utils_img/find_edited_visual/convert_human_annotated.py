#!/usr/bin/env python3
"""
Convert human_annotated.jsonl to visual_edit_nega format.

This script converts human-annotated visual negative data to the standard
training format with edited image references.
"""

import json
import sys
import os
from pathlib import Path


def modify_image_name(image_name):
    """
    Modify image name by adding '_edit' before the extension.

    Example: camera_top_0020.jpg -> camera_top_0020_edit.jpg
    """
    if '.' in image_name:
        name, ext = image_name.rsplit('.', 1)
        return f"{name}_edit.{ext}"
    else:
        return f"{image_name}_edit"


def convert_entry(entry):
    """
    Convert a single human-annotated entry to standard format.

    Args:
        entry: Dictionary containing the human-annotated data

    Returns:
        Dictionary in standard training format
    """
    meta_data = entry.get('meta_data', {})

    # Extract required fields
    record_id = meta_data.get('id')
    task_goal = meta_data.get('task_goal')
    text_demo = meta_data.get('text_demo', [])
    data_source = meta_data.get('data_source')
    image = meta_data.get('image')

    # Calculate total steps
    total_steps = str(len(text_demo))

    # Modify image name to edited version
    if image:
        edited_image = modify_image_name(image)
        stage_to_estimate = [edited_image]
    else:
        stage_to_estimate = ["n/a"]

    # Build the converted entry
    converted = {
        'id': record_id,
        'task_goal': task_goal,
        'text_demo': text_demo,
        'total_steps': total_steps,
        'stage_to_estimate': stage_to_estimate,
        'closest_idx': 'n/a',
        'progress_score': 'n/a',
        'data_source': data_source
    }

    return converted


def convert_jsonl(input_path, output_path):
    """
    Convert an entire JSONL file from human-annotated format to standard format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    converted_count = 0
    error_count = 0

    print(f"Converting: {input_path}")
    print(f"Output to: {output_path}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON
                entry = json.loads(line)

                # Convert to standard format
                converted = convert_entry(entry)

                # Write to output file
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                converted_count += 1

            except Exception as e:
                print(f"Error on line {line_num}: {e}", file=sys.stderr)
                error_count += 1

    print(f"\nConversion complete!")
    print(f"  Successfully converted: {converted_count} entries")
    if error_count > 0:
        print(f"  Errors encountered: {error_count} entries")

    return converted_count, error_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_human_annotated.py <input_file> [output_file]")
        print("\nExample:")
        print("  python convert_human_annotated.py human_annotated_2.jsonl visual_edit_nega_2.jsonl")
        print("  python convert_human_annotated.py human_annotated_2.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]

    # Generate output filename if not provided
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Auto-generate output filename
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"

    try:
        convert_jsonl(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
