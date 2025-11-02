#!/usr/bin/env python3
"""
Replace closest_idx and progress_score fields with "n/a" in JSONL files.

This script reads a JSONL file and replaces the values of 'closest_idx'
and 'progress_score' fields with the string "n/a".

Usage:
    python replace_fields_to_na.py <input_file> [output_file]

Example:
    python replace_fields_to_na.py input.jsonl output.jsonl
    python replace_fields_to_na.py input.jsonl  # Auto-generates output name
"""

import json
import sys
from pathlib import Path


def replace_fields_to_na(entry):
    """
    Replace closest_idx and progress_score with "n/a".

    Args:
        entry: Dictionary containing the JSONL entry

    Returns:
        Dictionary with replaced fields
    """
    # Create a copy to avoid modifying the original
    modified = entry.copy()

    # Replace the fields with "n/a"
    if 'closest_idx' in modified:
        modified['closest_idx'] = "n/a"

    if 'progress_score' in modified:
        modified['progress_score'] = "n/a"

    return modified


def process_jsonl(input_path, output_path):
    """
    Process an entire JSONL file and replace fields.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    processed_count = 0
    error_count = 0

    print(f"Processing: {input_path}")
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

                # Replace fields
                modified = replace_fields_to_na(entry)

                # Write to output file
                outfile.write(json.dumps(modified, ensure_ascii=False) + '\n')
                processed_count += 1

            except Exception as e:
                print(f"Error on line {line_num}: {e}", file=sys.stderr)
                error_count += 1

    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {processed_count} entries")
    if error_count > 0:
        print(f"  Errors encountered: {error_count} entries")

    return processed_count, error_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python replace_fields_to_na.py <input_file> [output_file]")
        print("\nExample:")
        print("  python replace_fields_to_na.py input.jsonl output.jsonl")
        print("  python replace_fields_to_na.py input.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]

    # Generate output filename if not provided
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Auto-generate output filename
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_na{input_path.suffix}"

    try:
        process_jsonl(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
