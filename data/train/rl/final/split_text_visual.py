#!/usr/bin/env python3
"""
Split candidate_rl_data_20k.jsonl into separate text and visual demo files.
"""

import json
from pathlib import Path

# File paths
BASE_DIR = Path("/gpfs/projects/p32958/chengxuan/ProgressLM/data/train/rl/final")
INPUT_FILE = BASE_DIR / "candidate_rl_data_20k.jsonl"
TEXT_OUTPUT = BASE_DIR / "candidate_text_20k.jsonl"
VISUAL_OUTPUT = BASE_DIR / "candidate_visual_20k.jsonl"


def split_data():
    """Split data into text and visual demo files."""
    print("=" * 80)
    print("Splitting Text and Visual Demo Data")
    print("=" * 80)
    print()

    text_records = []
    visual_records = []

    # Read and categorize records
    print(f"Reading from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if 'text_demo' in record:
                    text_records.append(record)
                elif 'visual_demo' in record:
                    visual_records.append(record)
                else:
                    print(f"Warning: Record with neither text_demo nor visual_demo: {record.get('id', 'unknown')}")

    print(f"Total records read: {len(text_records) + len(visual_records)}")
    print(f"  Text demo records: {len(text_records)}")
    print(f"  Visual demo records: {len(visual_records)}")
    print()

    # Write text demo file
    print(f"Writing text demos to: {TEXT_OUTPUT}")
    with open(TEXT_OUTPUT, 'w', encoding='utf-8') as f:
        for record in text_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Wrote {len(text_records)} records")

    # Write visual demo file
    print(f"Writing visual demos to: {VISUAL_OUTPUT}")
    with open(VISUAL_OUTPUT, 'w', encoding='utf-8') as f:
        for record in visual_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Wrote {len(visual_records)} records")

    print()
    print("=" * 80)
    print("Split completed successfully!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Text output: {TEXT_OUTPUT} ({len(text_records)} records)")
    print(f"  Visual output: {VISUAL_OUTPUT} ({len(visual_records)} records)")
    print()


if __name__ == "__main__":
    split_data()
