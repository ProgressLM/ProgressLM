#!/bin/bash
################################################################################
# Batch convert Visual Demo datasets to LLaMA-Factory format
#
# This script converts multiple Visual Demo datasets by merging original data
# with CoT responses into ShareGPT format.
#
# Usage:
#   bash run_convert_visual.sh
#
# Configuration:
#   - Edit ORIGINAL_DIR: Directory containing original *_sft.jsonl files
#   - Edit COT_DIR: Directory containing CoT response *_cot.jsonl files
#   - Edit OUTPUT_DIR: Directory for output JSON files
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Configuration ====================
# Original data directory
ORIGINAL_DIR="/Users/cxqian/Codes/ProgressLM/data/train/visual_demo"

# CoT responses directory (UPDATE THIS PATH!)
COT_DIR="/Users/cxqian/Codes/ProgressLM/data/sft_data/visual"

# Output directory
OUTPUT_DIR="/Users/cxqian/Codes/ProgressLM/data/sft_data/visual"

# Dataset configurations: "dataset_name|original_file|cot_file"
DATASETS=(
    "visual_h5_agilex_3rgb|visual_h5_agilex_3rgb_sft.jsonl|visual_agilex_cold.jsonl"
    "visual_h5_franka_3rgb|visual_h5_franka_3rgb_sft.jsonl|visual_franka_cold.jsonl"
    "visual_h5_tienkung_xsens|visual_h5_tienkung_xsens_sft.jsonl|visual_h5_tienkung_cold.jsonl"
    "visual_h5_ur_1rgb|visual_h5_ur_1rgb_sft.jsonl|visual_hr_cold.jsonl"
)

# ==================== Validation ====================
echo "========================================"
echo "Visual Demo SFT Data Conversion"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo "Original data:    $ORIGINAL_DIR"
echo "CoT responses:    $COT_DIR"
echo "Output:           $OUTPUT_DIR"
echo ""

# Check directories
if [ ! -d "$ORIGINAL_DIR" ]; then
    echo "❌ Error: Original data directory not found: $ORIGINAL_DIR"
    exit 1
fi

if [ ! -d "$COT_DIR" ]; then
    echo "⚠️  Warning: CoT directory not found: $COT_DIR"
    echo "Please update COT_DIR in this script to point to your CoT responses directory"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==================== Process Datasets ====================
TOTAL_DATASETS=${#DATASETS[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0

for dataset_config in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name original_file cot_file <<< "$dataset_config"

    echo ""
    echo "----------------------------------------"
    echo "Processing: $dataset_name"
    echo "----------------------------------------"

    ORIGINAL_PATH="$ORIGINAL_DIR/$original_file"
    COT_PATH="$COT_DIR/$cot_file"
    OUTPUT_PATH="$OUTPUT_DIR/${dataset_name}_llamafactory.json"

    # Check if files exist
    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo "⚠️  Skipping: Original file not found: $ORIGINAL_PATH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    if [ ! -f "$COT_PATH" ]; then
        echo "⚠️  Skipping: CoT file not found: $COT_PATH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    echo "  Original: $ORIGINAL_PATH"
    echo "  CoT:      $COT_PATH"
    echo "  Output:   $OUTPUT_PATH"
    echo ""

    # Run conversion
    python "$SCRIPT_DIR/convert_visual_demo.py" \
        --original-data "$ORIGINAL_PATH" \
        --cot-responses "$COT_PATH" \
        --output-file "$OUTPUT_PATH" \
        --filter-success \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✅ Successfully converted: $dataset_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Run validation
        echo ""
        echo "Validating output..."
        python "$SCRIPT_DIR/validate_output.py" \
            --input-file "$OUTPUT_PATH" \
            --show-samples 0

        if [ $? -eq 0 ]; then
            echo "✅ Validation passed"
        else
            echo "⚠️  Validation found issues (but conversion succeeded)"
        fi
    else
        echo "❌ Failed to convert: $dataset_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ==================== Summary ====================
echo ""
echo "========================================"
echo "Batch Conversion Summary"
echo "========================================"
echo "Total datasets: $TOTAL_DATASETS"
echo "Successful:     $SUCCESS_COUNT"
echo "Failed:         $FAILED_COUNT"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Output files saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Register datasets in LLaMA-Factory/data/dataset_info.json"
    echo "  2. Update training config to use: dataset: visual_h5_agilex_3rgb_llamafactory,..."
    echo "  3. Run training: bash LLaMA-Factory/our_scripts/train_qwen2_5vl_lora_sft.sh"
fi

echo "========================================"

if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
