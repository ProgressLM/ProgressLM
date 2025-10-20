#!/bin/bash

#####################################################################
# Text Cleaning Script - Clean and Polish Text Demonstrations
#
# This script runs text cleaning on text demonstration dataset using
# Qwen2-VL model with distributed GPU support.
#####################################################################

# ======================== Configuration ========================

# Model configuration
MODEL_PATH="/home/runsheng/personal_3/qiancx/Sources/models/Qwen2.5-VL-7B-Instruct"

# Dataset configuration
DATASET_PATH="/home/runsheng/personal_3/qiancx/Process/Data/FRM/annotations/comm_frm_text_cleaned.jsonl"

# Output configuration
OUTPUT_DIR="/home/runsheng/personal_3/qiancx/Sources/results/clean_text"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/cleaned_text_${TIMESTAMP}.jsonl"
LOG_FILE="${OUTPUT_DIR}/clean_text_${TIMESTAMP}.log"

# GPU configuration
GPU_IDS="0,1,2,3,4,5,6,7"  # Comma-separated GPU IDs to use
BATCH_SIZE=32              # Batch size per GPU (can be higher for text-only tasks)

# Inference configuration
NUM_INFERENCES=1           # Number of inferences per sample (default: 1, no replication)

# Model parameters
TEMPERATURE=0.5            # Lower temperature for more consistent cleaning
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=30000        # Enough for full text demonstrations

# Processing parameters
LIMIT=1000                   # Limit samples to process (-1 for all)

# Misc
VERBOSE=false              # Set to true for detailed output

# ======================== Auto Configuration ========================

echo "======================================================================"
echo "Text Cleaning - Batch Inference"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Inferences per Sample: $NUM_INFERENCES"
echo "======================================================================"

# ======================== Validation ========================

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH is not set!"
    echo "Please set DATASET_PATH to your dataset JSONL file."
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ======================== Run Inference ========================

# Set CUDA visible devices to all GPUs
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REFINE_ACTION_DIR="$PROJECT_DIR/refine_action"

# Change to refine_action directory
cd "$REFINE_ACTION_DIR" || exit 1

# Build command
CMD="python run_clean_text.py \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --output-file $OUTPUT_FILE \
    --batch-size $BATCH_SIZE \
    --num-inferences $NUM_INFERENCES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-new-tokens $MAX_NEW_TOKENS"

# Add limit if specified
if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add verbose flag if enabled
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "Starting batch inference..."
echo ""

# Execute command with logging
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo " Completed | Results: $OUTPUT_FILE"
    echo "======================================================================"

    # Display summary if exists
    SUMMARY_FILE="${OUTPUT_FILE%.jsonl}_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        echo "Summary:"
        cat "$SUMMARY_FILE"
        echo ""
    fi

    # Count results
    if [ -f "$OUTPUT_FILE" ]; then
        TOTAL_COUNT=$(wc -l < "$OUTPUT_FILE")
        FORMAT_ERRORS=$(grep -c '"format_error": *true' "$OUTPUT_FILE" || echo "0")
        PROCESSING_ERRORS=$(grep -c '"error": *true' "$OUTPUT_FILE" || echo "0")

        echo ""
        echo "Quick Statistics:"
        echo "  Total samples: $TOTAL_COUNT"
        echo "  Format errors: $FORMAT_ERRORS ($(awk "BEGIN {printf \"%.2f\", $FORMAT_ERRORS/$TOTAL_COUNT*100}")%)"
        echo "  Processing errors: $PROCESSING_ERRORS ($(awk "BEGIN {printf \"%.2f\", $PROCESSING_ERRORS/$TOTAL_COUNT*100}")%)"
        echo ""
    fi
else
    echo ""
    echo "======================================================================"
    echo " Failed (exit code $EXIT_CODE) | Log: $LOG_FILE"
    echo "======================================================================"
    exit $EXIT_CODE
fi
