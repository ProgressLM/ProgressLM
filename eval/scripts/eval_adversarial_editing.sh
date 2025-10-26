#!/bin/bash

#####################################################################
# Adversarial Image Editing Evaluation Script
#
# This script runs adversarial image editing inference using
# Qwen2.5-VL 32B model with multi-GPU data parallel support.
#
# Expected JSONL format:
# {
#   "id": "h5_tienkung_xsens_1rgb/tool_liftn_box_place/2024-09-28-10-54-08",
#   "task_goal": "placing a tool into a box",
#   "text_demo": ["reach for the tool", "grasp the tool", "lift the tool", ...],
#   "total_steps": 6,
#   "stage_to_estimate": "camera_top_0556.jpg",
#   "closest_idx": 6,
#   "progress_score": "100%",
#   "data_source": "h5_tienkung_xsens_1rgb"
# }
#####################################################################

# ======================== Configuration ========================

# Model configuration
MODEL_PATH="/projects/p32958/chengxuan/models/Qwen2.5-VL-32B-Instruct"  # UPDATE THIS

# Dataset configuration
DATASET_PATH="/path/to/dataset.jsonl"  # UPDATE THIS
IMAGE_ROOT="/path/to/images"  # UPDATE THIS - will be prepended to image paths

# Output configuration
OUTPUT_DIR="/path/to/output"  # UPDATE THIS
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/adversarial_editing_${TIMESTAMP}.jsonl"
LOG_FILE="${OUTPUT_DIR}/adversarial_editing_${TIMESTAMP}.log"

# GPU configuration
GPU_IDS="0,1,2,3"  # Comma-separated GPU IDs to use
BATCH_SIZE=8  # Batch size per GPU (adjust based on GPU memory)

# Inference configuration
NUM_INFERENCES=1  # Number of inferences per sample (for multiple sampling)
LIMIT=-1  # -1 for all samples, or specify a number to limit

# Model parameters
TEMPERATURE=0.7  # Sampling temperature (0.0 = greedy, higher = more random)
TOP_P=0.9  # Top-p (nucleus) sampling
TOP_K=50  # Top-k sampling
MAX_NEW_TOKENS=1024  # Maximum tokens to generate

# Image processing parameters
MIN_PIXELS=$((1280 * 28 * 28))  # Minimum image pixels
MAX_PIXELS=$((5120 * 28 * 28))  # Maximum image pixels

# Misc
VERBOSE=false  # Set to true for detailed output

# ======================== Display Configuration ========================

echo "======================================================================"
echo "Adversarial Image Editing Evaluation"
echo "======================================================================"
echo "Model Path       : $MODEL_PATH"
echo "Dataset Path     : $DATASET_PATH"
echo "Image Root       : $IMAGE_ROOT"
echo "Output File      : $OUTPUT_FILE"
echo "Log File         : $LOG_FILE"
echo "GPU IDs          : $GPU_IDS"
echo "Batch Size/GPU   : $BATCH_SIZE"
echo "Num Inferences   : $NUM_INFERENCES"
echo "Temperature      : $TEMPERATURE"
echo "Max New Tokens   : $MAX_NEW_TOKENS"
if [ $LIMIT -gt 0 ]; then
    echo "Sample Limit     : $LIMIT"
fi
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

# Check if image root exists (if not default placeholder)
if [ -n "$IMAGE_ROOT" ] && [ "$IMAGE_ROOT" != "/path/to/images" ]; then
    if [ ! -d "$IMAGE_ROOT" ]; then
        echo "Error: Image root directory not found: $IMAGE_ROOT"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ======================== Run Inference ========================

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Get the script directory and navigate to eval directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_DIR="$PROJECT_DIR/qwen25vl"

# Change to eval directory
cd "$EVAL_DIR" || exit 1

echo ""
echo "Working directory: $(pwd)"
echo "Starting inference..."
echo ""

# Build command
CMD="python run_adversarial_editing.py \
    --model-path \"$MODEL_PATH\" \
    --dataset-path \"$DATASET_PATH\" \
    --output-file \"$OUTPUT_FILE\" \
    --image-root \"$IMAGE_ROOT\" \
    --batch-size $BATCH_SIZE \
    --num-inferences $NUM_INFERENCES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --max-new-tokens $MAX_NEW_TOKENS \
    --min-pixels $MIN_PIXELS \
    --max-pixels $MAX_PIXELS"

# Add limit if specified
if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Add verbose flag if enabled
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# Execute with logging
echo "Command: $CMD"
echo ""

# Run with both console output and log file
eval $CMD 2>&1 | tee "$LOG_FILE"

# ======================== Post-Processing ========================

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Inference completed successfully!"
    echo "======================================================================"
    echo "Results saved to  : $OUTPUT_FILE"
    echo "Summary saved to  : ${OUTPUT_FILE%.jsonl}_summary.json"
    echo "Log saved to      : $LOG_FILE"

    # Display GPU files
    echo ""
    echo "Per-GPU result files:"
    for gpu_id in ${GPU_IDS//,/ }; do
        GPU_FILE="${OUTPUT_FILE%.jsonl}_gpu${gpu_id}.jsonl"
        if [ -f "$GPU_FILE" ]; then
            COUNT=$(wc -l < "$GPU_FILE")
            echo "  GPU $gpu_id: $GPU_FILE ($COUNT samples)"
        fi
    done

    # Display summary if exists
    SUMMARY_FILE="${OUTPUT_FILE%.jsonl}_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        echo "Summary:"
        cat "$SUMMARY_FILE" | python -m json.tool 2>/dev/null || cat "$SUMMARY_FILE"
    fi

    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "✗ ERROR: Inference failed with exit code $EXIT_CODE"
    echo "======================================================================"
    echo "Check log file for details: $LOG_FILE"
    echo "======================================================================"
    exit $EXIT_CODE
fi
