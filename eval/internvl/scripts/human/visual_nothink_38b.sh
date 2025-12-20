#!/bin/bash
#####################################################################
# Human Activities Visual Demo - InternVL 38B (NoThink Mode)
#####################################################################

MODEL_PATH="/projects/p32958/jianshu/weight/OpenGVLab/InternVL3_5-38B"
DATASET_PATH="/projects/p32958/chengxuan/ProgressLM/data/benchmark/human/jsonl/visual_demo_human_activities.jsonl"
IMAGE_ROOT="/projects/p32958/chengxuan/data/images"

BASE_OUTPUT_DIR="/projects/p32958/chengxuan/results/internvl/human/visual_nothink_38b"
PROJECT_NAME="internvl_38B_nothink"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${PROJECT_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_FILE="${OUTPUT_DIR}/run.log"

GPU_IDS="0,1,2,3"
BATCH_SIZE=1
NUM_INFERENCES=1
TEMPERATURE=0.6
TOP_P=0.9
MAX_NEW_TOKENS=512
MAX_NUM_TILES=4
INPUT_SIZE=448
LIMIT=-1
VERBOSE=false

echo "======================================================================"
echo "Human Activities Visual Demo - InternVL 38B (NoThink Mode)"
echo "======================================================================"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $GPU_IDS"
echo "======================================================================"

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNVL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CODES_DIR="$INTERNVL_DIR/codes"

cd "$CODES_DIR" || exit 1

CMD="python run_visual_demo_nothink_single.py \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --output-file $OUTPUT_FILE \
    --batch-size $BATCH_SIZE \
    --num-inferences $NUM_INFERENCES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-new-tokens $MAX_NEW_TOKENS \
    --max-num-tiles $MAX_NUM_TILES \
    --input-size $INPUT_SIZE"

if [ -n "$IMAGE_ROOT" ]; then
    CMD="$CMD --image-root $IMAGE_ROOT"
fi

if [ $LIMIT -gt 0 ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

$CMD 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo " Completed | Results: $OUTPUT_FILE"
    SUMMARY_FILE="${OUTPUT_FILE%.jsonl}_summary.json"
    [ -f "$SUMMARY_FILE" ] && cat "$SUMMARY_FILE"
else
    echo " Failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
