#!/bin/bash
################################################################################
# VLAC Multi-GPU Evaluation Example Script
#
# This is an example configuration for multi-GPU data parallel inference.
# Copy this file and modify the paths for your use case.
#
# Usage:
#   bash scripts/eval_multi_gpu_example.sh
################################################################################

# ============================================================================
# Multi-GPU Configuration Example
# ============================================================================

# ---------- 模型配置 ----------
MODEL_PATH="/path/to/your/VLAC/model"
MODEL_TYPE="internvl2"
TEMPERATURE=0.5
TOP_K=1

# ---------- 多GPU配置 ----------
# 使用4个GPU进行数据并行推理
NUM_GPUS=4

# 每个GPU的批量大小（根据显存调整）
# 总批量 = NUM_GPUS × BATCH_NUM
BATCH_NUM=5

# ---------- 数据配置 ----------
DATA_DIR="/path/to/your/test/images"
REF_DIR="/path/to/your/reference/images"  # 可选
TASK_DESCRIPTION="Pick up the bowl and place it in the box"
IMAGE_PATTERN="*.jpg,*.png"
MAX_IMAGES=""  # 留空加载全部

# ---------- 评估参数配置 ----------
REF_NUM=6
SKIP=1
RICH_MODE=true
THINK_MODE=false
REVERSE_EVAL=false

# ---------- 输出配置 ----------
OUTPUT_DIR="./results_multi_gpu"
OUTPUT_NAME=""  # 留空自动生成

# ============================================================================
# 执行脚本（一般不需要修改）
# ============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}VLAC Multi-GPU Evaluation${NC}"
echo -e "${BLUE}=================================${NC}"

# 检查CUDA可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}[ERROR] nvidia-smi not found. CUDA may not be installed.${NC}"
    exit 1
fi

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}[INFO] Detected $AVAILABLE_GPUS GPUs${NC}"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo -e "${YELLOW}[WARNING] Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available.${NC}"
    echo -e "${YELLOW}[WARNING] Will use $AVAILABLE_GPUS GPUs.${NC}"
fi

# 检查必需参数
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}[ERROR] Model path does not exist: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}[ERROR] Data directory does not exist: $DATA_DIR${NC}"
    exit 1
fi

# 构建命令行参数
CMD="python $PROJECT_ROOT/run_eval.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --task \"$TASK_DESCRIPTION\" \
    --model_type $MODEL_TYPE \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --batch_num $BATCH_NUM \
    --ref_num $REF_NUM \
    --skip $SKIP \
    --num_gpus $NUM_GPUS \
    --image_pattern \"$IMAGE_PATTERN\" \
    --output_dir $OUTPUT_DIR"

# 添加可选参数
if [ -n "$REF_DIR" ] && [ -d "$REF_DIR" ]; then
    CMD="$CMD --ref_dir $REF_DIR"
fi

if [ -n "$MAX_IMAGES" ]; then
    CMD="$CMD --max_images $MAX_IMAGES"
fi

if [ -n "$OUTPUT_NAME" ]; then
    CMD="$CMD --output_name $OUTPUT_NAME"
fi

if [ "$RICH_MODE" = true ]; then
    CMD="$CMD --rich"
fi

if [ "$THINK_MODE" = true ]; then
    CMD="$CMD --think"
fi

if [ "$REVERSE_EVAL" = true ]; then
    CMD="$CMD --reverse_eval"
fi

# 显示配置信息
echo -e "${GREEN}[INFO] Multi-GPU Configuration:${NC}"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Batch Size per GPU: $BATCH_NUM"
echo "  Total Parallel Batch: $((NUM_GPUS * BATCH_NUM))"
echo "  Model Path: $MODEL_PATH"
echo "  Data Dir: $DATA_DIR"
if [ -n "$REF_DIR" ] && [ -d "$REF_DIR" ]; then
    echo "  Reference Dir: $REF_DIR"
fi
echo "  Task: $TASK_DESCRIPTION"
echo "  Skip: $SKIP"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# 显示GPU信息
echo -e "${GREEN}[INFO] Available GPUs:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s (Total: %s, Free: %s)\n", $1, $2, $3, $4}'
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行评估
echo -e "${GREEN}[INFO] Starting multi-GPU evaluation...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

eval $CMD

EVAL_EXIT_CODE=$?

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# 检查执行结果
echo ""
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}Multi-GPU Evaluation Completed!${NC}"
    echo -e "${GREEN}Total Time: ${ELAPSED_TIME}s${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}=================================${NC}"

    # 显示性能信息
    echo ""
    echo -e "${BLUE}Performance Summary:${NC}"
    echo "  Total Execution Time: ${ELAPSED_TIME}s"
    echo "  GPUs Used: $NUM_GPUS"
    echo "  Batch Size per GPU: $BATCH_NUM"

    # 估算单GPU时间（假设线性加速）
    ESTIMATED_SINGLE_GPU_TIME=$((ELAPSED_TIME * NUM_GPUS * 9 / 10))
    SPEEDUP=$(echo "scale=2; $ESTIMATED_SINGLE_GPU_TIME / $ELAPSED_TIME" | bc)
    echo "  Estimated Speedup: ${SPEEDUP}x"
    echo ""
else
    echo -e "${RED}=================================${NC}"
    echo -e "${RED}Multi-GPU Evaluation Failed!${NC}"
    echo -e "${RED}Exit Code: $EVAL_EXIT_CODE${NC}"
    echo -e "${RED}=================================${NC}"
    exit 1
fi
