# Text Cleaning System - 文本清理系统

基于 Qwen2-VL 模型的多GPU并行文本清理和格式验证系统。

## 📁 文件结构

```
refine_action/
├── clean_text_dataset.py        # 数据集加载模块
├── clean_text_prompt.py         # 提示词构建模块
├── text_format_validator.py     # 格式验证模块
├── run_clean_text.py            # 主推理脚本
└── README.md                    # 使用文档

scripts/
└── clean_text_comm.sh           # 执行脚本
```

## 🚀 快速开始

### 1. 准备数据集

数据集格式（JSONL）：
```json
{
  "id": "WikiHow_40810_1",
  "text_demo": "Back Up Messages in the Future...\n\nStep 1: Click...\nBy now, our progress is 0.12.\n\nStep 2: ...",
  "total_steps": "8"
}
```

**必需字段**：
- `id`: 样本唯一标识
- `text_demo`: 原始文本演示内容
- `total_steps`: 总步骤数（字符串或整数）

**可选字段**：
- `stage_to_estimate`, `progress_score`, `data_source` 等（会被保留但不使用）

### 2. 配置执行脚本

编辑 `scripts/clean_text_comm.sh`：

```bash
# 模型路径
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"

# 数据集路径
DATASET_PATH="/path/to/your/dataset.jsonl"

# 输出目录
OUTPUT_DIR="/path/to/output/directory"

# GPU配置
GPU_IDS="0,1,2,3,4,5,6,7"  # 使用的GPU列表
BATCH_SIZE=32              # 每个GPU的批量大小

# 推理配置
NUM_INFERENCES=1           # 每个样本的推理次数
TEMPERATURE=0.3            # 采样温度（0.3较低，更一致）
MAX_NEW_TOKENS=2048        # 最大生成token数

# 处理参数
LIMIT=-1                   # 限制处理样本数（-1表示全部）
```

### 3. 执行推理

```bash
cd /Users/cxqian/Codes/WorldVLM/data_preprocess/qwen_rollouts
bash scripts/clean_text_comm.sh
```

## 📊 输出格式

### 主要输出文件

**cleaned_text_TIMESTAMP.jsonl** - 清理后的结果：
```json
{
  "id": "WikiHow_40810_1",
  "new_text_demo": "Back Up Messages in the Future...",
  "error": false,
  "format_error": false
}
```

**输出字段说明**：
- `id`: 样本ID
- `new_text_demo`: 模型输出的清理后文本
- `error`: 推理过程是否出错（True/False）
- `format_error`: 格式验证是否失败（True/False）
- `error_message`: 错误信息（仅在出错时存在）
- `format_errors`: 格式错误详情（仅在verbose模式下存在）

### 辅助输出文件

- **cleaned_text_TIMESTAMP_summary.json** - 统计摘要
- **cleaned_text_TIMESTAMP_gpuX.jsonl** - 各GPU的中间结果
- **clean_text_TIMESTAMP.log** - 完整日志

## 🔍 格式验证规则

系统会自动验证输出文本是否符合以下规范：

1. ✅ `total_steps` 为正整数
2. ✅ 包含所有步骤（Step 1 到 Step N），不跳步、不重复
3. ✅ 每个步骤后有正确的进度标记：`By now, our progress is X`
   - X = step_number / total_steps
   - 支持多种浮点数格式：0.2, 0.20, 1.0, 1 等
4. ✅ 不存在多余步骤（Step N+1）

## 🛠️ 高级用法

### 直接使用 Python 脚本

```bash
cd refine_action

python run_clean_text.py \
    --model-path /path/to/model \
    --dataset-path /path/to/dataset.jsonl \
    --output-file /path/to/output.jsonl \
    --batch-size 32 \
    --num-inferences 1 \
    --temperature 0.3 \
    --top-p 0.9 \
    --top-k 50 \
    --max-new-tokens 2048 \
    --limit -1
```

### 参数说明

**必需参数**：
- `--model-path`: Qwen2-VL模型路径
- `--dataset-path`: 输入数据集路径
- `--output-file`: 输出文件路径

**可选参数**：
- `--batch-size`: 每个GPU的批量大小（默认：16）
- `--num-inferences`: 每个样本的推理次数（默认：1）
- `--limit`: 限制处理样本数（默认：-1，处理全部）
- `--temperature`: 采样温度（默认：0.3）
- `--top-p`: Top-p采样参数（默认：0.9）
- `--top-k`: Top-k采样参数（默认：50）
- `--max-new-tokens`: 最大生成token数（默认：2048）
- `--verbose`: 打印详细输出

## 📈 性能估算

假设配置：
- **8个GPU**
- **每个GPU batch_size=32**
- **1000个样本**
- **每个样本平均300 tokens输出**

预计性能：
- 并行批次数：1000 / (8 × 32) ≈ 4 批次
- 每批次时间：~30秒（取决于模型和硬件）
- **总时间：~2-3分钟**

## 🔧 故障排查

### 问题：GPU内存不足

**解决方案**：
1. 减小 `BATCH_SIZE`
2. 减小 `MAX_NEW_TOKENS`
3. 使用更小的模型

### 问题：格式错误率高

**解决方案**：
1. 降低 `TEMPERATURE`（更确定性的输出）
2. 调整提示词（编辑 `clean_text_prompt.py`）
3. 增加 `NUM_INFERENCES`，选择最佳结果

### 问题：进程卡住

**解决方案**：
1. 检查GPU是否正常工作
2. 查看日志文件：`cat clean_text_TIMESTAMP.log`
3. 使用 Ctrl+C 中断，系统会保存部分结果到 `*_partial.jsonl`

## 📝 示例统计输出

```
======================================================================
TEXT CLEANING SUMMARY
======================================================================
Total samples (expanded): 1000
Original samples: 1000
Inferences per sample: 1
Processed: 1000
Processing errors: 5 (0.50%)
Format errors: 23 (2.30%)
Valid samples: 977 (97.70%)
Results saved to: /path/to/output.jsonl
======================================================================
```

## 🔗 相关模块

- **数据加载**: `clean_text_dataset.py`
- **提示词构建**: `clean_text_prompt.py`
- **格式验证**: `text_format_validator.py`
- **主推理**: `run_clean_text.py`
- **模型接口**: `../qwen2_vl/model.py`

## 📄 许可证

本项目遵循 WorldVLM 项目的许可证。
