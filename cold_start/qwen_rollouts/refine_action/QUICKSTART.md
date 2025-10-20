# 快速开始指南 - Text Cleaning System

## 🚀 5分钟上手

### 步骤 1: 验证环境

确保你有：
- ✅ Python 3.8+
- ✅ PyTorch with CUDA
- ✅ transformers 库
- ✅ Qwen2-VL 模型
- ✅ 至少 1 个可用 GPU

运行测试验证模块：
```bash
cd /Users/cxqian/Codes/WorldVLM/data_preprocess/qwen_rollouts/refine_action
python test_modules.py
```

如果看到 "🎉 ALL TESTS PASSED! 🎉"，说明环境正常。

---

### 步骤 2: 准备数据

创建你的数据集文件（JSONL格式），例如 `my_data.jsonl`：

```json
{"id": "sample_001", "text_demo": "Step 1: First action\nBy now, our progress is 0.5.\n\nStep 2: Second action\nBy now, our progress is 1.0.", "total_steps": "2"}
{"id": "sample_002", "text_demo": "Step 1: Only step\nBy now, our progress is 1.0.", "total_steps": 1}
```

**必需字段**：
- `id` - 样本唯一标识
- `text_demo` - 需要清理的文本
- `total_steps` - 总步骤数

---

### 步骤 3: 配置脚本

编辑 `scripts/clean_text_comm.sh`，修改以下路径：

```bash
# 你的模型路径
MODEL_PATH="/path/to/your/Qwen2.5-VL-3B-Instruct"

# 你的数据集路径
DATASET_PATH="/path/to/your/my_data.jsonl"

# 输出目录
OUTPUT_DIR="/path/to/your/output"

# 使用的GPU（根据你的硬件调整）
GPU_IDS="0,1,2,3"  # 例如使用4个GPU

# 批量大小（根据GPU内存调整）
BATCH_SIZE=32
```

**关键参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `GPU_IDS` | 使用的GPU列表 | 根据可用GPU设置 |
| `BATCH_SIZE` | 每个GPU的批量大小 | 32（3B模型）<br>16（7B模型）<br>8（更大模型） |
| `TEMPERATURE` | 采样温度 | 0.3（更一致）<br>0.7（更多样） |
| `NUM_INFERENCES` | 每个样本推理次数 | 1（快速）<br>3-5（更好质量） |

---

### 步骤 4: 运行推理

```bash
cd /Users/cxqian/Codes/WorldVLM/data_preprocess/qwen_rollouts
bash scripts/clean_text_comm.sh
```

你会看到实时进度：
```
Processing 1000 samples with 4 GPUs (batch_size=32 per GPU)...

Progress: |████████████████████| 1000/1000 [02:34<00:00, 6.48it/s, FormatErrors=2.3%]
```

---

### 步骤 5: 检查结果

完成后，查看输出目录：

```bash
# 主要结果文件
cat /path/to/output/cleaned_text_TIMESTAMP.jsonl

# 统计摘要
cat /path/to/output/cleaned_text_TIMESTAMP_summary.json

# 完整日志
less /path/to/output/clean_text_TIMESTAMP.log
```

**结果格式**：
```json
{
  "id": "sample_001",
  "new_text_demo": "清理后的文本...",
  "error": false,
  "format_error": false
}
```

---

## 📊 结果分析

### 查看统计信息

```bash
# 总样本数
wc -l cleaned_text_TIMESTAMP.jsonl

# 格式错误数
grep '"format_error": *true' cleaned_text_TIMESTAMP.jsonl | wc -l

# 处理错误数
grep '"error": *true' cleaned_text_TIMESTAMP.jsonl | wc -l

# 成功样本数（无格式错误）
grep '"format_error": *false' cleaned_text_TIMESTAMP.jsonl | wc -l
```

### 提取特定样本

```bash
# 查看某个样本的结果
grep '"id": "sample_001"' cleaned_text_TIMESTAMP.jsonl | jq .

# 提取所有格式错误的样本
grep '"format_error": *true' cleaned_text_TIMESTAMP.jsonl > format_errors.jsonl

# 提取所有成功的样本
grep '"format_error": *false' cleaned_text_TIMESTAMP.jsonl > valid_samples.jsonl
```

---

## 🔧 常见问题

### Q1: GPU内存不足 (CUDA out of memory)

**解决方案**：
```bash
# 减小批量大小
BATCH_SIZE=16  # 或更小 (8, 4)

# 减小最大token数
MAX_NEW_TOKENS=1024  # 从2048降到1024
```

---

### Q2: 格式错误率高 (>10%)

**解决方案**：
```bash
# 降低温度（更确定性）
TEMPERATURE=0.1

# 或增加推理次数，多次采样选最佳
NUM_INFERENCES=3
```

---

### Q3: 处理速度慢

**优化方案**：
```bash
# 增加批量大小（如果GPU内存足够）
BATCH_SIZE=64

# 增加GPU数量
GPU_IDS="0,1,2,3,4,5,6,7"

# 减少最大token数（如果文本较短）
MAX_NEW_TOKENS=1024
```

---

### Q4: 进程卡住不动

**排查步骤**：
1. 检查日志文件：`tail -f clean_text_TIMESTAMP.log`
2. 检查GPU状态：`nvidia-smi`
3. 使用 Ctrl+C 中断，系统会保存部分结果

---

## 💡 最佳实践

### 1. 小批量测试
先用小数据集测试（LIMIT=10）：
```bash
LIMIT=10  # 仅处理10个样本
```

### 2. 检查输出质量
查看几个样本的输出：
```bash
head -n 5 cleaned_text_TIMESTAMP.jsonl | jq .
```

### 3. 调整参数
根据结果调整温度和批量大小。

### 4. 监控资源
使用 `nvidia-smi` 或 `watch -n 1 nvidia-smi` 监控GPU使用情况。

### 5. 保留中间文件
不要删除 `*_gpuX.jsonl` 文件，它们可以用于调试。

---

## 📖 更多信息

- 详细文档：[README.md](README.md)
- 模块测试：`python test_modules.py`
- 格式验证逻辑：见 `text_format_validator.py`

---

## 🎯 下一步

完成清理后，你可以：
1. 使用清理后的文本训练模型
2. 分析格式错误的样本，改进提示词
3. 对格式错误的样本重新处理

**祝你使用愉快！** 🚀
