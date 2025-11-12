# Progress Estimation Evaluation Module

完整的进度估计评估工具，支持 N/A 值处理、False Positive 检测和轨迹顺序一致性（VOC）计算。

## 功能特性

### ✅ 核心评估指标

1. **误差计算**
   - 相对误差（Score Error）: `|GT - Pred| / GT`
   - 绝对误差（Ref Error）: `|GT - Pred|`
   - 仅在 GT 和预测都是数值时计算

2. **False Positive 检测**
   - Ref False Positive: GT 和预测的 n/a 状态不匹配
   - Score False Positive: GT 和预测的 n/a 状态不匹配
   - 示例：GT 是数值但预测是 n/a，或 GT 是 n/a 但预测是数值

3. **VOC (Visual/Trajectory Order Consistency)**
   - 使用 Spearman 相关性评估轨迹排序一致性
   - 按轨迹 ID 分组计算
   - 仅在 GT 为数值的轨迹上计算

4. **GT 分布统计**
   - 数值 GT 样本数量
   - N/A GT 样本数量
   - 分别统计 ref 和 score

## 安装依赖

```bash
pip install numpy scipy
```

## 使用方法

### 1. 命令行使用

#### 评估单个文件

```bash
# 基础评估
python progress_evaluation.py results.jsonl

# 保存摘要到指定文件
python progress_evaluation.py results.jsonl --output summary.json
```

#### 比较多个模型

```bash
python progress_evaluation.py --compare \
    baseline:results_baseline.jsonl \
    sft_3b:results_sft_3b.jsonl \
    sft_7b:results_sft_7b.jsonl \
    --output comparison.json
```

### 2. Python API 使用

#### 基础评估

```python
from progress_evaluation import generate_summary_report

# 评估单个文件
stats = generate_summary_report('results.jsonl')

print(f"Score Error Mean: {stats['score_error_mean']:.4f}")
print(f"VOC Mean: {stats['voc_mean']:.4f}")
print(f"Score FP Rate: {stats['score_fp_rate']*100:.2f}%")
```

#### 加载和分析

```python
from progress_evaluation import load_results, analyze_results

# 加载结果
results = load_results('results.jsonl')

# 分析（带详细输出）
stats = analyze_results(results, verbose=True)

# 分析（静默模式）
stats = analyze_results(results, verbose=False)
```

#### 模型比较

```python
from progress_evaluation import compare_models

# 比较多个模型
result_files = {
    'Baseline': 'results_baseline.jsonl',
    'SFT-3B': 'results_sft_3b.jsonl',
    'SFT-7B': 'results_sft_7b.jsonl'
}

comparison = compare_models(result_files, output_file='comparison.json')

# 访问特定模型的统计
baseline_stats = comparison['Baseline']
print(f"Baseline VOC: {baseline_stats['voc_mean']:.4f}")
```

#### 使用核心函数

```python
from progress_evaluation import (
    calculate_false_positives,
    calculate_evaluation_score,
    calculate_ref_error,
    calculate_voc_metrics
)

# 计算 False Positive
ref_fp, score_fp = calculate_false_positives(
    predicted_ref=5,        # 或 "n/a"
    predicted_score=0.33,   # 或 "n/a"
    gt_ref=3,              # 或 None
    gt_score=0.30          # 或 None
)

# 计算评估误差
score_error = calculate_evaluation_score(
    predicted=0.33,
    ground_truth=0.30
)

# 计算参考误差
ref_error = calculate_ref_error(
    predicted_ref=5,
    ground_truth_ref=3
)

# 计算 VOC
voc_metrics = calculate_voc_metrics(results)
print(f"VOC Mean: {voc_metrics['voc_mean']:.4f}")
print(f"VOC Std: {voc_metrics['voc_std']:.4f}")
print(f"Valid Trajectories: {voc_metrics['voc_count']}")
```

## 输出格式

### 评估统计字典

```python
{
    # 基础统计
    'total_samples': 1000,
    'valid_samples': 950,
    'error_samples': 50,
    'error_rate': 0.05,

    # Score 误差统计
    'score_error_mean': 0.1234,
    'score_error_median': 0.0987,
    'score_error_std': 0.0543,
    'score_error_count': 900,

    # Ref 误差统计
    'ref_error_mean': 1.5678,
    'ref_error_median': 1.0000,
    'ref_error_std': 0.8765,
    'ref_error_count': 920,

    # False Positive 统计
    'ref_fp_count': 45,
    'ref_fp_rate': 0.045,
    'score_fp_count': 38,
    'score_fp_rate': 0.038,

    # VOC 统计
    'voc_mean': 0.8765,
    'voc_std': 0.1234,
    'voc_median': 0.9012,
    'voc_count': 120,

    # GT 分布
    'gt_numeric_count': 850,
    'gt_na_count': 150,
    'gt_ref_numeric': 920,
    'gt_ref_na': 80,
    'gt_score_numeric': 900,
    'gt_score_na': 100
}
```

### 命令行输出示例

```
================================================================================
PROGRESS ESTIMATION EVALUATION REPORT
================================================================================

📊 Basic Statistics:
  Total samples:     1000
  Valid samples:     950
  Error samples:     50 (5.00%)

📈 Score Error Metrics:
  Mean error:        0.1234
  Median error:      0.0987
  Std error:         0.0543
  Valid samples:     900/1000

📍 Ref Error Metrics:
  Mean error:        1.5678
  Median error:      1.0000
  Std error:         0.8765
  Valid samples:     920/1000

⚠️  False Positive Rates:
  Ref FP rate:       4.50% (45/1000)
  Score FP rate:     3.80% (38/1000)

🔄 VOC (Trajectory Order Consistency):
  Mean VOC:          0.8765
  Median VOC:        0.9012
  Std VOC:           0.1234
  Valid trajectories: 120

📋 Ground Truth Distribution:
  Both numeric:      850 (85.0%)
  Contains N/A:      150 (15.0%)
    - Ref numeric:   920
    - Ref N/A:       80
    - Score numeric: 900
    - Score N/A:     100
================================================================================
```

## 数据格式要求

### 输入 JSONL 格式

每行一个 JSON 对象，包含以下字段：

```json
{
  "ref": "3",           // 或 "n/a"
  "score": "33%",       // 或 "n/a" 或 "0.33"
  "closest_idx": 3,     // GT ref (1-based)
  "ground_truth_score": "33%",
  "ref_score": 0.1234,  // 可选：预计算的 ref 误差
  "pred_score": 0.0987, // 可选：预计算的 score 误差
  "ref_false_positive": false,
  "score_false_positive": false,
  "response": "...",    // 原始模型输出
  "meta_data": {
    "id": "trajectory_001",  // 轨迹 ID
    "closest_idx": 3,        // GT ref
    "progress_score": 0.33,  // GT score (0-1)，或 null 表示 n/a
    "status": "success"      // 或 "failed"
  }
}
```

### N/A 值表示

- **GT 中的 N/A**: `meta_data` 中的 `closest_idx` 或 `progress_score` 为 `null`
- **预测中的 N/A**: `ref` 或 `score` 字段为字符串 `"n/a"`

## 评估逻辑说明

### 1. False Positive 定义

False Positive 发生在以下情况：
- **GT 是数值 + 预测是 n/a**: 模型应该预测数值但预测了 n/a
- **GT 是 n/a + 预测是数值**: 模型应该预测 n/a 但预测了数值

正确的情况：
- **GT 是数值 + 预测是数值**: 使用误差计算
- **GT 是 n/a + 预测是 n/a**: 正确识别 n/a

### 2. 条件性误差计算

仅在以下条件下计算误差：
- **GT 和预测都是数值**: 计算相对误差或绝对误差
- **其他情况**: 返回 `inf`，不计入统计

### 3. VOC 计算流程

1. 按轨迹 ID 分组所有样本
2. **仅保留 GT 为数值的轨迹**
3. 对每个轨迹：
   - 按 GT score 排序得到真实顺序
   - 按预测 score 排序得到预测顺序（n/a → 0.0）
   - 计算 Spearman 相关性
4. 返回所有有效轨迹的 VOC 均值、中位数和标准差

### 4. GT 分布统计

分别统计：
- 两者都是数值的样本数
- 至少一个是 n/a 的样本数
- ref 为数值/n/a 的样本数
- score 为数值/n/a 的样本数

## 常见问题

### Q1: 为什么我的 VOC 是 None？

**A**: VOC 需要满足以下条件：
- 至少有一个轨迹包含 2 个或以上样本
- GT 的 `closest_idx` 和 `progress_score` 都必须是数值
- 轨迹内的 GT score 必须有变化（不能都相同）

### Q2: 误差计算为什么返回 inf？

**A**: 以下情况会返回 inf：
- GT 或预测为 None
- GT 或预测为 "n/a"
- GT 为 0（避免除以零）

### Q3: 如何处理百分比格式？

**A**: 模块自动处理以下格式：
- `"33%"` → 0.33
- `"0.33"` → 0.33
- `33` → 0.33（假设是百分比）
- `0.33` → 0.33

### Q4: False Positive 和错误样本的区别？

**A**:
- **False Positive**: GT 和预测的 n/a 状态不匹配
- **错误样本**: `meta_data.status == "failed"`，通常是解析错误或验证失败

## 与原始 eval_results.py 的区别

| 特性 | 原始版本 | 新版本 (progress_evaluation.py) |
|------|---------|--------------------------------|
| N/A 支持 | ❌ | ✅ 完整支持 |
| False Positive | ❌ | ✅ Ref 和 Score 分别跟踪 |
| VOC 计算 | ⚠️ 简单版本 | ✅ 完整 Spearman 相关性 |
| 条件性误差 | ❌ | ✅ 仅数值对计算 |
| GT 分布统计 | ❌ | ✅ 详细统计 |
| 模型比较 | ❌ | ✅ 内置比较功能 |
| 文档 | ⚠️ 最少 | ✅ 完整文档和类型注解 |

## 贡献和反馈

如有问题或建议，请联系开发团队。

## 版本历史

- **v1.0** (2025-01): 初始版本，完整的 N/A 支持和评估功能
