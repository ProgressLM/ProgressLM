# Visual Demo Progress Estimation

Batch inference system for visual progress estimation using Qwen2.5-VL model.

## Overview

This module performs progress estimation on visual demonstrations by:
1. Loading a dataset with variable-length visual demos and progress labels
2. Expanding each sample N times (default: 4) for multiple inferences
3. Running batch inference across multiple GPUs
4. Parsing model outputs to extract progress scores
5. Calculating evaluation scores: `1 - |predicted - ground_truth|`

## Dataset Format

Input JSONL file, where each line is:

```json
{
  "id": "WikiHow_85639_1",
  "visual_demo": ["img1.jpg", "img2.jpg", "img3.jpg", ...],
  "stage_to_estimate": ["current.jpg"],
  "progress_score": 0.2,
  "data_source": "WikiHow"
}
```

**Required fields:**
- `id`: Unique identifier for the sample
- `visual_demo`: List of image paths (variable length, showing progress stages)
- `stage_to_estimate`: List with 1 image path (current state to estimate)
- `progress_score`: Ground truth progress score (0.0 - 1.0)

**Optional fields:**
- `data_source`: Source of the data

## Output Format

Output JSONL file, where each line is one inference result:

```json
{
  "id": "WikiHow_85639_1",
  "ground_truth": 0.2,
  "predicted_score": 0.25,
  "evaluation_score": 0.95,
  "error": false,
  "model_outputs": "<ref_think>...</ref_think><ref>0.2</ref><score_think>...</score_think><score>0.25</score>"
}
```

**Note:** With `num_inferences=4`, each original sample generates 4 output lines (same `id`, different predictions).

## Files

- **`visual_demo_dataset.py`** - Dataset loading and N-times expansion
- **`visual_demo_prompt.py`** - Three-part prompt construction:
  1. Demo presentation (variable-length images)
  2. Current state (single image)
  3. Task instructions
- **`run_visual_demo.py`** - Main inference script with multi-GPU support
- **`../scripts/comm_visual_demo.sh`** - Execution script

## Usage

### 1. Configure the script

Edit `scripts/comm_visual_demo.sh`:

```bash
# Model and data paths
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
DATASET_PATH="/path/to/visual_demo.jsonl"
IMAGE_ROOT="/path/to/images"  # Optional

# GPU configuration
GPU_IDS="0,1,2,3"
BATCH_SIZE=16

# Inference configuration
NUM_INFERENCES=4  # Each sample -> 4 inferences
SAVE_INTERVAL=1000
```

### 2. Run inference

```bash
cd /path/to/qwen_rollouts
bash scripts/comm_visual_demo.sh
```

### 3. Monitor progress

The script shows real-time progress via tqdm:

```
Progress: 2500/10000 [00:15<00:45, 165.3it/s, MeanScore=0.857, ErrorRate=2.3%]
```

- **MeanScore**: Average evaluation score across all processed samples
- **ErrorRate**: Percentage of samples with parsing/inference errors

### 4. Check results

Results are saved to `OUTPUT_DIR/visual_demo_results_TIMESTAMP.jsonl`:

```bash
# View results
head -n 5 results/visual_demo_results_20250106_123456.jsonl

# View summary
cat results/visual_demo_results_20250106_123456_summary.json
```

## Advanced Usage

### Direct Python execution

```bash
cd frm

python run_visual_demo.py \
    --model-path /path/to/Qwen2.5-VL-3B-Instruct \
    --dataset-path /path/to/data.jsonl \
    --output-file /path/to/output.jsonl \
    --image-root /path/to/images \
    --batch-size 16 \
    --num-inferences 4 \
    --save-interval 1000 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-new-tokens 512
```

### Key parameters

- `--num-inferences N`: Replicate each sample N times (default: 4)
- `--batch-size B`: Batch size per GPU (adjust based on VRAM and image count)
- `--save-interval K`: Save results every K samples (default: 1000)
- `--temperature T`: Sampling temperature (0.7 for diversity, lower for consistency)
- `--limit L`: Process only first L samples (after expansion)

### Environment variables

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Then run
python run_visual_demo.py ...
```

## Model Output Format

The model is prompted to output:

```xml
<ref_think>Reasoning for reference frame selection</ref_think>
<ref>0.2</ref>
<score_think>Reasoning for score estimation</score_think>
<score>0.25</score>
```

The system extracts the `<score>` value and calculates:

```
evaluation_score = 1 - |predicted_score - ground_truth|
```

## Features

✅ Variable-length visual demo support (auto-padding in batch)
✅ Multi-GPU parallel batch inference
✅ Data expansion (N inferences per sample)
✅ Real-time progress monitoring (tqdm with metrics)
✅ Incremental saving (every K samples)
✅ Atomic writes (no file corruption)
✅ Graceful interrupt handling (Ctrl+C)
✅ Comprehensive error tracking
✅ Summary statistics output

## Troubleshooting

### Out of memory

Reduce `BATCH_SIZE` in the script (especially for samples with many visual_demo images).

### Parsing errors

Check model outputs in the result file. Common issues:
- Model didn't generate `<score>` tag
- Score value is not numeric
- Adjust `temperature` or `max_new_tokens`

### Image not found

Ensure `IMAGE_ROOT` is set correctly, or use absolute paths in the dataset.

## Example Workflow

```bash
# 1. Prepare dataset
cat data.jsonl
# {"id": "sample1", "visual_demo": ["1.jpg", "2.jpg"], "stage_to_estimate": ["current.jpg"], "progress_score": 0.5}

# 2. Configure script
vim scripts/comm_visual_demo.sh
# Set DATASET_PATH, MODEL_PATH, GPU_IDS

# 3. Run inference
bash scripts/comm_visual_demo.sh

# 4. Check results
jq '.evaluation_score' output.jsonl | awk '{sum+=$1; count++} END {print "Mean:", sum/count}'
```

## Performance Tips

1. **Batch size**: Larger batches → better GPU utilization, but need more VRAM
2. **Num GPUs**: More GPUs → faster processing (linear speedup)
3. **Save interval**: Larger intervals → less I/O overhead, but lose more on interrupt
4. **Temperature**: Lower (0.1) for consistency, higher (0.9) for diversity

## Architecture

```
Dataset (N samples)
    ↓
Expand ×4 (4N samples)
    ↓
Split across GPUs
    ↓
GPU0: 1000 samples → Batch inference → output_gpu0.jsonl
GPU1: 1000 samples → Batch inference → output_gpu1.jsonl
GPU2: 1000 samples → Batch inference → output_gpu2.jsonl
GPU3: 1000 samples → Batch inference → output_gpu3.jsonl
    ↓
Merge & Sort by ID
    ↓
Final output.jsonl (4N lines, grouped by ID)
```
