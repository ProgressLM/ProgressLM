# Example Usage Guide

## Quick Start Example

### 1. Prepare a sample dataset

Create `sample_data.jsonl`:

```json
{"id": "sample_1", "visual_demo": ["images/demo1_1.jpg", "images/demo1_2.jpg", "images/demo1_3.jpg"], "stage_to_estimate": ["images/current1.jpg"], "progress_score": 0.33, "data_source": "test"}
{"id": "sample_2", "visual_demo": ["images/demo2_1.jpg", "images/demo2_2.jpg"], "stage_to_estimate": ["images/current2.jpg"], "progress_score": 0.67, "data_source": "test"}
{"id": "sample_3", "visual_demo": ["images/demo3_1.jpg", "images/demo3_2.jpg", "images/demo3_3.jpg", "images/demo3_4.jpg"], "stage_to_estimate": ["images/current3.jpg"], "progress_score": 0.5, "data_source": "test"}
```

### 2. Configure the script

Edit `../scripts/comm_visual_demo.sh`:

```bash
# Change these lines:
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
DATASET_PATH="/path/to/sample_data.jsonl"
IMAGE_ROOT="/path/to/your/images"  # Or set to "" if using absolute paths

GPU_IDS="0,1,2,3"  # Use available GPUs
BATCH_SIZE=8       # Smaller batch for testing
NUM_INFERENCES=4   # Each sample will be processed 4 times
```

### 3. Run the script

```bash
cd /path/to/qwen_rollouts
bash scripts/comm_visual_demo.sh
```

### 4. Expected output during execution

```
======================================================================
Visual Demo Progress Estimation - Batch Inference
======================================================================
Dataset: /path/to/sample_data.jsonl
Output: /path/to/results/visual_demo_results_20250106_143022.jsonl
GPUs: 0,1,2,3
Batch Size per GPU: 8
Inferences per Sample: 4
======================================================================

Loading dataset from /path/to/sample_data.jsonl
Loaded 3 raw samples from /path/to/sample_data.jsonl
Expanded to 12 samples (×4)

Using 4 GPUs: [0, 1, 2, 3]
Total samples (expanded): 12
Original samples: 3
Inferences per sample: 4
Batch size per GPU: 8

GPU 0: processing samples 0-2 (3 samples)
GPU 1: processing samples 3-5 (3 samples)
GPU 2: processing samples 6-8 (3 samples)
GPU 3: processing samples 9-11 (3 samples)

Starting 4 worker processes...
  Started GPU 0 worker (PID: 12345)
  Started GPU 1 worker (PID: 12346)
  Started GPU 2 worker (PID: 12347)
  Started GPU 3 worker (PID: 12348)

Processing 12 samples with 4 GPUs (batch_size=8 per GPU)...

Progress: 12/12 [00:05<00:00, 2.3it/s, MeanScore=0.857, ErrorRate=0.0%]

Waiting for all workers to finish...
Collecting results from all GPUs...
  Received 3 results from GPU 0
  Received 3 results from GPU 1
  Received 3 results from GPU 2
  Received 3 results from GPU 3

Writing 12 results to /path/to/results/visual_demo_results_20250106_143022.jsonl...

======================================================================
VISUAL DEMO PROGRESS ESTIMATION SUMMARY
======================================================================
Total samples (expanded): 12
Original samples: 3
Inferences per sample: 4
Processed: 12
Errors: 0 (0.00%)
Mean evaluation score (all): 0.8574
Mean evaluation score (valid only): 0.8574
Results saved to: /path/to/results/visual_demo_results_20250106_143022.jsonl
======================================================================
Summary saved to: /path/to/results/visual_demo_results_20250106_143022_summary.json

======================================================================
✓ Completed | Results: /path/to/results/visual_demo_results_20250106_143022.jsonl
======================================================================
```

### 5. Inspect the results

```bash
# View first few results
head -n 4 /path/to/results/visual_demo_results_20250106_143022.jsonl
```

Output (4 lines for `sample_1`, each is one inference):

```json
{"id": "sample_1", "ground_truth": 0.33, "predicted_score": 0.35, "evaluation_score": 0.98, "error": false, "model_outputs": "<ref_think>The current state matches frame 2...</ref_think><ref>0.33</ref><score_think>Slightly more progress...</score_think><score>0.35</score>"}
{"id": "sample_1", "ground_truth": 0.33, "predicted_score": 0.32, "evaluation_score": 0.99, "error": false, "model_outputs": "<ref_think>Closest to frame 2...</ref_think><ref>0.33</ref><score_think>Very similar...</score_think><score>0.32</score>"}
{"id": "sample_1", "ground_truth": 0.33, "predicted_score": 0.38, "evaluation_score": 0.95, "error": false, "model_outputs": "<ref_think>Between frame 2 and 3...</ref_think><ref>0.4</ref><score_think>Shows more progress...</score_think><score>0.38</score>"}
{"id": "sample_1", "ground_truth": 0.33, "predicted_score": 0.31, "evaluation_score": 0.98, "error": false, "model_outputs": "<ref_think>Matches frame 2...</ref_think><ref>0.33</ref><score_think>Almost identical...</score_think><score>0.31</score>"}
```

### 6. View summary statistics

```bash
cat /path/to/results/visual_demo_results_20250106_143022_summary.json
```

```json
{
  "total_samples_expanded": 12,
  "original_samples": 3,
  "num_inferences_per_sample": 4,
  "processed": 12,
  "errors": 0,
  "error_rate": 0.0,
  "mean_evaluation_score_all": 0.8574,
  "mean_evaluation_score_valid": 0.8574,
  "batch_size": 8,
  "num_gpus": 4,
  "dataset_path": "/path/to/sample_data.jsonl",
  "model_path": "/path/to/Qwen2.5-VL-3B-Instruct",
  "output_file": "/path/to/results/visual_demo_results_20250106_143022.jsonl"
}
```

## Advanced Examples

### Example 1: Single GPU with larger batch size

```bash
export CUDA_VISIBLE_DEVICES=0

python frm/run_visual_demo.py \
    --model-path /path/to/model \
    --dataset-path data.jsonl \
    --output-file results.jsonl \
    --batch-size 32 \
    --num-inferences 5 \
    --temperature 0.8
```

### Example 2: Test with limited samples

```bash
# Process only first 10 samples (before expansion)
# With num_inferences=4, this will process 40 total inferences

python frm/run_visual_demo.py \
    --model-path /path/to/model \
    --dataset-path data.jsonl \
    --output-file test_results.jsonl \
    --limit 40 \
    --batch-size 8 \
    --num-inferences 4
```

### Example 3: High diversity sampling

```bash
# Use higher temperature for more diverse predictions
python frm/run_visual_demo.py \
    --model-path /path/to/model \
    --dataset-path data.jsonl \
    --output-file diverse_results.jsonl \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 100 \
    --num-inferences 10
```

## Data Analysis Examples

### Calculate per-sample statistics

```python
import json
from collections import defaultdict

# Load results
results = []
with open('results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

# Group by sample ID
samples = defaultdict(list)
for r in results:
    samples[r['id']].append(r)

# Calculate statistics per sample
for sample_id, inferences in samples.items():
    scores = [inf['evaluation_score'] for inf in inferences if not inf['error']]
    predictions = [inf['predicted_score'] for inf in inferences if not inf['error']]

    print(f"Sample: {sample_id}")
    print(f"  Ground truth: {inferences[0]['ground_truth']:.2f}")
    print(f"  Mean prediction: {sum(predictions)/len(predictions):.2f}")
    print(f"  Std prediction: {np.std(predictions):.3f}")
    print(f"  Mean eval score: {sum(scores)/len(scores):.3f}")
    print(f"  Error rate: {sum(1 for inf in inferences if inf['error']) / len(inferences):.1%}")
    print()
```

### Filter best predictions

```bash
# Keep only the best prediction per sample
jq -s 'group_by(.id) | map(max_by(.evaluation_score))' results.jsonl > best_results.json
```

### Calculate overall metrics

```bash
# Mean evaluation score
jq -s 'map(.evaluation_score) | add / length' results.jsonl

# Error rate
jq -s 'map(select(.error == true)) | length' results.jsonl
```

## Troubleshooting Examples

### Handle missing images gracefully

The system automatically skips samples with missing images and logs errors:

```json
{"id": "sample_X", "ground_truth": 0.5, "predicted_score": null, "evaluation_score": 0.0, "error": true, "error_message": "visual_demo image not found: /path/to/missing.jpg", "model_outputs": ""}
```

### Check parsing errors

```bash
# Find samples with parsing errors
jq 'select(.error == true and .error_message | contains("parse"))' results.jsonl
```

### Resume interrupted runs

The system saves intermediate results every 1000 samples (configurable). To resume:

1. Check the last saved file: `results_gpu0.jsonl`, `results_gpu1.jsonl`, etc.
2. Note how many samples were processed
3. Adjust `--limit` to skip processed samples (not recommended) or rerun entirely

## Performance Benchmarks

Typical performance on A100 GPUs:

- **Single GPU**: ~15-20 samples/sec (batch_size=16, avg 3 demo images)
- **4 GPUs**: ~60-80 samples/sec (linear scaling)
- **8 GPUs**: ~120-150 samples/sec

Bottlenecks:
- Image loading (use SSD storage)
- Long visual_demo sequences (reduce batch_size)
- Response parsing (optimize regex)

## Best Practices

1. **Start small**: Test with `--limit 40` (10 samples × 4 inferences) first
2. **Monitor VRAM**: Adjust batch_size based on `nvidia-smi`
3. **Save frequently**: Set `--save-interval` to 100-500 for long runs
4. **Validate data**: Check that all images exist before running
5. **Use absolute paths**: Or set `--image-root` correctly
6. **Temperature tuning**: 0.7 for balance, 0.3 for consistency, 1.0 for diversity
