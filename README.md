<h1 align="center">
  <img src="docs/imgs/icon.png" height="30" />&nbsp;ProgressLM: Towards Progress Reasoning in Vision-Language Models
</h1>

<p align="center">
  <a href="https://sterzhang.github.io/">Jianshu Zhang</a>*, <a href="https://qiancx.com/">Chengxuan Qian</a>*, <a href="https://haosensun.github.io/">Haosen Sun</a>, <a href="https://luhr2003.github.io/">Haoran Lu</a>, <a href="https://www.linkedin.com/in/desmond-dingcheng-wang/">Dingcheng Wang</a>, Letian Xue, <a href="https://www.mccormick.northwestern.edu/research-faculty/directory/profiles/liu-han.html">Han Liu</a>
  <br>
  <i>(* equal contribution)</i>
</p>

<p align="center">
  <a href="https://progresslm.github.io/ProgressLM/"><img src="https://img.shields.io/badge/🌐-Website-blue?style=for-the-badge" alt="Website" /></a>
  <a href="https://github.com/ProgressLM/ProgressLM/blob/main/public/ProgressLM_Tech_Report.pdf"><img src="https://img.shields.io/badge/📄-Paper-black?style=for-the-badge" alt="Paper" /></a>
  <a href="https://huggingface.co/Raymond-Qiancx/ProgressLM-3B-SFT"><img src="https://img.shields.io/badge/🤗-SFT Model-orange?style=for-the-badge" alt="SFT Model" /></a>
  <a href="https://huggingface.co/Raymond-Qiancx/ProgressLM-3B-RL"><img src="https://img.shields.io/badge/🤗-RL Model-orange?style=for-the-badge" alt="RL Model" /></a>
  <a href="https://huggingface.co/datasets/Raymond-Qiancx/ProgressLM-Dataset"><img src="https://img.shields.io/badge/🤗-Dataset-yellow?style=for-the-badge" alt="Dataset" /></a>
</p>

<p align="center">
  <img src="docs/imgs/teaser.png" width="100%" />
</p>

<p align="justify">
Given a task demonstration and a single observation, the goal is to estimate <b>how much of the task has already been completed</b>. Direct prediction can often judge whether the task is unfinished, but struggles to assign a well-calibrated progress score. Progress reasoning instead follows a coarse-to-fine process: it first performs <b>episodic retrieval</b> to coarsely locate the observation along the demonstrated task, then applies <b>mental simulation</b> to imagine the transition from the retrieved anchor to the current observation, enabling a fine-grained estimate of completed progress.
</p>

<p align="justify">
<b><i>Can vision-language models acquire progress estimation as a general reasoning capability from a single observation?</i></b> To systematically study this question, we select robotic manipulation tasks as a controlled and representative domain, where task execution exhibits clear, interpretable, and temporally ordered progressions. Each instance provides a task <i>demonstration</i> and a single <i>observation</i>, and the model is required to predict a normalized <i>progress score</i> indicating how far the task has progressed.
</p>

<p align="center">
  <img src="docs/imgs/bench_curation.png" width="100%" />
</p>

<p align="justify">
Progress annotated data is constructed along three key dimensions.
<b>(i) Demonstration modality</b> compares vision-based demonstrations that present state trajectories with text-based demonstrations that provide step-by-step action descriptions.
<b>(ii) Viewpoint correspondence</b> controls whether demonstrations and observations are captured from the same camera viewpoint or from different viewpoints.
<b>(iii) Answerability</b> explicitly distinguishes between cases where progress is well-defined and cases where reliable estimation is inherently ambiguous.
This design allows us to disentangle perception, temporal reasoning, and uncertainty awareness in progress estimation.
</p>

### Overview of ProgressLM-Dataset

<p align="center">
  <img src="docs/imgs/statistics.png" width="100%" />
</p>

<p align="justify">
Data statistics of <b>Progress-Bench</b> and <b>ProgressLM-45K</b> (25K for SFT while 20K for RL). Traj and Samp denote the numbers of task trajectories and sampled observations to be estimated, respectively. The upper-right panel shows the four distinct robotic embodiments included, while the lower-right panel visualizes the diversity of objects involved in task interactions.
</p>



## Installation



### SFT Environment

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning. It supports LoRA, QLoRA, and full fine-tuning with various model architectures including Qwen2.5-VL.

```bash
conda create -n progsft python=3.11 -y
conda activate progsft
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```


### RL Environment

We use [EasyR1](https://github.com/hiyouga/EasyR1) for reinforcement learning with GRPO (Group Relative Policy Optimization). It provides distributed training support with FSDP and efficient rollout generation.

```bash
conda create -n progrl python=3.11 -y
conda activate progrl
cd EasyR1
pip install -e .
```


### Evaluation Environment

```bash
conda create -n progresslm python=3.11 -y
conda activate progresslm
pip install -r eval/requirement.txt
```

## Datasets

### prog-bench

Programmatic task benchmarks for progress reasoning evaluation.

| File | Description |
|------|-------------|
| `text-normal.jsonl` | Text demonstration normal samples |
| `text-unanswerable.jsonl` | Unanswerable text samples |
| `visual_same_view.jsonl` | Same-view visual demonstrations |
| `visual_cross_view.jsonl` | Cross-view visual demonstrations |
| `visual-unanswerable.jsonl` | Unanswerable visual samples |

### human-bench

Human activity benchmarks for progress reasoning evaluation.

| File | Description |
|------|-------------|
| `text_demo_human_activities.jsonl` | Text demonstrations of human activities |
| `visual_demo_human_activities.jsonl` | Visual demonstrations of human activities |

## SFT Training


### Configuration Files

| Config File | Description |
|-------------|-------------|
| `qwen2_5vl_lora_sft_small.yaml` | Qwen2.5-VL-3B LoRA SFT config |
| `qwen2_5vl_lora_sft_7b.yaml` | Qwen2.5-VL-7B LoRA SFT config |
| `qwen3vl_4b_lora_sft.yaml` | Qwen3-VL-4B LoRA SFT config |

### Running SFT Training

```bash
cd LLaMA-Factory

# Qwen2.5-VL-3B (Single GPU)
bash our_scripts/train_qwen2_5vl_lora_sft.sh

# Qwen2.5-VL-3B (Multi GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash our_scripts/train_qwen2_5vl_lora_sft.sh

# Qwen2.5-VL-7B
bash our_scripts/train_qwen2_5vl_lora_sft_7b.sh

# Qwen3-VL-4B
bash our_scripts/train_qwen3vl_4b_lora_sft.sh
```

### Merge LoRA Weights

After SFT training, merge LoRA weights into the base model:

```bash
# Merge Qwen2.5-VL-3B LoRA
llamafactory-cli export LLaMA-Factory/our_scripts/merge_qwen2_5vl_lora.yaml

# Merge Qwen2.5-VL-7B LoRA
llamafactory-cli export LLaMA-Factory/our_scripts/merge_qwen25vl_7b_lora.yaml
```

### Scaling more CoT Data?

**Step 1: Generate CoT responses using Qwen2.5-VL**

```bash
cd eval/qwen25vl/scripts/cot_gen

# Generate CoT for Text Demo data
MODEL_PATH=/path/to/Qwen2.5-VL-32B-Instruct \
DATASET_PATH=/path/to/text_demo.jsonl \
IMAGE_ROOT=/path/to/images \
OUTPUT_DIR=/path/to/output \
GPU_IDS=0,1,2,3 \
BATCH_SIZE=5 \
bash think_text_demo.sh

# Generate CoT for Visual Demo data
MODEL_PATH=/path/to/Qwen2.5-VL-32B-Instruct \
DATASET_PATH=/path/to/visual_demo.jsonl \
IMAGE_ROOT=/path/to/images \
OUTPUT_DIR=/path/to/output \
GPU_IDS=0,1,2,3 \
BATCH_SIZE=2 \
bash think_visual_demo.sh
```

We also provide scripts for **Qwen2.5-VL-72B** with multi-GPU model parallelism:

```bash
# 72B model for Text Demo (requires 4+ GPUs for model parallelism)
MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct \
DATASET_PATH=/path/to/text_demo.jsonl \
GPU_IDS=0,1,2,3 \
BATCH_SIZE=40 \
bash think_text_demo_72b.sh

# 72B model for Visual Demo
MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct \
DATASET_PATH=/path/to/visual_demo.jsonl \
GPU_IDS=0,1,2,3 \
BATCH_SIZE=20 \
bash think_visual_demo_72b.sh
```

**Step 2: Convert CoT responses to LLaMA-Factory format**

```bash
cd LLaMA-Factory/our_scripts/data_convert

# Convert Text Demo data
python convert_text_demo.py \
    --original-data /path/to/text_demo.jsonl \
    --cot-responses /path/to/cot_responses.jsonl \
    --output-file /path/to/output.json \
    --filter-success

# Convert Visual Demo data
python convert_visual_demo.py \
    --original-data /path/to/visual_demo.jsonl \
    --cot-responses /path/to/cot_responses.jsonl \
    --output-file /path/to/output.json \
    --filter-success

# Batch convert and merge all datasets
bash run_convert_and_merge.sh
```




## RL Training (GRPO)

### Configuration Files

| Config File | Description |
|-------------|-------------|
| `configs/visual_demo_grpo.yaml` | Qwen2.5-VL-3B GRPO config |
| `configs/visual_demo_grpo_7b.yaml` | Qwen2.5-VL-7B GRPO config |
| `configs/multinodes.yaml` | Multi-node training config |

### Running RL Training

```bash
cd EasyR1

# Qwen2.5-VL-3B (Single Node)
bash progresslm/run_grpo_3b.sh

# Qwen2.5-VL-3B (Multi Node)
bash progresslm/run_grpo_3b_multinode.sh

# Qwen2.5-VL-7B
bash progresslm/run_grpo_7b.sh
```

## Evaluation

### Supported Models

| Model | Directory | Description |
|-------|-----------|-------------|
| Qwen2.5-VL | `eval/qwen25vl/` | Qwen2.5-VL series (3B, 7B, 32B, 72B) |
| Qwen3-VL | `eval/qwen3vl/` | Qwen3-VL series (2B, 4B, 8B, 32B) |
| Intern3.5-VL | `eval/internvl/` | Intern3.5-VL series (4B, 8B, 14B, 38B) |
| OpenAI GPT-5 | `eval/openai/` | GPT-5, GPT-5-mini via API |

### Benchmark Scripts

Evaluation scripts are organized in `eval/qwen25vl/scripts/benchmarks/`:

| Benchmark | Description | Scripts |
|-----------|-------------|---------|
| `text_based/` | Text demonstration (normal) | `eval_text_normal_sft_3b.sh`, `eval_text_normal_rl_3b.sh`, ... |
| `same_view/` | Visual demonstration (same view) | `visual_eval_one_view_3B_SFT.sh`, `visual_eval_one_view_3B_RL.sh`, ... |
| `cross_view/` | Visual demonstration (cross view) | Cross-view evaluation scripts |
| `text_unanswer/` | Text unanswerable samples | Text unanswerable evaluation scripts |
| `vision_unanswer/` | Visual unanswerable samples | Visual unanswerable evaluation scripts |
| `human/` | Human activity benchmarks | Human activity evaluation scripts |

### Running Evaluation

#### Text Demo Evaluation (prog-bench)

```bash
cd eval/qwen25vl/scripts/benchmarks/text_based

# SFT Model (3B)
bash eval_text_normal_sft_3b.sh

# RL Model (3B)
bash eval_text_normal_rl_3b.sh

# SFT Model (7B)
bash eval_text_normal_sft_7b.sh

# Large Models (72B)
bash eval_text_normal_72b.sh
```

#### Visual Demo Evaluation (prog-bench)

```bash
cd eval/qwen25vl/scripts/benchmarks/same_view

# SFT Model (3B)
bash visual_eval_one_view_3B_SFT.sh

# RL Model (3B)
bash visual_eval_one_view_3B_RL.sh

# SFT Model (7B)
bash visual_eval_one_view_7B_SFT.sh

# Large Models (72B)
bash visual_eval_one_view_72B.sh
```

#### Human Activity Evaluation (human-bench)

```bash
cd eval/qwen25vl/scripts/benchmarks/human

# Text Demo - Human Activities
bash text_eval_human_rl_3b.sh

# Visual Demo - Human Activities
bash visual_eval_human_3B_RL.sh
```

#### Nothink Mode Evaluation

For models without thinking process:

```bash
# Text Demo Nothink
cd eval/qwen25vl/scripts/benchmarks/text_based
bash nothink_3b.sh
bash nothink_7b.sh
bash nothink_72b.sh

# Visual Demo Nothink
cd eval/qwen25vl/scripts/benchmarks/same_view
bash visual_eval_one_view_nothink_3B.sh
bash visual_eval_one_view_nothink_7B.sh
bash visual_eval_one_view_nothink_72B.sh
```

### Manual Evaluation Command

```bash
cd eval/qwen25vl/codes

# Text Demo Evaluation
python run_text_demo.py \
    --model-path /path/to/model \
    --dataset-path /path/to/text_demo.jsonl \
    --output-file /path/to/results.jsonl \
    --image-root /path/to/images \
    --batch-size 100 \
    --temperature 0.6 \
    --max-new-tokens 4096

# Visual Demo Evaluation
python run_visual_demo.py \
    --model-path /path/to/model \
    --dataset-path /path/to/visual_demo.jsonl \
    --output-file /path/to/results.jsonl \
    --image-root /path/to/images \
    --batch-size 50 \
    --temperature 0.6 \
    --max-new-tokens 4096

# Nothink Mode
python run_text_demo_nothink.py \
    --model-path /path/to/model \
    --dataset-path /path/to/text_demo.jsonl \
    --output-file /path/to/results.jsonl

python run_visual_demo_nothink.py \
    --model-path /path/to/model \
    --dataset-path /path/to/visual_demo.jsonl \
    --output-file /path/to/results.jsonl
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **NSE** (Normalized Score Error) | Measures single-point progress accuracy on answerable samples by quantifying the normalized deviation between the predicted progress score and ground truth |
| **PRC** (Progress Rank Correlation) | Measures trajectory-level temporal consistency by evaluating whether predicted progress preserves the correct relative ordering along a task trajectory (Spearman rank correlation) |
| **AFRR** (Answerable False Rejection Rate) | Measures answerability awareness on answerable samples by computing the fraction of valid cases incorrectly predicted as unanswerable (N/A) |
| **UDA** (Unanswerable Detection Accuracy) | Measures unanswerable case recognition by computing the fraction of unanswerable samples correctly predicted as unanswerable (N/A) |


### Other Model Evaluation

#### Qwen3-VL

We support Qwen3-VL series models (2B, 4B, 8B, 32B, 30B-MoE) with both thinking and non-thinking modes:

```bash
cd eval/qwen3vl/scripts

# Run all benchmarks for a specific model size
bash run_all/run_8b.sh          # Qwen3-VL-8B with thinking
bash run_all/run_8b_nothink.sh  # Qwen3-VL-8B without thinking

# Or run specific benchmarks
bash text_based/qwen3vl_8b.sh   # Text Demo evaluation
bash same_view/qwen3vl_8b.sh   # Visual Demo evaluation
bash text_unanswer/qwen3vl_8b.sh     # Text unanswerable evaluation
bash cross_view/qwen3vl_8b.sh    # Cross-view evaluation
```


#### Intern3.5-VL

```bash
cd eval/internvl/codes
python run_text_demo.py --model-path /path/to/internvl --dataset-path /path/to/data.jsonl
python run_visual_demo.py --model-path /path/to/internvl --dataset-path /path/to/data.jsonl
```

#### OpenAI API

```bash
cd eval/openai/codes
export OPENAI_API_KEY=your_api_key
python run_text_demo.py --dataset-path /path/to/data.jsonl
python run_visual_demo.py --dataset-path /path/to/data.jsonl
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2025progresslm,
  title={ProgressLM: Towards Progress Reasoning in Vision-Language Models},
  author={Zhang, Jianshu and Qian, Chengxuan and Sun, Haosen and Lu, Haoran and Wang, Dingcheng and Xue, Letian and Liu, Han},
  year={2025}
}
```
