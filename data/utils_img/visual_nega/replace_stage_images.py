#!/usr/bin/env python3
"""
使用 jina-clip-v2 对 stage_to_estimate 图片进行相似度匹配和替换
优化版：支持多GPU并行处理
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel
from tqdm import tqdm
from collections import defaultdict

# ==================== 配置参数 ====================
INPUT_JSONL = "/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_raw.jsonl"
OUTPUT_JSONL = "/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_replaced.jsonl"
IMAGE_DIR = "/projects/p32958/chengxuan/new_extracted_images/images/visual_negative_replacement"
LOG_FILE = "/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/replacement_log.json"

BATCH_SIZE = 512  # H100 可以处理更大的 batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_IDS = [0, 1, 2, 3]  # 指定使用的GPU ID
NUM_GPUS = len(GPU_IDS)

print(f"🚀 使用 {NUM_GPUS} 块 GPU: {GPU_IDS}")
print(f"📦 批处理大小: {BATCH_SIZE}")


# ==================== 工具函数 ====================
def construct_image_path(record_id: str, img_name: str) -> str:
    """构建图片完整路径"""
    # 将 id 中的 '/' 替换为 '_'
    safe_id = record_id.replace('/', '_')
    filename = f"{safe_id}_{img_name}"
    return os.path.join(IMAGE_DIR, filename)


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    print(f"📖 加载数据: {file_path}")
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="读取JSONL"):
        records.append(json.loads(line.strip()))
    print(f"✅ 加载了 {len(records)} 条记录")
    return records


def save_jsonl(records: List[Dict], file_path: str):
    """保存 JSONL 文件"""
    print(f"💾 保存到: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in tqdm(records, desc="保存JSONL"):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✅ 保存完成")


def build_candidate_pool(records: List[Dict]) -> Dict[int, List[Tuple[int, str]]]:
    """
    为每条记录构建候选图片池
    返回: {record_idx: [(candidate_record_idx, candidate_img_path), ...]}
    """
    print("🔍 构建候选图片池...")

    # 按 data_source 分组
    data_source_groups = defaultdict(list)
    for idx, record in enumerate(records):
        data_source_groups[record['data_source']].append(idx)

    candidate_pool = {}
    missing_images = []

    for idx, record in enumerate(tqdm(records, desc="构建候选池")):
        candidates = []
        current_data_source = record['data_source']
        current_task_goal = record['task_goal']
        current_id = record['id']

        # 在相同 data_source 的记录中查找
        for candidate_idx in data_source_groups[current_data_source]:
            if candidate_idx == idx:
                continue

            candidate_record = records[candidate_idx]

            # 检查：task_goal 和 id 必须不同
            if (candidate_record['task_goal'] != current_task_goal and
                candidate_record['id'] != current_id):

                # 获取候选图片（stage_to_estimate 的第一张）
                candidate_img_name = candidate_record['stage_to_estimate'][0]
                candidate_img_path = construct_image_path(
                    candidate_record['id'],
                    candidate_img_name
                )

                # 验证图片存在
                if os.path.exists(candidate_img_path):
                    candidates.append((candidate_idx, candidate_img_path))
                else:
                    missing_images.append(candidate_img_path)

        candidate_pool[idx] = candidates

    if missing_images:
        print(f"⚠️  警告: 发现 {len(missing_images)} 张图片不存在")
        print(f"   前5个示例: {missing_images[:5]}")

    # 统计
    avg_candidates = np.mean([len(v) for v in candidate_pool.values()])
    print(f"✅ 候选池构建完成，平均每条记录有 {avg_candidates:.0f} 个候选图片")

    return candidate_pool


def load_images_batch(image_paths: List[str], show_progress: bool = False) -> List[Image.Image]:
    """批量加载图片"""
    images = []
    iterator = tqdm(image_paths, desc="加载图片", leave=False) if show_progress else image_paths
    for path in iterator:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"⚠️  加载图片失败: {path}, 错误: {e}")
            # 使用黑色占位图
            images.append(Image.new('RGB', (512, 512), color='black'))
    return images


def encode_images_batch(model, image_paths: List[str], batch_size: int) -> np.ndarray:
    """批量编码图片"""
    all_embeddings = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    # 处理 DataParallel 包装的模型
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    for i in tqdm(range(0, len(image_paths), batch_size),
                  desc="编码图片",
                  total=total_batches,
                  unit="batch"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = load_images_batch(batch_paths)

        with torch.no_grad():
            embeddings = actual_model.encode_image(batch_images)
            # embeddings 已经是 numpy array，无需转换
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("🎯 开始执行图片相似度匹配任务")
    print("=" * 60)

    # 1. 加载模型
    print("\n📥 加载 jina-clip-v2 模型...")
    model = AutoModel.from_pretrained(
        'jinaai/jina-clip-v2',
        trust_remote_code=True
    )

    # 使用多GPU
    if NUM_GPUS > 1:
        print(f"🔧 启用 {NUM_GPUS} 块 GPU 并行处理: {GPU_IDS}")
        model = torch.nn.DataParallel(model, device_ids=GPU_IDS)
        model = model.to(f'cuda:{GPU_IDS[0]}')
    else:
        model = model.to(DEVICE)

    model.eval()
    print("✅ 模型加载完成")

    # 2. 加载数据
    records = load_jsonl(INPUT_JSONL)

    # 3. 构建候选池
    candidate_pool = build_candidate_pool(records)

    # 4. 收集所有需要编码的图片
    print("\n📊 收集所有需要编码的图片...")
    all_images = set()

    # 收集所有 stage_to_estimate 图片
    for record in tqdm(records, desc="收集stage_to_estimate图片"):
        img_path = construct_image_path(record['id'], record['stage_to_estimate'][0])
        all_images.add(img_path)

    # 收集所有候选图片
    for candidates in tqdm(candidate_pool.values(), desc="收集候选图片"):
        for _, img_path in candidates:
            all_images.add(img_path)

    all_images = list(all_images)
    print(f"✅ 共需编码 {len(all_images)} 张唯一图片")

    # 5. 批量编码所有图片
    print("\n🎨 开始批量编码图片...")
    embeddings_array = encode_images_batch(model, all_images, BATCH_SIZE)

    # 构建路径到嵌入的映射
    image_to_embedding = {path: emb for path, emb in zip(all_images, embeddings_array)}
    print(f"✅ 编码完成，嵌入维度: {embeddings_array.shape[1]}")

    # 6. 相似度匹配
    print("\n🔗 开始相似度匹配...")
    replacement_log = []

    with tqdm(total=len(records), desc="匹配进度", unit="record") as pbar:
        for idx, record in enumerate(records):
            # 获取当前图片的嵌入
            current_img_name = record['stage_to_estimate'][0]
            current_img_path = construct_image_path(record['id'], current_img_name)

            if current_img_path not in image_to_embedding:
                pbar.set_postfix_str(f"⚠️ 跳过: 图片不存在")
                pbar.update(1)
                continue

            current_embedding = image_to_embedding[current_img_path]

            # 获取候选池
            candidates = candidate_pool[idx]

            if not candidates:
                pbar.set_postfix_str(f"⚠️ 跳过: 无候选")
                pbar.update(1)
                continue

            # 计算相似度
            max_similarity = -1
            best_candidate_idx = None
            best_img_filename = None  # 完整的文件名（带 safe_id 前缀）
            best_original_img_name = None  # 原始图片名（用于日志）

            for candidate_idx, candidate_img_path in candidates:
                candidate_embedding = image_to_embedding[candidate_img_path]

                # 余弦相似度（归一化后的点积）
                similarity = np.dot(current_embedding, candidate_embedding)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_candidate_idx = candidate_idx
                    # 从完整路径中提取文件名
                    best_img_filename = os.path.basename(candidate_img_path)
                    best_original_img_name = records[candidate_idx]['stage_to_estimate'][0]

            # 替换
            if best_img_filename:
                original_img = record['stage_to_estimate'][0]
                record['stage_to_estimate'] = [best_img_filename]  # 使用完整文件名

                replacement_log.append({
                    'record_idx': idx,
                    'record_id': record['id'],
                    'original_image': original_img,
                    'replaced_image': best_img_filename,  # 完整文件名
                    'replaced_original_name': best_original_img_name,  # 原始图片名（用于参考）
                    'similarity_score': float(max_similarity),
                    'source_record_id': records[best_candidate_idx]['id']
                })

                pbar.set_postfix_str(f"已替换: {len(replacement_log)}, 相似度: {max_similarity:.4f}")

            pbar.update(1)

    print(f"✅ 完成 {len(replacement_log)} 条记录的替换")

    # 7. 保存结果
    save_jsonl(records, OUTPUT_JSONL)

    # 8. 保存日志
    print(f"\n📝 保存替换日志...")
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'total_records': len(records),
            'replaced_records': len(replacement_log),
            'avg_similarity': np.mean([r['similarity_score'] for r in replacement_log]),
            'min_similarity': np.min([r['similarity_score'] for r in replacement_log]),
            'max_similarity': np.max([r['similarity_score'] for r in replacement_log]),
            'details': replacement_log
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ 日志保存到: {LOG_FILE}")

    # 9. 统计报告
    print("\n" + "=" * 60)
    print("📊 执行统计")
    print("=" * 60)
    print(f"总记录数: {len(records)}")
    print(f"成功替换: {len(replacement_log)}")
    print(f"替换率: {len(replacement_log)/len(records)*100:.2f}%")
    print(f"平均相似度: {np.mean([r['similarity_score'] for r in replacement_log]):.4f}")
    print(f"相似度范围: [{np.min([r['similarity_score'] for r in replacement_log]):.4f}, "
          f"{np.max([r['similarity_score'] for r in replacement_log]):.4f}]")
    print("=" * 60)
    print("🎉 任务完成！")


if __name__ == "__main__":
    main()
