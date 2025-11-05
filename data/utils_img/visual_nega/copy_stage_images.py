import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

def copy_stage_images(jsonl_file):
    """
    从 visual_negative_replacement 文件夹复制 stage_to_estimate 图片到对应的 id 文件夹
    不自动创建目标文件夹，所有异常都会被记录和报告
    """
    # 路径配置
    source_base = Path('/projects/p32958/chengxuan/new_extracted_images/images/visual_negative_replacement')
    target_base = Path('/projects/p32958/chengxuan/new_extracted_images/images')

    # 统计信息
    stats = {
        'total_samples': 0,
        'total_images': 0,
        'copied_images': 0,
        'missing_source_images': 0,
        'missing_target_dirs': 0,
        'failed_copies': 0,
        'json_parse_errors': 0,
        'missing_id_field': 0,
        'missing_stage_field': 0
    }

    # 异常详情列表
    exceptions = {
        'missing_source_images': [],
        'missing_target_dirs': [],
        'failed_copies': [],
        'json_parse_errors': [],
        'missing_id_field': [],
        'missing_stage_field': []
    }

    print("开始处理文件...")
    print(f"源文件夹：{source_base}")
    print(f"目标基础路径：{target_base}")
    print("="*60)

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                stats['json_parse_errors'] += 1
                exceptions['json_parse_errors'].append({
                    'line': line_num,
                    'error': str(e)
                })
                print(f"[异常] 第 {line_num} 行 JSON 解析失败: {e}")
                continue

            stats['total_samples'] += 1

            # 检查 id 字段
            sample_id = data.get('id', '')
            if not sample_id:
                stats['missing_id_field'] += 1
                exceptions['missing_id_field'].append({
                    'line': line_num
                })
                print(f"[异常] 第 {line_num} 行缺少 id 字段")
                continue

            # 检查 stage_to_estimate 字段
            stage_images = data.get('stage_to_estimate', [])
            if not stage_images:
                stats['missing_stage_field'] += 1
                exceptions['missing_stage_field'].append({
                    'line': line_num,
                    'id': sample_id
                })
                print(f"[异常] 第 {line_num} 行缺少 stage_to_estimate 字段 (id: {sample_id})")
                continue

            # 检查目标文件夹是否存在
            target_dir = target_base / sample_id
            if not target_dir.exists():
                stats['missing_target_dirs'] += 1
                exceptions['missing_target_dirs'].append({
                    'line': line_num,
                    'id': sample_id,
                    'path': str(target_dir)
                })
                print(f"[异常] 第 {line_num} 行目标文件夹不存在: {target_dir}")
                # 跳过这个样本的所有图片
                stats['total_images'] += len(stage_images)
                continue

            # 处理每个 stage_to_estimate 图片
            for img_name in stage_images:
                stats['total_images'] += 1

                source_path = source_base / img_name
                target_path = target_dir / img_name

                # 检查源文件是否存在
                if not source_path.exists():
                    stats['missing_source_images'] += 1
                    exceptions['missing_source_images'].append({
                        'line': line_num,
                        'id': sample_id,
                        'image': img_name,
                        'path': str(source_path)
                    })
                    print(f"[异常] 第 {line_num} 行源图片不存在: {img_name}")
                    continue

                # 复制文件
                try:
                    shutil.copy2(source_path, target_path)
                    stats['copied_images'] += 1
                    if stats['copied_images'] <= 5:  # 只打印前5个成功案例
                        print(f"[成功] 复制 {img_name} -> {target_dir}")
                except Exception as e:
                    stats['failed_copies'] += 1
                    exceptions['failed_copies'].append({
                        'line': line_num,
                        'id': sample_id,
                        'image': img_name,
                        'error': str(e)
                    })
                    print(f"[异常] 第 {line_num} 行复制失败: {img_name}，错误: {e}")

            # 每处理100个样本打印一次进度
            if stats['total_samples'] % 100 == 0:
                print(f"[进度] 已处理 {stats['total_samples']} 个样本，成功复制 {stats['copied_images']} 张图片...")

    # 打印详细统计报告
    print("\n" + "="*60)
    print("统计报告")
    print("="*60)
    print(f"总样本数：{stats['total_samples']}")
    print(f"总图片数：{stats['total_images']}")
    print(f"成功复制：{stats['copied_images']}")
    print(f"成功率：{stats['copied_images']/stats['total_images']*100:.2f}%" if stats['total_images'] > 0 else "N/A")
    print()
    print("异常统计：")
    print(f"  - 源图片不存在：{stats['missing_source_images']}")
    print(f"  - 目标文件夹不存在：{stats['missing_target_dirs']}")
    print(f"  - 文件复制失败：{stats['failed_copies']}")
    print(f"  - JSON 解析错误：{stats['json_parse_errors']}")
    print(f"  - 缺少 id 字段：{stats['missing_id_field']}")
    print(f"  - 缺少 stage_to_estimate 字段：{stats['missing_stage_field']}")

    # 打印详细异常列表（每类最多显示10个）
    print("\n" + "="*60)
    print("异常详情（每类最多显示前10个）")
    print("="*60)

    if exceptions['missing_source_images']:
        print(f"\n源图片不存在 ({len(exceptions['missing_source_images'])} 个):")
        for item in exceptions['missing_source_images'][:10]:
            print(f"  行 {item['line']}: {item['image']}")

    if exceptions['missing_target_dirs']:
        print(f"\n目标文件夹不存在 ({len(exceptions['missing_target_dirs'])} 个):")
        for item in exceptions['missing_target_dirs'][:10]:
            print(f"  行 {item['line']}: {item['path']}")

    if exceptions['failed_copies']:
        print(f"\n复制失败 ({len(exceptions['failed_copies'])} 个):")
        for item in exceptions['failed_copies'][:10]:
            print(f"  行 {item['line']}: {item['image']} - {item['error']}")

    if exceptions['json_parse_errors']:
        print(f"\nJSON 解析错误 ({len(exceptions['json_parse_errors'])} 个):")
        for item in exceptions['json_parse_errors'][:10]:
            print(f"  行 {item['line']}: {item['error']}")

    if exceptions['missing_id_field']:
        print(f"\n缺少 id 字段 ({len(exceptions['missing_id_field'])} 个):")
        for item in exceptions['missing_id_field'][:10]:
            print(f"  行 {item['line']}")

    if exceptions['missing_stage_field']:
        print(f"\n缺少 stage_to_estimate 字段 ({len(exceptions['missing_stage_field'])} 个):")
        for item in exceptions['missing_stage_field'][:10]:
            print(f"  行 {item['line']}: id = {item['id']}")

    print("\n" + "="*60)
    print("处理完成！")


if __name__ == '__main__':
    jsonl_file = '/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_replaced_processed.jsonl'
    copy_stage_images(jsonl_file)
