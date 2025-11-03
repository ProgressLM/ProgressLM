#!/usr/bin/env python3
"""
验证JSONL文件中的所有图片是否都能在ProgressLM/images目录下找到
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ImageVerifier:
    """图像验证器"""

    def __init__(self, jsonl_path: str, image_base_dir: str, num_threads: int = 16):
        """
        初始化验证器

        Args:
            jsonl_path: JSONL文件路径
            image_base_dir: 图像基础目录 (ProgressLM/images)
            num_threads: 验证线程数
        """
        self.jsonl_path = Path(jsonl_path)
        self.image_base_dir = Path(image_base_dir)
        self.num_threads = num_threads

        # 统计信息（线程安全）
        self.lock = threading.Lock()
        self.total_samples = 0
        self.total_images = 0
        self.valid_images = 0
        self.missing_images = 0

        # 样本级别统计
        self.complete_samples = 0  # 所有图片都存在的样本
        self.incomplete_samples = 0  # 至少有一张图片缺失的样本

        # 记录详细信息
        self.missing_records = []  # 缺失的图片记录
        self.incomplete_sample_ids = []  # 不完整的样本ID

    def build_image_path(self, sample_id: str, filename: str) -> Path:
        """
        构建图像路径

        Args:
            sample_id: 样本ID "data_source/action_type/trajectory_id"
            filename: 图像文件名

        Returns:
            图像完整路径
        """
        parts = sample_id.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid sample_id format: {sample_id}")

        data_source, action_type, trajectory_id = parts
        return self.image_base_dir / data_source / action_type / trajectory_id / filename

    def verify_sample(self, sample: Dict) -> Tuple[int, int, List[Dict]]:
        """
        验证单个样本的所有图片

        Args:
            sample: JSONL样本

        Returns:
            (total_images, valid_images, missing_records)
        """
        sample_id = sample['id']
        total_images = 0
        valid_images = 0
        missing_records = []

        # 收集所有图片
        all_images = []
        if 'visual_demo' in sample:
            for img in sample['visual_demo']:
                all_images.append(('visual_demo', img))
        if 'stage_to_estimate' in sample:
            for img in sample['stage_to_estimate']:
                all_images.append(('stage_to_estimate', img))

        # 验证每张图片
        for field, filename in all_images:
            total_images += 1
            image_path = self.build_image_path(sample_id, filename)

            if image_path.exists():
                valid_images += 1
            else:
                missing_records.append({
                    'sample_id': sample_id,
                    'field': field,
                    'filename': filename,
                    'expected_path': str(image_path)
                })

        return total_images, valid_images, missing_records

    def update_statistics(self, sample_id: str, total_images: int,
                         valid_images: int, missing_records: List[Dict]):
        """
        更新统计信息（线程安全）

        Args:
            sample_id: 样本ID
            total_images: 该样本的总图片数
            valid_images: 有效图片数
            missing_records: 缺失记录
        """
        with self.lock:
            self.total_samples += 1
            self.total_images += total_images
            self.valid_images += valid_images
            self.missing_images += len(missing_records)

            # 样本级别统计
            if len(missing_records) == 0:
                self.complete_samples += 1
            else:
                self.incomplete_samples += 1
                self.incomplete_sample_ids.append(sample_id)
                self.missing_records.extend(missing_records)

    def verify_all(self) -> Dict:
        """
        验证所有样本

        Returns:
            验证报告
        """
        print(f"开始验证JSONL文件...")
        print(f"JSONL文件: {self.jsonl_path}")
        print(f"图像目录: {self.image_base_dir}")
        print(f"线程数: {self.num_threads}")
        print("-" * 80)

        # 读取所有样本
        samples = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        print(f"读取样本数: {len(samples)}")

        # 多线程验证
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(self.verify_sample, sample): sample
                      for sample in samples}

            completed = 0
            for future in as_completed(futures):
                sample = futures[future]
                total_images, valid_images, missing_records = future.result()
                self.update_statistics(sample['id'], total_images, valid_images, missing_records)

                completed += 1
                if completed % 100 == 0:
                    print(f"已验证: {completed}/{len(samples)} ({completed*100//len(samples)}%)")

        print(f"\n验证完成！已验证 {completed}/{len(samples)} 样本")

        # 生成报告
        report = self.generate_report()
        return report

    def generate_report(self) -> Dict:
        """
        生成验证报告

        Returns:
            报告字典
        """
        # 按样本ID分组缺失信息
        missing_by_sample = defaultdict(list)
        for record in self.missing_records:
            missing_by_sample[record['sample_id']].append({
                'field': record['field'],
                'filename': record['filename'],
                'expected_path': record['expected_path']
            })

        report = {
            'summary': {
                'jsonl_file': str(self.jsonl_path),
                'image_base_dir': str(self.image_base_dir),

                # 样本级别
                'total_samples': self.total_samples,
                'complete_samples': self.complete_samples,
                'incomplete_samples': self.incomplete_samples,
                'sample_complete_rate': f"{self.complete_samples * 100 / self.total_samples:.2f}%" if self.total_samples > 0 else "0%",

                # 图片级别
                'total_images': self.total_images,
                'valid_images': self.valid_images,
                'missing_images': self.missing_images,
                'image_valid_rate': f"{self.valid_images * 100 / self.total_images:.2f}%" if self.total_images > 0 else "0%",
            },
            'incomplete_samples': [
                {
                    'sample_id': sample_id,
                    'missing_images': missing_by_sample[sample_id]
                }
                for sample_id in self.incomplete_sample_ids
            ],
            'timestamp': datetime.now().isoformat()
        }

        return report

    def print_summary(self, report: Dict):
        """
        打印统计摘要

        Args:
            report: 验证报告
        """
        summary = report['summary']

        print("\n" + "=" * 80)
        print("图像验证报告")
        print("=" * 80)

        print("\n【文件信息】")
        print(f"  JSONL文件:    {summary['jsonl_file']}")
        print(f"  图像目录:     {summary['image_base_dir']}")

        print("\n【样本级别统计】")
        print(f"  总样本数:           {summary['total_samples']}")
        print(f"  完整样本数:         {summary['complete_samples']} ({summary['sample_complete_rate']})")
        print(f"    (该样本的所有图片都存在)")
        print(f"  不完整样本数:       {summary['incomplete_samples']}")
        print(f"    (该样本至少有一张图片缺失)")

        print("\n【图片级别统计】")
        print(f"  总图片数:           {summary['total_images']}")
        print(f"  存在的图片数:       {summary['valid_images']} ({summary['image_valid_rate']})")
        print(f"  缺失的图片数:       {summary['missing_images']}")

        # 显示部分缺失记录
        if report['incomplete_samples']:
            print("\n【缺失图片示例】（前10条）")
            for i, item in enumerate(report['incomplete_samples'][:10], 1):
                print(f"\n  {i}. 样本: {item['sample_id']}")
                for missing in item['missing_images'][:3]:  # 每个样本最多显示3张缺失图片
                    print(f"     ✗ {missing['field']}: {missing['filename']}")
                    print(f"       期望路径: {missing['expected_path']}")
                if len(item['missing_images']) > 3:
                    print(f"     ... 还有 {len(item['missing_images']) - 3} 张缺失图片")

            if len(report['incomplete_samples']) > 10:
                print(f"\n  ... 还有 {len(report['incomplete_samples']) - 10} 个不完整样本")

        print("=" * 80)

        # 验证结果判断
        if summary['missing_images'] == 0:
            print("\n✓ 验证通过！所有图片都存在。")
        else:
            print(f"\n✗ 验证失败！发现 {summary['missing_images']} 张缺失图片。")

    def save_report(self, report: Dict, output_path: Path):
        """
        保存详细报告到JSON文件

        Args:
            report: 报告字典
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存至: {output_path}")


def main():
    """主函数"""
    # 配置路径
    JSONL_PATH = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/train/visual_demo/visual_franka_cross_camera_augmented_sft.jsonl"
    IMAGE_BASE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/robomind/data/ProgressLM/images"
    REPORT_DIR = Path("/home/vcj9002/jianshu/chengxuan/ProgressLM/data/utils_img/sft_manage")

    # 配置参数
    NUM_THREADS = 16  # 验证是IO密集型，线程数不需要太多

    print("=" * 80)
    print("ProgressLM图像验证工具")
    print("=" * 80)

    # 创建验证器
    verifier = ImageVerifier(
        jsonl_path=JSONL_PATH,
        image_base_dir=IMAGE_BASE_DIR,
        num_threads=NUM_THREADS
    )

    # 执行验证
    report = verifier.verify_all()

    # 打印摘要
    verifier.print_summary(report)

    # 保存详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"image_verification_report_{timestamp}.json"
    verifier.save_report(report, report_path)


if __name__ == "__main__":
    main()
