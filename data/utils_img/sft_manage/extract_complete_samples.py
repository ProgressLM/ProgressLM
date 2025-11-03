#!/usr/bin/env python3
"""
提取所有完全匹配的样本（所有图像在其他相机视角都存在对应帧）
并生成新的JSONL文件和统计报告
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from verify_multi_camera_images import MultiCameraVerifier


class CompleteSampleExtractor:
    """完整样本提取器"""

    def __init__(self, jsonl_path: str, base_image_dir: str,
                 output_jsonl_path: str, num_threads: int = 16):
        """
        初始化提取器

        Args:
            jsonl_path: 输入JSONL文件路径
            base_image_dir: 图像基础目录路径
            output_jsonl_path: 输出JSONL文件路径
            num_threads: 验证线程数
        """
        self.jsonl_path = Path(jsonl_path)
        self.base_image_dir = Path(base_image_dir)
        self.output_jsonl_path = Path(output_jsonl_path)
        self.num_threads = num_threads

        # 创建验证器
        self.verifier = MultiCameraVerifier(
            jsonl_path=str(self.jsonl_path),
            base_image_dir=str(self.base_image_dir),
            num_threads=self.num_threads
        )

        # 统计信息
        self.total_samples = 0
        self.complete_samples = 0
        self.incomplete_samples = 0
        self.total_images_original = 0
        self.total_images_extracted = 0

    def extract_complete_samples(self) -> Dict:
        """
        提取所有完全匹配的样本

        Returns:
            统计报告字典
        """
        print(f"开始提取完全匹配的样本")
        print(f"输入文件: {self.jsonl_path}")
        print(f"输出文件: {self.output_jsonl_path}")
        print(f"图像基础目录: {self.base_image_dir}")
        print("-" * 80)

        # 执行验证
        verification_report = self.verifier.verify_all()

        # 读取原始样本
        print("\n读取原始样本...")
        samples = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        self.total_samples = len(samples)

        # 根据验证结果过滤完全匹配的样本
        print("\n过滤完全匹配的样本...")
        complete_sample_ids = set()

        # 从样本结果中找出所有完全匹配的样本ID
        for sample_result in self.verifier.sample_results:
            if sample_result.is_sample_complete:
                complete_sample_ids.add(sample_result.sample_id)

        # 提取完全匹配的样本
        complete_samples = []
        for sample in samples:
            sample_id = sample['id']

            # 统计原始图像数
            image_count = 0
            if 'visual_demo' in sample:
                image_count += len(sample['visual_demo'])
            if 'stage_to_estimate' in sample:
                image_count += len(sample['stage_to_estimate'])
            self.total_images_original += image_count

            # 检查是否完全匹配
            if sample_id in complete_sample_ids:
                complete_samples.append(sample)
                self.complete_samples += 1
                self.total_images_extracted += image_count
            else:
                self.incomplete_samples += 1

        # 保存完全匹配的样本
        print(f"\n保存完全匹配的样本到: {self.output_jsonl_path}")
        self.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_jsonl_path, 'w', encoding='utf-8') as f:
            for sample in complete_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 生成统计报告
        report = self.generate_report(verification_report)

        return report

    def generate_report(self, verification_report: Dict) -> Dict:
        """
        生成统计报告

        Args:
            verification_report: 验证报告

        Returns:
            统计报告字典
        """
        report = {
            'extraction_summary': {
                'input_file': str(self.jsonl_path),
                'output_file': str(self.output_jsonl_path),
                'base_image_dir': str(self.base_image_dir),

                # 样本统计
                'total_samples_in_input': self.total_samples,
                'complete_samples_extracted': self.complete_samples,
                'incomplete_samples_skipped': self.incomplete_samples,
                'extraction_rate': f"{self.complete_samples * 100 / self.total_samples:.2f}%" if self.total_samples > 0 else "0%",

                # 图像统计
                'total_images_in_input': self.total_images_original,
                'total_images_in_output': self.total_images_extracted,
                'image_retention_rate': f"{self.total_images_extracted * 100 / self.total_images_original:.2f}%" if self.total_images_original > 0 else "0%",
            },
            'verification_details': verification_report['summary'],
            'timestamp': datetime.now().isoformat()
        }

        return report

    def print_summary(self, report: Dict):
        """
        打印统计摘要

        Args:
            report: 统计报告字典
        """
        summary = report['extraction_summary']
        verification = report['verification_details']

        print("\n" + "=" * 80)
        print("完整样本提取报告")
        print("=" * 80)

        print("\n【文件信息】")
        print(f"  输入文件:     {summary['input_file']}")
        print(f"  输出文件:     {summary['output_file']}")
        print(f"  图像目录:     {summary['base_image_dir']}")

        print("\n【样本统计】")
        print(f"  输入总样本数:           {summary['total_samples_in_input']}")
        print(f"  提取的完整样本数:       {summary['complete_samples_extracted']} ({summary['extraction_rate']})")
        print(f"    ✓ 该样本的所有图像在其他两个相机视角都存在")
        print(f"  跳过的不完整样本数:     {summary['incomplete_samples_skipped']}")
        print(f"    ✗ 该样本至少有一张图像缺少某个视角")

        print("\n【图像统计】")
        print(f"  输入总图像数:           {summary['total_images_in_input']}")
        print(f"  输出总图像数:           {summary['total_images_in_output']}")
        print(f"  图像保留率:             {summary['image_retention_rate']}")

        print("\n【验证详情】")
        print(f"  完全匹配图像:           {verification['fully_matched_images']} ({verification['image_match_percentage']})")
        print(f"  部分匹配图像:           {verification['partially_matched_images']}")
        print(f"  完全缺失图像:           {verification['fully_missing_images']}")

        print("=" * 80)

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
    INPUT_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/train/visual_demo/visual_franka_3rgb_new_sft.jsonl"
    BASE_IMAGE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/robomind/data/3rgb"
    OUTPUT_DIR = Path("/home/vcj9002/jianshu/chengxuan/ProgressLM/data/train/visual_demo")
    REPORT_DIR = Path("/home/vcj9002/jianshu/chengxuan/ProgressLM/data/utils_img/sft_manage")

    # 生成输出文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_JSONL = OUTPUT_DIR / f"visual_franka_3rgb_complete_samples.jsonl"

    # 配置参数
    NUM_THREADS = 512  # 可根据机器性能调整

    print("=" * 80)
    print("完整样本提取工具")
    print("=" * 80)

    # 创建提取器
    extractor = CompleteSampleExtractor(
        jsonl_path=INPUT_JSONL,
        base_image_dir=BASE_IMAGE_DIR,
        output_jsonl_path=str(OUTPUT_JSONL),
        num_threads=NUM_THREADS
    )

    # 执行提取
    report = extractor.extract_complete_samples()

    # 打印摘要
    extractor.print_summary(report)

    # 保存详细报告
    report_path = REPORT_DIR / f"extraction_report_{timestamp}.json"
    extractor.save_report(report, report_path)

    print("\n✓ 提取完成！")


if __name__ == "__main__":
    main()
