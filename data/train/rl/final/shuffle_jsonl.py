#!/usr/bin/env python3
"""
随机打乱JSONL文件的行顺序
"""
import json
import random
import argparse
from pathlib import Path


def shuffle_jsonl(input_file, output_file=None, seed=None):
    """
    随机打乱JSONL文件的行顺序

    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径，如果为None则自动生成
        seed: 随机种子，用于复现结果
    """
    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    input_path = Path(input_file)

    # 如果没有指定输出文件，则自动生成文件名
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_shuffled{input_path.suffix}"
    else:
        output_file = Path(output_file)

    print(f"读取文件: {input_file}")

    # 读取所有行
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                lines.append(line)

    print(f"总共读取了 {len(lines)} 行")

    # 随机打乱
    random.shuffle(lines)
    print(f"已随机打乱顺序")

    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='随机打乱JSONL文件的行顺序')
    parser.add_argument('input_file', help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output', help='输出的JSONL文件路径（可选）')
    parser.add_argument('-s', '--seed', type=int, help='随机种子（可选，用于复现结果）')

    args = parser.parse_args()

    shuffle_jsonl(args.input_file, args.output, args.seed)


if __name__ == '__main__':
    main()
