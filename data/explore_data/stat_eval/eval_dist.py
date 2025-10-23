import json
import pandas as pd
from collections import defaultdict
import numpy as np

def analyze_jsonl(file_path):
    """
    分析JSONL文件，统计trajectory和sample的各种指标
    """
    # 读取数据
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"{'='*80}")
    print(f"数据集统计分析报告")
    print(f"{'='*80}\n")
    
    # 基本统计
    total_samples = len(data)
    print(f"📊 总样本数: {total_samples}\n")
    
    # 解析ID结构
    parsed_data = []
    for item in data:
        id_parts = item['id'].split('/')
        if len(id_parts) == 3:
            parsed_data.append({
                'id': item['id'],
                'source_short': id_parts[0],
                'action_type': id_parts[1],
                'trajectory_id': id_parts[2],
                'data_source': item['data_source']
            })
    
    df = pd.DataFrame(parsed_data)
    
    # 1. 统计每个data_source的trajectory数量
    print(f"{'='*80}")
    print(f"1. 每个data_source的trajectory数量")
    print(f"{'='*80}")
    
    trajectories_per_source = df.groupby('data_source')['id'].nunique()
    for source, count in trajectories_per_source.items():
        print(f"  - {source}: {count} 个trajectories")
    print()
    
    # 2. 统计每个data_source的样本数量
    print(f"{'='*80}")
    print(f"2. 每个data_source的样本数量")
    print(f"{'='*80}")
    
    samples_per_source = df.groupby('data_source').size()
    for source, count in samples_per_source.items():
        print(f"  - {source}: {count} 个samples")
    print()
    
    # 3. 计算每个trajectory的sample数量，并进行五值统计
    print(f"{'='*80}")
    print(f"3. 每个trajectory的sample数量统计")
    print(f"{'='*80}")
    
    samples_per_trajectory = df.groupby(['data_source', 'id']).size()
    
    # 按data_source分组进行五值统计
    for source in df['data_source'].unique():
        source_df = df[df['data_source'] == source]
        samples_per_traj = source_df.groupby('id').size()
        
        print(f"\n  {source}:")
        print(f"    - Trajectory数量: {len(samples_per_traj)}")
        print(f"    - 样本数量: {len(source_df)}")
        print(f"    - 平均每个trajectory的样本数: {samples_per_traj.mean():.2f}")
        print(f"    - 五值统计:")
        print(f"      • 最小值: {samples_per_traj.min()}")
        print(f"      • 第一四分位数(Q1): {samples_per_traj.quantile(0.25):.2f}")
        print(f"      • 中位数(Median): {samples_per_traj.median():.2f}")
        print(f"      • 第三四分位数(Q3): {samples_per_traj.quantile(0.75):.2f}")
        print(f"      • 最大值: {samples_per_traj.max()}")
    
    # 4. 统计source简称
    print(f"\n{'='*80}")
    print(f"4. Source简称统计")
    print(f"{'='*80}")
    
    source_short_counts = df.groupby('source_short')['id'].nunique()
    for source_short, count in source_short_counts.items():
        print(f"  - {source_short}: {count} 个trajectories")
    print()
    
    # 5. 统计动作类型
    print(f"{'='*80}")
    print(f"5. 动作类型(Action Type)统计")
    print(f"{'='*80}")
    
    action_type_counts = df.groupby('action_type')['id'].nunique()
    print(f"  总共 {len(action_type_counts)} 种动作类型:\n")
    for action_type, count in sorted(action_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {action_type}: {count} 个trajectories")
    print()
    
    # 6. 统计每个source_short下的动作类型
    print(f"{'='*80}")
    print(f"6. 每个source简称下的动作类型统计")
    print(f"{'='*80}")
    
    for source_short in df['source_short'].unique():
        source_df = df[df['source_short'] == source_short]
        action_types = source_df.groupby('action_type')['id'].nunique()
        print(f"\n  {source_short} ({len(action_types)} 种动作):")
        for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {action_type}: {count} 个trajectories")
    
    # 7. 交叉统计：source_short × action_type
    print(f"\n{'='*80}")
    print(f"7. Source简称 × 动作类型 交叉统计表")
    print(f"{'='*80}\n")
    
    cross_tab = pd.crosstab(df['source_short'], df['action_type'], 
                            values=df['id'], aggfunc='nunique', margins=True)
    print(cross_tab)
    print()
    
    # 8. 总体五值统计
    print(f"{'='*80}")
    print(f"8. 全局统计：每个trajectory的样本数")
    print(f"{'='*80}")
    
    all_samples_per_traj = df.groupby('id').size()
    print(f"  - 总trajectory数: {len(all_samples_per_traj)}")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 平均每个trajectory的样本数: {all_samples_per_traj.mean():.2f}")
    print(f"  - 五值统计:")
    print(f"    • 最小值: {all_samples_per_traj.min()}")
    print(f"    • 第一四分位数(Q1): {all_samples_per_traj.quantile(0.25):.2f}")
    print(f"    • 中位数(Median): {all_samples_per_traj.median():.2f}")
    print(f"    • 第三四分位数(Q3): {all_samples_per_traj.quantile(0.75):.2f}")
    print(f"    • 最大值: {all_samples_per_traj.max()}")
    
    print(f"\n{'='*80}")
    print(f"分析完成！")
    print(f"{'='*80}\n")
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python analyze_jsonl.py <jsonl_file_path>")
        print("示例: python analyze_jsonl.py data.jsonl")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = analyze_jsonl(file_path)

# python /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/explore_data/stat_eval/eval_dist.py /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_eval_all.jsonl