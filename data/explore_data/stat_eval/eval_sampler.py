import json
import pandas as pd
import numpy as np
from collections import defaultdict
import random

def stratified_sampling_trajectories(input_file, output_file, target_samples=3000, 
                                     stratify_by='data_source', random_seed=42):
    """
    对JSONL文件进行分层采样，以trajectory为单位进行采样
    
    参数:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        target_samples: 目标样本数量（约数）
        stratify_by: 分层依据 ('data_source', 'action_type', 'both')
        random_seed: 随机种子
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 读取数据
    print(f"{'='*80}")
    print(f"开始分层采样...")
    print(f"{'='*80}\n")
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 解析ID结构
    parsed_data = []
    for item in data:
        id_parts = item['id'].split('/')
        if len(id_parts) == 3:
            parsed_data.append({
                **item,
                'source_short': id_parts[0],
                'action_type': id_parts[1],
                'trajectory_id': id_parts[2]
            })
    
    df = pd.DataFrame(parsed_data)
    
    total_samples = len(df)
    total_trajectories = df['id'].nunique()
    target_ratio = target_samples / total_samples
    
    print(f"📊 原始数据统计:")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 总trajectory数: {total_trajectories}")
    print(f"  - 目标样本数: {target_samples}")
    print(f"  - 目标采样比例: {target_ratio:.2%}\n")
    
    # 计算每个trajectory的样本数
    traj_sample_counts = df.groupby('id').size().to_dict()
    
    # 根据分层依据进行采样
    if stratify_by == 'data_source':
        sampled_trajectories = _sample_by_single_column(
            df, 'data_source', target_ratio, traj_sample_counts
        )
    elif stratify_by == 'action_type':
        sampled_trajectories = _sample_by_single_column(
            df, 'action_type', target_ratio, traj_sample_counts
        )
    elif stratify_by == 'both':
        sampled_trajectories = _sample_by_both_columns(
            df, target_ratio, traj_sample_counts
        )
    else:
        raise ValueError(f"不支持的分层方式: {stratify_by}")
    
    # 筛选采样的数据
    sampled_df = df[df['id'].isin(sampled_trajectories)].copy()
    
    # 打印采样后的统计信息（在移除辅助列之前）
    print(f"\n{'='*80}")
    print(f"✅ 采样完成！")
    print(f"{'='*80}\n")
    
    print(f"📊 采样后数据统计:")
    print(f"  - 采样样本数: {len(sampled_df)}")
    print(f"  - 采样trajectory数: {sampled_df['id'].nunique()}")
    print(f"  - 实际采样比例: {len(sampled_df)/total_samples:.2%}")
    print(f"  - 达成率: {len(sampled_df)/target_samples:.2%}\n")
    
    # 对比原始分布和采样后分布（在移除辅助列之前）
    _compare_distributions(df, sampled_df, stratify_by)
    
    # 移除辅助列
    sampled_df = sampled_df.drop(['source_short', 'action_type', 'trajectory_id'], axis=1)
    
    # 保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in sampled_df.iterrows():
            json.dump(row.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\n💾 结果已保存到: {output_file}\n")
    
    return sampled_df


def _sample_by_single_column(df, column, target_ratio, traj_sample_counts):
    """按单列分层采样"""
    sampled_trajectories = set()
    
    print(f"📌 按 {column} 进行分层采样:\n")
    
    # 获取每个分层的trajectory列表
    groups = df.groupby(column)['id'].unique()
    
    for group_name, trajectories in groups.items():
        # 计算这个分层应该采样多少个trajectory
        group_total_samples = sum(traj_sample_counts[traj] for traj in trajectories)
        target_group_samples = int(group_total_samples * target_ratio)
        
        # 贪婪采样：按trajectory大小排序，尽可能接近目标
        traj_sizes = [(traj, traj_sample_counts[traj]) for traj in trajectories]
        selected = _greedy_sample_trajectories(traj_sizes, target_group_samples)
        
        sampled_trajectories.update(selected)
        
        actual_samples = sum(traj_sample_counts[traj] for traj in selected)
        print(f"  {group_name}:")
        print(f"    - 原始: {len(trajectories)} trajectories, {group_total_samples} samples")
        print(f"    - 目标: ~{target_group_samples} samples")
        print(f"    - 采样: {len(selected)} trajectories, {actual_samples} samples")
    
    return sampled_trajectories


def _sample_by_both_columns(df, target_ratio, traj_sample_counts):
    """按data_source和action_type两列进行分层采样"""
    sampled_trajectories = set()
    
    print(f"📌 按 data_source × action_type 进行分层采样:\n")
    
    # 获取每个组合的trajectory列表
    groups = df.groupby(['data_source', 'action_type'])['id'].unique()
    
    for (data_source, action_type), trajectories in groups.items():
        # 计算这个分层应该采样多少个样本
        group_total_samples = sum(traj_sample_counts[traj] for traj in trajectories)
        target_group_samples = int(group_total_samples * target_ratio)
        
        # 贪婪采样
        traj_sizes = [(traj, traj_sample_counts[traj]) for traj in trajectories]
        selected = _greedy_sample_trajectories(traj_sizes, target_group_samples)
        
        sampled_trajectories.update(selected)
        
        actual_samples = sum(traj_sample_counts[traj] for traj in selected)
        print(f"  {data_source} × {action_type}:")
        print(f"    原始: {len(trajectories)} traj, {group_total_samples} samples → "
              f"采样: {len(selected)} traj, {actual_samples} samples")
    
    return sampled_trajectories


def _greedy_sample_trajectories(traj_sizes, target_samples):
    """
    贪婪算法选择trajectory，使总样本数尽可能接近目标
    
    策略：
    1. 随机打乱trajectory顺序
    2. 按顺序添加trajectory，直到接近或超过目标
    3. 如果超过太多，尝试去掉最后一个，看哪个更接近目标
    """
    random.shuffle(traj_sizes)
    
    selected = []
    current_samples = 0
    
    for traj, size in traj_sizes:
        if current_samples + size <= target_samples * 1.2:  # 允许20%的超出
            selected.append(traj)
            current_samples += size
        elif current_samples < target_samples * 0.8:  # 如果还差很多，继续添加
            selected.append(traj)
            current_samples += size
    
    # 如果没有选中任何trajectory，至少选一个
    if not selected and traj_sizes:
        selected.append(traj_sizes[0][0])
    
    return selected


def _compare_distributions(original_df, sampled_df, stratify_by):
    """对比原始分布和采样后分布"""
    print(f"\n{'='*80}")
    print(f"📊 分布对比")
    print(f"{'='*80}\n")
    
    # 按data_source对比
    print("按 data_source 的分布对比:")
    print(f"{'':30} {'原始':<20} {'采样后':<20} {'差异':<15}")
    print(f"{'-'*85}")
    
    orig_dist = original_df.groupby('data_source').size()
    samp_dist = sampled_df.groupby('data_source').size()
    
    for source in orig_dist.index:
        orig_count = orig_dist[source]
        samp_count = samp_dist.get(source, 0)
        orig_ratio = orig_count / len(original_df)
        samp_ratio = samp_count / len(sampled_df)
        diff = samp_ratio - orig_ratio
        
        print(f"{source:30} "
              f"{orig_count:6d} ({orig_ratio:6.2%})   "
              f"{samp_count:6d} ({samp_ratio:6.2%})   "
              f"{diff:+.2%}")
    
    # 如果需要，也对比action_type
    if stratify_by in ['action_type', 'both']:
        print(f"\n按 action_type 的分布对比 (前10个):")
        print(f"{'':30} {'原始':<20} {'采样后':<20} {'差异':<15}")
        print(f"{'-'*85}")
        
        orig_dist_action = original_df.groupby('action_type').size().sort_values(ascending=False)
        samp_dist_action = sampled_df.groupby('action_type').size()
        
        for i, action in enumerate(orig_dist_action.head(10).index):
            orig_count = orig_dist_action[action]
            samp_count = samp_dist_action.get(action, 0)
            orig_ratio = orig_count / len(original_df)
            samp_ratio = samp_count / len(sampled_df) if len(sampled_df) > 0 else 0
            diff = samp_ratio - orig_ratio
            
            print(f"{action:30} "
                  f"{orig_count:6d} ({orig_ratio:6.2%})   "
                  f"{samp_count:6d} ({samp_ratio:6.2%})   "
                  f"{diff:+.2%}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python sample_dataset.py <input_file> <output_file> [target_samples] [stratify_by] [random_seed]")
        print("\n参数说明:")
        print("  input_file     : 输入JSONL文件路径")
        print("  output_file    : 输出JSONL文件路径")
        print("  target_samples : 目标样本数 (默认: 3000)")
        print("  stratify_by    : 分层方式 - 'data_source', 'action_type', 'both' (默认: 'data_source')")
        print("  random_seed    : 随机种子 (默认: 42)")
        print("\n示例:")
        print("  python sample_dataset.py data.jsonl sampled_data.jsonl 3000 data_source 42")
        print("  python sample_dataset.py data.jsonl sampled_data.jsonl 3000 both")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    stratify_by = sys.argv[4] if len(sys.argv) > 4 else 'data_source'
    random_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    
    stratified_sampling_trajectories(
        input_file, 
        output_file, 
        target_samples=target_samples,
        stratify_by=stratify_by,
        random_seed=random_seed
    )

# python /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/explore_data/stat_eval/eval_sampler.py /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_eval_all.jsonl /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_eval_3k.jsonl 3000 both 42