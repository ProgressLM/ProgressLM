import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==================== 配置参数 ====================
BASE_DATA_PATH = "/projects/p32958/chengxuan/data/robomind_fail/failure_data"
OUTPUT_BASE_PATH = "/projects/p32958/chengxuan/data/images/failures"
NUM_WORKERS = 48  # 线程数，可根据CPU核心数调整

# ==================== 核心函数 ====================

def decode_image(compressed_image):
    """解码压缩的图像数据"""
    if compressed_image is None or len(compressed_image) == 0:
        return None
    
    # 使用OpenCV解码
    img = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    if img is not None:
        # 转换BGR到RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def parse_hdf5_path(hdf5_path):
    """从HDF5路径解析：动作种类、原因、trajectory_id"""
    # 路径格式: .../failure_data/{动作种类}/{原因}/{trajectory_id}/data/trajectory.hdf5
    parts = hdf5_path.parts
    
    # 找到failure_data的索引
    try:
        failure_idx = parts.index('failure_data')
        action_type = parts[failure_idx + 1]  # 动作种类
        reason = parts[failure_idx + 2]       # 原因
        trajectory_id = parts[failure_idx + 3] # trajectory_id
        return action_type, reason, trajectory_id
    except (ValueError, IndexError):
        return None, None, None

def extract_images_from_hdf5(hdf5_path, output_base):
    """从单个HDF5文件提取所有RGB图像（线程安全版本）"""
    # 解析路径信息
    action_type, reason, trajectory_id = parse_hdf5_path(hdf5_path)
    if not all([action_type, reason, trajectory_id]):
        return {'error': f'路径解析失败: {hdf5_path}'}
    
    stats = {'action_type': action_type, 'reason': reason, 'trajectory_id': trajectory_id, 
             'cameras': {}, 'errors': []}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查数据路径
            if 'observations' not in f or 'rgb_images' not in f['observations']:
                stats['errors'].append('未找到observations/rgb_images路径')
                return stats
            
            rgb_images_group = f['observations']['rgb_images']
            
            # 遍历所有相机（camera_chest, camera_head等）
            for camera_name in rgb_images_group.keys():
                camera_data = rgb_images_group[camera_name]
                
                # 创建输出目录: {原因}/{动作种类}/{trajectory_id}/{camera_name}/
                output_dir = output_base / reason / action_type / trajectory_id / camera_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                saved_count = 0
                failed_count = 0
                
                # 提取每一帧图像
                for idx in range(len(camera_data)):
                    try:
                        compressed_img = camera_data[idx]
                        img = decode_image(compressed_img)
                        
                        if img is not None:
                            # 保持原始帧编号格式: frame_XXXX.jpg
                            frame_filename = f'frame_{idx:04d}.jpg'
                            output_path = output_dir / frame_filename
                            
                            # 转换回BGR用于保存
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(output_path), img_bgr)
                            saved_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        if failed_count <= 3:  # 只记录前3个错误
                            stats['errors'].append(f'{camera_name}[{idx}]: {str(e)}')
                
                stats['cameras'][camera_name] = {'saved': saved_count, 'failed': failed_count}
    
    except Exception as e:
        stats['errors'].append(f'HDF5读取错误: {str(e)}')
    
    return stats

def process_all_hdf5_files(base_path, output_base, num_workers=8):
    """多线程处理所有HDF5文件"""
    base_path = Path(base_path)
    output_base = Path(output_base)
    
    # 查找所有trajectory.hdf5文件
    print(f"🔍 扫描目录: {base_path}")
    hdf5_files = list(base_path.rglob('*/data/trajectory.hdf5'))
    print(f"📊 找到 {len(hdf5_files)} 个HDF5文件")
    print(f"🚀 使用 {num_workers} 个线程并行处理\n")
    
    if len(hdf5_files) == 0:
        print("❌ 未找到任何HDF5文件！请检查路径是否正确。")
        return
    
    # 统计信息（使用锁保证线程安全）
    stats_lock = threading.Lock()
    total_stats = defaultdict(int)
    all_errors = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_path = {
            executor.submit(extract_images_from_hdf5, hdf5_path, output_base): hdf5_path 
            for hdf5_path in hdf5_files
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(hdf5_files), desc="提取图像", unit="文件") as pbar:
            for future in as_completed(future_to_path):
                hdf5_path = future_to_path[future]
                
                try:
                    stats = future.result()
                    
                    # 线程安全地更新统计
                    with stats_lock:
                        if 'error' in stats:
                            total_stats['failed_files'] += 1
                            all_errors.append(stats['error'])
                        else:
                            total_stats['success_files'] += 1
                            for camera, counts in stats['cameras'].items():
                                total_stats[f'{camera}_saved'] += counts['saved']
                                total_stats[f'{camera}_failed'] += counts['failed']
                            
                            if stats['errors']:
                                all_errors.extend([f"{stats['trajectory_id']}: {err}" for err in stats['errors']])
                        
                        # 更新进度条
                        pbar.set_postfix({
                            '成功': total_stats['success_files'],
                            '失败': total_stats['failed_files']
                        })
                
                except Exception as e:
                    with stats_lock:
                        total_stats['failed_files'] += 1
                        all_errors.append(f'{hdf5_path}: {str(e)}')
                        pbar.set_postfix({
                            '成功': total_stats['success_files'],
                            '失败': total_stats['failed_files']
                        })
                
                pbar.update(1)
    
    # 输出最终统计
    print("\n" + "="*80)
    print("✅ 处理完成！")
    print("="*80)
    print(f"\n📁 文件统计:")
    print(f"   成功处理: {total_stats['success_files']} 个文件")
    print(f"   处理失败: {total_stats['failed_files']} 个文件")
    
    print(f"\n🎥 图像统计:")
    camera_names = set()
    for key in total_stats.keys():
        if key.endswith('_saved'):
            camera_names.add(key.replace('_saved', ''))
    
    for camera in sorted(camera_names):
        saved = total_stats.get(f'{camera}_saved', 0)
        failed = total_stats.get(f'{camera}_failed', 0)
        print(f"   {camera}:")
        print(f"      ✓ 成功保存: {saved:,} 张")
        if failed > 0:
            print(f"      ✗ 解码失败: {failed:,} 张")
    
    # 显示错误（如果有）
    if all_errors:
        print(f"\n⚠️  遇到 {len(all_errors)} 个错误")
        print(f"   显示前10个错误:")
        for i, error in enumerate(all_errors[:10], 1):
            print(f"   {i}. {error}")
        if len(all_errors) > 10:
            print(f"   ... 还有 {len(all_errors) - 10} 个错误未显示")
    
    print(f"\n💾 输出目录: {output_base}")
    print("="*80)

# ==================== 执行处理 ====================

if __name__ == "__main__":
    process_all_hdf5_files(BASE_DATA_PATH, OUTPUT_BASE_PATH, NUM_WORKERS)
