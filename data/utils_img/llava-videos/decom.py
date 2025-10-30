import os
import sys
import tarfile
from tqdm import tqdm

def extract_targz_with_progress(tar_path, output_dir):
    """
    解压单个 tar.gz 文件并显示进度条
    """
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getmembers()
            total = len(members)
            print(f"\n📦 解压中: {os.path.basename(tar_path)} -> {output_dir}")
            for member in tqdm(members, total=total, desc="   进度", unit="文件"):
                tar.extract(member, path=output_dir)
        print(f"✅ 完成: {os.path.basename(tar_path)}\n")
    except Exception as e:
        print(f"❌ 解压失败: {tar_path} - {e}")

def extract_all_targz(source_dir, target_dir):
    """
    扫描 source_dir 下的所有 .tar.gz 文件，并解压到 target_dir
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    targz_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.endswith(".tar.gz"):
                targz_files.append(os.path.join(root, f))

    if not targz_files:
        print("⚠️ 没有找到任何 .tar.gz 文件。")
        return

    print(f"✅ 找到 {len(targz_files)} 个压缩包，开始解压...\n")

    for tar_path in targz_files:
        subfolder = os.path.splitext(os.path.splitext(os.path.basename(tar_path))[0])[0]
        output_subdir = os.path.join(target_dir, subfolder)

        if os.path.exists(output_subdir) and os.listdir(output_subdir):
            print(f"⏩ 跳过已解压文件夹: {subfolder}")
            continue

        os.makedirs(output_subdir, exist_ok=True)
        extract_targz_with_progress(tar_path, output_subdir)

    print("\n🎉 所有文件已成功解压到:", target_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ 用法: python extract_all_targz.py <源目录> [目标目录]")
        sys.exit(1)

    SOURCE_DIR = sys.argv[1]
    TARGET_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "extracted")

    print(f"📁 源目录: {SOURCE_DIR}")
    print(f"📂 输出目录: {TARGET_DIR}")

    extract_all_targz(SOURCE_DIR, TARGET_DIR)


# python /home/vcj9002/jianshu/workspace/code/ProgressLM/data/utils_img/llava-videos/decom.py /home/vcj9002/jianshu/workspace/data/llava-video/30_60_s_academic_v0_1 /home/vcj9002/jianshu/workspace/data/llava-video/30_60_s_academic_v0_1
# python /home/vcj9002/jianshu/workspace/code/ProgressLM/data/utils_img/llava-videos/decom.py /home/vcj9002/jianshu/workspace/data/llava-video/0_30_s_activitynetqa /home/vcj9002/jianshu/workspace/data/llava-video/0_30_s_activitynetqa
# python /home/vcj9002/jianshu/workspace/code/ProgressLM/data/utils_img/llava-videos/decom.py /home/vcj9002/jianshu/workspace/data/llava-video/0_30_s_activitynetqa /home/vcj9002/jianshu/workspace/data/llava-video/0_30_s_activitynetqa
# python /home/vcj9002/jianshu/workspace/code/ProgressLM/data/utils_img/llava-videos/decom.py /home/vcj9002/jianshu/workspace/data/llava-video/0_30_s_perceptiontest /home/vcj9002/jianshu/workspace/data/llava-video/0_30_s_perceptiontest