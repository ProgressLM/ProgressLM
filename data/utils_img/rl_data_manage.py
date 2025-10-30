import os
import json

# 目标文件夹路径
folder = "/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/train/rl"

# 输出文件名
output_file = os.path.join(folder, "rl_all.jsonl")

# 打开输出文件
with open(output_file, "w", encoding="utf-8") as outfile:
    # 遍历该文件夹下所有文件
    for filename in os.listdir(folder):
        if filename.endswith(".jsonl") and filename != "rl_all.jsonl":
            file_path = os.path.join(folder, filename)
            print(f"正在合并: {file_path}")
            # 逐行读取并写入输出文件
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

print(f"✅ 已完成合并！输出文件：{output_file}")
