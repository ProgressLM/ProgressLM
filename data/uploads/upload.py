import os
from huggingface_hub import HfApi

# ======== 配置信息 ========
token = os.environ.get("HF_TOKEN")  # Set HF_TOKEN environment variable before running
repo_id = "Raymond-Qiancx/Prog-Data"  # ← 例如 "vcj9002/my-dataset"
folder_path = "/Users/cxqian/Codes/Plot_Lib/ProgressLM/codes/ProgressLM/final_data"  # 本地文件夹路径
target_path = ""  # 上传后在仓库中的路径（保持原文件夹名）
repo_type = "dataset"  # 可选值: "model" | "dataset" | "space"
# ===========================

api = HfApi()

api.upload_folder(
    folder_path=folder_path,
    path_in_repo=target_path,
    repo_id=repo_id,
    repo_type=repo_type,
    token=token,
)

print("✅ 文件夹上传成功！")
