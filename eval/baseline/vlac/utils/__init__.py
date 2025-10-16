# VLAC Utils Module
# 导入原始VLAC项目的核心工具类，并添加多GPU推理支持

from .model_utils import GAC_model
from . import data_processing_vlm
from . import video_tool
from . import magic_detect
from . import multi_gpu_utils

__all__ = ["GAC_model", "data_processing_vlm", "video_tool", "magic_detect", "multi_gpu_utils"]
