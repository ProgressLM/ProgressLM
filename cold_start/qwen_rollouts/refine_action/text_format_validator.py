import re
from typing import Tuple, List, Dict, Any


def is_sample_format_valid(sample: dict) -> Tuple[bool, List[str]]:
    """
    检测单个样本是否符合预定义的步骤格式规范。

    要求样本包含字段:
      - 'id': 样本唯一标识（用于报错）
      - 'text_demo': 包含 Step 1:, Step 2:, ... 的文本
      - 'total_steps': 步骤总数（可为字符串或数字）

    格式规范:
      1. total_steps 必须为正整数
      2. 必须包含 Step 1 到 Step N（N = total_steps），不能跳步或重复
      3. 每个 Step i 后必须紧跟进度标记: "By now, our progress is X"
         其中 X 应等于 i / total_steps（支持 0.2, 0.20, 1.0, 1 等常见格式）
      4. 不能存在 Step (N+1) 或更高编号的步骤

    返回:
        (是否有效: bool, 错误信息列表: list[str])
    """
    errors = []

    # 获取必要字段
    sample_id = sample.get('id', 'unknown')
    text_demo = str(sample.get('text_demo', ''))
    total_steps_raw = sample.get('total_steps', 0)

    # 验证 total_steps
    try:
        total_steps = int(total_steps_raw)
    except (ValueError, TypeError):
        return False, [f"样本 {sample_id}: total_steps 无效（应为正整数）: {total_steps_raw}"]

    if total_steps <= 0:
        return False, [f"样本 {sample_id}: total_steps 必须 ≥ 1，当前值: {total_steps}"]

    # 检查每个步骤是否存在
    for step_num in range(1, total_steps + 1):
        step_marker = f"Step {step_num}:"
        if step_marker not in text_demo:
            errors.append(f"样本 {sample_id}: 缺少 {step_marker}")
            continue

        # 定位当前步骤的内容范围（到下一个 Step 或结尾）
        start = text_demo.find(step_marker)
        next_marker = f"Step {step_num + 1}:" if step_num < total_steps else None
        end = text_demo.find(next_marker) if next_marker else len(text_demo)
        if end == -1:
            end = len(text_demo)
        step_content = text_demo[start:end]

        # 计算期望进度值
        expected_progress = step_num / total_steps

        # 构造几种合法的进度字符串（覆盖常见浮点表示）
        valid_progress_strs = {
            f"By now, our progress is {expected_progress:.2f}".rstrip('0').rstrip('.'),  # 0.20 → 0.2, 1.00 → 1
            f"By now, our progress is {expected_progress:.1f}",                         # 0.2, 1.0
            f"By now, our progress is {expected_progress}",                             # 原始浮点（如 0.20000000000000001）
        }
        # 特殊处理整数情况（如 1.0 → 1）
        if expected_progress.is_integer():
            valid_progress_strs.add(f"By now, our progress is {int(expected_progress)}")

        # 检查是否包含任一合法进度标记
        if not any(prog_str in step_content for prog_str in valid_progress_strs):
            errors.append(
                f"样本 {sample_id}: {step_marker} 缺少正确的进度标记 "
                f"(期望包含类似 'By now, our progress is {expected_progress}')"
            )

    # 检查是否存在多余步骤
    extra_step = f"Step {total_steps + 1}:"
    if extra_step in text_demo:
        errors.append(f"样本 {sample_id}: 存在多余步骤 '{extra_step}'")

    return len(errors) == 0, errors


def validate_text_format(text_demo: str, total_steps: int, sample_id: str = "unknown") -> bool:
    """
    简化版格式验证函数，直接返回布尔值

    参数:
        text_demo: 文本演示内容
        total_steps: 总步骤数
        sample_id: 样本ID（用于日志）

    返回:
        是否格式有效
    """
    sample = {
        'id': sample_id,
        'text_demo': text_demo,
        'total_steps': total_steps
    }
    is_valid, errors = is_sample_format_valid(sample)
    return is_valid
