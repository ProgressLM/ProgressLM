from __future__ import annotations
from typing import Dict, Any, List


VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of an ongoing task based on a visual demonstration of its progress.

The demonstration consists of a sequence of video frames (images) showing how the task evolves over time. Each frame represents a specific progress stage, ranging from 0% to 100%."""


VISUAL_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


VISUAL_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


VISUAL_DEMO_INSTRUCTION_PART3 = """Your task:
1. Analyze the demonstration images to understand how the task visually progresses from start to completion.
2. Identify the frame (or frames) from the demonstration that are visually most similar to the current state image.
3. Compare the current state to that reference frame and determine whether it shows more or less progress.
4. Finally, provide a numeric progress estimation between 0.00 and 1.00.

Your response must strictly follow this format:
<ref_think>Your reasoning for choosing the closest demonstration frame(s) as the reference</ref_think>
<ref>The progress score of your chosen reference frame(s)</ref>
<ref_think>Your reasoning for comparing the current state image with the reference frame(s)</score_think>
<score>Your final estimated progress score here</score>"""


def build_visual_demo_prompt(
    visual_demo_paths: List[str],
    stage_to_estimate_path: str,
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Build a three-part prompt for Visual Demo progress estimation task.

    Prompt structure:
    1. Text: "Here is the demonstration:"
    2. Images: visual_demo (N images, variable length)
    3. Text: "Here is the current state that you need to estimate:"
    4. Image: stage_to_estimate (1 image)
    5. Text: Task instructions

    Args:
        visual_demo_paths: List of paths to demonstration images (variable length)
        stage_to_estimate_path: Path to the current state image
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model in the format:
        [
            {"type": "text", "value": "Here is the demonstration:"},
            {"type": "image", "value": "img1.jpg", "min_pixels": ..., "max_pixels": ...},
            {"type": "image", "value": "img2.jpg", "min_pixels": ..., "max_pixels": ...},
            ...
            {"type": "text", "value": "Here is the current state..."},
            {"type": "image", "value": "current.jpg", "min_pixels": ..., "max_pixels": ...},
            {"type": "text", "value": "Your task: ..."}
        ]
    """
    msgs = []

    # Part 1: Demonstration introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART1})

    # Part 2: Visual demo images (variable length)
    for demo_img_path in visual_demo_paths:
        img_msg = {"type": "image", "value": demo_img_path}
        if min_pixels is not None:
            img_msg["min_pixels"] = min_pixels
        if max_pixels is not None:
            img_msg["max_pixels"] = max_pixels
        msgs.append(img_msg)

    # Part 3: Current state introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART2})

    # Part 4: Current state image (single image)
    stage_img_msg = {"type": "image", "value": stage_to_estimate_path}
    if min_pixels is not None:
        stage_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_msg["max_pixels"] = max_pixels
    msgs.append(stage_img_msg)

    # Part 5: Task instructions
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART3})

    return msgs


def build_visual_demo_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to build Visual Demo prompt from a dataset item.

    Args:
        item: Dataset item with 'visual_demo' and 'stage_to_estimate' fields
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model
    """
    visual_demo_paths = item['visual_demo']
    stage_to_estimate_path = item['stage_to_estimate'][0]

    return build_visual_demo_prompt(
        visual_demo_paths=visual_demo_paths,
        stage_to_estimate_path=stage_to_estimate_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
