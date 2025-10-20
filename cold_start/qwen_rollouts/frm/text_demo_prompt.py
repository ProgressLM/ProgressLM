from __future__ import annotations
from typing import Dict, Any, List


TEXT_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of an ongoing task based on a textual demonstration of its step-by-step progression.

The demonstration consists of a sequence of text instructions (text_demo), each describing one step of the process.
Each step explicitly states the corresponding progress value (ranging from 0.00 to 1.00), showing how the task evolves from start to completion."""


TEXT_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


TEXT_DEMO_INSTRUCTION_PART2 = """Here is the current state you need to evaluate:"""


TEXT_DEMO_INSTRUCTION_PART3 = """Your task:
1. Analyze the text_demo to understand how the task visually and conceptually progresses from start to completion.
2. Identify the step(s) from the text_demo that are most visually and semantically similar to the current state image.
3. Compare the current state image with the chosen reference step(s) to determine whether it represents an earlier or later stage.
4. Estimate the progress numerically as a floating-point value between 0.00 and 1.00.

Your response must strictly follow this format:
<ref_think>Your reasoning for choosing the most similar text_demo step(s) as the reference</ref_think>
<ref>The progress score of your chosen reference step(s)</ref>
<score_think>Your reasoning for comparing the current state image with the reference step(s)</score_think>
<score>Your final estimated progress score here</score>"""


def build_text_demo_prompt(
    text_demo: str,
    stage_to_estimate_path: str,
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Build a multi-part prompt for Text Demo progress estimation task.

    Prompt structure:
    1. Text: "Here is the demonstration:"
    2. Text: full text_demo content (all steps with progress values)
    3. Text: "Here is the current state that you need to estimate:"
    4. Image: stage_to_estimate (1 image)
    5. Text: Task instructions

    Args:
        text_demo: Full text demonstration with all steps and progress values
        stage_to_estimate_path: Path to the current state image
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model in the format:
        [
            {"type": "text", "value": "Here is the demonstration:"},
            {"type": "text", "value": "<full text_demo content>"},
            {"type": "text", "value": "Here is the current state..."},
            {"type": "image", "value": "current.jpg", "min_pixels": ..., "max_pixels": ...},
            {"type": "text", "value": "Your task: ..."}
        ]
    """
    msgs = []

    # Part 1: Demonstration introduction
    msgs.append({"type": "text", "value": TEXT_DEMO_INSTRUCTION_PART1})

    # Part 2: Full text_demo content
    msgs.append({"type": "text", "value": text_demo})

    # Part 3: Current state introduction
    msgs.append({"type": "text", "value": TEXT_DEMO_INSTRUCTION_PART2})

    # Part 4: Current state image (single image)
    stage_img_msg = {"type": "image", "value": stage_to_estimate_path}
    if min_pixels is not None:
        stage_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_msg["max_pixels"] = max_pixels
    msgs.append(stage_img_msg)

    # Part 5: Task instructions
    msgs.append({"type": "text", "value": TEXT_DEMO_INSTRUCTION_PART3})

    return msgs


def build_text_demo_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to build Text Demo prompt from a dataset item.

    Args:
        item: Dataset item with 'text_demo' and 'stage_to_estimate' fields
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model
    """
    text_demo = item['text_demo']
    stage_to_estimate_path = item['stage_to_estimate']

    return build_text_demo_prompt(
        text_demo=text_demo,
        stage_to_estimate_path=stage_to_estimate_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
