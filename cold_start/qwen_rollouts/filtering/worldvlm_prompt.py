from __future__ import annotations
from typing import Dict, Any, List


WORLDVLM_CONSISTENCY_SYSTEM_PROMPT = """You are a visual-language judge for an embodied world-model dataset.
You will receive an object with fields like:

* `"start_img"`: the initial state image (URL, path, or bytes)
* `"end_img"`: the resulting state image (URL, path, or bytes)
* `"action"`: a short textual description of the action that transforms the start state into the end state

Your task: determine whether the `"action"` description is **semantically and causally consistent** with the transition from `"start_img"` to `"end_img"`, and whether this sample is **suitable** for training a multimodal world model (i.e., clear, single-step/atomic, unambiguous, visually verifiable).

Decision criteria (all must be satisfied to accept):

1. Both images are present, viewable, and depict the **same scene/objects** across time.
2. The `"action"` is **plausible** and **visibly supported** by the change from start to end (cause → effect).
3. The action is **atomic** (single main action), **specific**, and **unambiguous**.
4. Key entities referenced in the action are **visible and identifiable** in both images as appropriate.
5. Images are sufficiently clear (not corrupted/blank) and the transition is not due to unrelated factors (e.g., camera jump, viewpoint-only change without action).

If any criterion fails (including missing/empty fields), reject.

Output format: respond with exactly one word in lowercase:

* `"yes"` if all criteria are met,
* `"no"` otherwise.

Do **not** include any explanations, punctuation, or extra text."""


class WorldVLMPromptMixin:
    """
    Mixin class for building prompts for WorldVLM consistency filtering.

    This mixin provides methods to build custom prompts with dual images and action text.
    It's designed to work with the Qwen2VLChat model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_worldvlm_consistency_prompt(
        self,
        start_img_path: str,
        end_img_path: str,
        action_text: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None
    ) -> List[Dict[str, Any]]:
        """
        Build a prompt for WorldVLM consistency filtering task.

        Format follows Qwen2-VL multi-image input:
        [
            {"type": "image", "value": start_img_path, "min_pixels": ..., "max_pixels": ...},
            {"type": "image", "value": end_img_path, "min_pixels": ..., "max_pixels": ...},
            {"type": "text", "value": action_text}
        ]

        Args:
            start_img_path: Path to the start state image
            end_img_path: Path to the end state image
            action_text: The action description text
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing

        Returns:
            List of message dicts for the model
        """
        msgs = []

        # Add start image
        start_img_msg = {"type": "image", "value": start_img_path}
        if min_pixels is not None:
            start_img_msg["min_pixels"] = min_pixels
        if max_pixels is not None:
            start_img_msg["max_pixels"] = max_pixels
        msgs.append(start_img_msg)

        # Add end image
        end_img_msg = {"type": "image", "value": end_img_path}
        if min_pixels is not None:
            end_img_msg["min_pixels"] = min_pixels
        if max_pixels is not None:
            end_img_msg["max_pixels"] = max_pixels
        msgs.append(end_img_msg)

        # Add action text as query
        action_query = f"Action: {action_text}\n\nIs this action consistent with the transition from the first image to the second image?"
        msgs.append({"type": "text", "value": action_query})

        return msgs


def build_worldvlm_prompt_from_item(
    item: Dict[str, Any],
    min_pixels: int | None = None,
    max_pixels: int | None = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to build WorldVLM prompt from a dataset item.

    Args:
        item: Dataset item with 'start_img', 'end_img', and 'action' fields
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        List of message dicts for the model
    """
    start_img = item['start_img']
    end_img = item['end_img']
    action = item['action']

    msgs = []

    # Add start image
    start_img_msg = {"type": "image", "value": start_img}
    if min_pixels is not None:
        start_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        start_img_msg["max_pixels"] = max_pixels
    msgs.append(start_img_msg)

    # Add end image
    end_img_msg = {"type": "image", "value": end_img}
    if min_pixels is not None:
        end_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        end_img_msg["max_pixels"] = max_pixels
    msgs.append(end_img_msg)

    # Add action text as query
    action_query = f"Action: {action}\n\nIs this action consistent with the transition from the first image to the second image?"
    msgs.append({"type": "text", "value": action_query})

    return msgs
