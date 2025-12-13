"""Reward shaping for ProgressLM visual progress estimation with n/a handling.

Updated to prevent reward hacking (ref=n/a + score=value strategy).
Key changes:
- GT has value but Pred=n/a: strong negative penalty (-0.8) to reduce FP rate
- Format reward includes consistency checks (n/a state + score_think)
- Weights: format=0.2, ref=0.5, score=0.3
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

NA_TOKEN = "n/a"

FORMAT_PATTERN = re.compile(
    r"\s*<ref_think>.*?</ref_think>\s*"
    r"<ref>\s*[^<]+\s*</ref>\s*"  # Accept any content, extract number later
    r"<score_think>\s*(?:.*?|n/a)\s*</score_think>\s*"
    r"<score>\s*(?:[0-9]+(?:\.[0-9]+)?%?|n/a)\s*</score>\s*$",
    re.DOTALL | re.IGNORECASE,
)
REF_PATTERN = re.compile(r"<ref>\s*([^<]+)\s*</ref>", re.IGNORECASE)
SCORE_PATTERN = re.compile(r"<score>\s*([^<]+)\s*</score>", re.IGNORECASE)
SCORE_THINK_PATTERN = re.compile(r"<score_think>\s*(.*?)\s*</score_think>", re.DOTALL | re.IGNORECASE)


def _load_ground_truth(raw: Any) -> Dict[str, Any]:
    """Normalize ground-truth payload into a dictionary."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ground-truth string must be valid JSON when using progresslm_reward."
            ) from exc
    raise TypeError("Ground-truth must be a dict or JSON string for progresslm_reward.")


def _extract_tag_value(pattern: re.Pattern[str], response: str) -> str | None:
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    return None


def _extract_score_think(response: str) -> str:
    """Extract score_think content from response."""
    match = SCORE_THINK_PATTERN.search(response)
    return match.group(1).strip() if match else ""


def _parse_ref_value(raw: Any) -> Tuple[bool, int | None]:
    """Return (is_na, value) for ref-like fields.

    Handles natural language like "The No. 8 text demo is the most relevant one"
    by extracting the first number found.
    """
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            return False, None
        return False, int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() == NA_TOKEN:
            return True, None
        # First try direct integer parsing
        try:
            return False, int(text)
        except ValueError:
            pass
        # Try to extract number from natural language (e.g., "No. 8", "Step 3")
        numbers = re.findall(r'\d+', text)
        if numbers:
            return False, int(numbers[0])
        # Try float parsing as fallback
        try:
            as_float = float(text)
            return (False, int(as_float)) if as_float.is_integer() else (False, None)
        except ValueError:
            pass
    return False, None


def _parse_score_value(raw: Any) -> Tuple[bool, float | None]:
    """Return (is_na, value) for score-like fields."""
    if isinstance(raw, (int, float)):
        return False, float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() == NA_TOKEN:
            return True, None
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            return False, float(text)
        except ValueError:
            pass
    return False, None


def _compute_format_reward(
    response: str,
    pred_ref_is_na: bool,
    pred_score_is_na: bool,
    gt_score_is_na: bool
) -> Tuple[float, float, float, float]:
    """
    Compute format reward with consistency checks.

    Returns:
        (format_reward, base_format, na_consistency, think_consistency)
    """
    # a) Base format check (0.4 weight)
    base_format = 1.0 if FORMAT_PATTERN.fullmatch(response.strip()) else 0.0

    # b) ref/score n/a state consistency (0.3 weight)
    # ref=n/a but score=value, or vice versa, is logically inconsistent
    na_consistency = 1.0 if (pred_ref_is_na == pred_score_is_na) else 0.0

    # c) score_think consistency (0.3 weight)
    score_think = _extract_score_think(response)
    is_think_na = score_think.lower() == NA_TOKEN

    if gt_score_is_na:
        # When GT=n/a, score_think must be n/a
        think_consistency = 1.0 if is_think_na else 0.0
    else:
        # When GT has value, score_think should have reasoning (not n/a)
        think_consistency = 0.0 if is_think_na else 1.0

    format_reward = 0.4 * base_format + 0.3 * na_consistency + 0.3 * think_consistency

    return format_reward, base_format, na_consistency, think_consistency


def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Compute format/ref/score rewards for a batch of responses.

    Returns a list of dictionaries. Each dictionary contains:
      - overall: weighted sum of component rewards (format=0.2, ref=0.5, score=0.3)
      - format: format reward with consistency checks [0, 1]
      - format_base: base XML format correctness
      - format_na_consistency: ref/score n/a state consistency
      - format_think_consistency: score_think n/a consistency
      - accuracy: alias for score reward (for logger compatibility)
      - score: score reward [-0.5, 1]
      - ref: reference-frame reward [-0.5, 1]
      - ref_error / score_error: diagnostic metrics for logging
      - hack_detected: 1.0 if ref=n/a but score=value (hack behavior)
    """
    scores: List[Dict[str, float]] = []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        gt_payload = _load_ground_truth(reward_input["ground_truth"])

        # Parse ground truth
        gt_ref_is_na, gt_ref_value = _parse_ref_value(gt_payload["ref"])
        demo_count = max(int(gt_payload.get("demo_count", 1)), 1)
        gt_score_is_na, gt_score_value = _parse_score_value(
            gt_payload["score_percent"]
        )

        # Parse predictions
        pred_ref_raw = _extract_tag_value(REF_PATTERN, response)
        pred_score_raw = _extract_tag_value(SCORE_PATTERN, response)

        pred_ref_is_na, pred_ref_value = _parse_ref_value(pred_ref_raw) if pred_ref_raw is not None else (False, None)
        pred_score_is_na, pred_score_value = _parse_score_value(pred_score_raw) if pred_score_raw is not None else (False, None)

        # Compute format reward with consistency checks
        format_reward, base_format, na_consistency, think_consistency = _compute_format_reward(
            response, pred_ref_is_na, pred_score_is_na, gt_score_is_na
        )

        # Detect hack behavior: ref=n/a but score=value
        hack_detected = 1.0 if (pred_ref_is_na and not pred_score_is_na) else 0.0

        # Initialize
        ref_error = 1.0
        ref_reward = 0.0
        score_error = 1.0
        score_reward = 0.0

        # Compute ref_reward
        if gt_ref_is_na:
            # GT is n/a (abnormal detection task)
            if pred_ref_is_na:
                ref_error = 0.0
                ref_reward = 0.8  # Correct abnormal detection
            else:
                ref_error = 1.0
                ref_reward = -0.3  # False positive (predicted value when should be n/a)
        else:
            # GT has value (normal task)
            if pred_ref_is_na:
                ref_error = 1.0
                ref_reward = -0.8  # Stronger penalty for FP (predicting n/a when GT has value)
            elif pred_ref_value is not None and gt_ref_value is not None:
                max_offset = max(demo_count - 1, 1)
                ref_error = min(abs(pred_ref_value - gt_ref_value) / max_offset, 1.0)
                ref_reward = max(1.0 - ref_error ** 2, 0.0)
            else:
                ref_error = 1.0
                ref_reward = 0.0  # Failed to parse

        # Compute score_reward
        if gt_score_is_na:
            # GT is n/a (abnormal detection task)
            if pred_score_is_na:
                score_error = 0.0
                score_reward = 0.8  # Correct abnormal detection
            else:
                score_error = 1.0
                score_reward = -0.3  # False positive
        else:
            # GT has value (normal task)
            if pred_score_is_na:
                score_error = 1.0
                score_reward = -0.8  # Stronger penalty for FP (predicting n/a when GT has value)
            elif pred_score_value is not None and gt_score_value is not None:
                score_error = min(
                    abs(pred_score_value - gt_score_value) / 100.0,
                    1.0,
                )
                score_reward = max(1.0 - score_error, 0.0)
            else:
                score_error = 1.0
                score_reward = 0.0  # Failed to parse

        # Overall reward with updated weights
        overall = (
            0.2 * format_reward
            + 0.5 * ref_reward
            + 0.3 * score_reward
        )

        scores.append(
            {
                "overall": overall,
                "format": format_reward,
                "format_base": base_format,
                "format_na_consistency": na_consistency,
                "format_think_consistency": think_consistency,
                "accuracy": score_reward,
                "score": score_reward,
                "ref": ref_reward,
                "ref_error": ref_error,
                "score_error": score_error,
                "hack_detected": hack_detected,
            }
        )

    return scores
