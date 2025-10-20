from typing import Dict, Any, List


# System prompt - Defines the model's role and task
CLEAN_TEXT_SYSTEM_PROMPT = """You are an expert text editor. Your task is to read and polish text demonstrations to make them grammatically correct, fluent, clear, and easy to read.

You must preserve:
1. All step labels (e.g., Step 1:, Step 2:, etc.)
2. All progress markers (e.g., "By now, our progress is 0.12.")
3. The original structure and order of steps
4. All factual content and instructions

You must improve:
1. Grammar and spelling
2. Sentence fluency and clarity
3. Logical coherence
4. Readability

Output only the cleaned text without any commentary or metadata."""


# User prompt template
CLEAN_TEXT_USER_PROMPT_TEMPLATE = """You are an expert text editor. Read the value of "text_demo" from the given JSON data. Your task is to rewrite and polish the text to make it:
1. Grammatically correct and fluent in English.
2. Clear, concise, and coherent.
3. Easy to read and logically structured.

Requirements:
1. Preserve the original structure, including each step label (e.g., Step 1:, Step 2:) and the progress markers (e.g., By now, our progress is 0.11.).
2. Keep all factual content and instructions accurate.
3. Do not change the order of steps or remove any important details.
4. Simplify overly long sentences and fix awkward phrasing.

Output only the cleaned and improved text, without adding commentary or metadata.

Here is the text_demo to clean:

{text_demo}"""


def build_clean_text_prompt(text_demo: str) -> List[Dict[str, Any]]:
    """
    Build prompt for text cleaning task.

    Args:
        text_demo: Original text demonstration content

    Returns:
        Message list in format: [{"type": "text", "value": "..."}]
    """
    # Format user prompt
    user_prompt = CLEAN_TEXT_USER_PROMPT_TEMPLATE.format(text_demo=text_demo)

    # Return pure text message (no images)
    return [
        {"type": "text", "value": user_prompt}
    ]


def build_clean_text_prompt_from_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build prompt from a dataset item.

    Args:
        item: Dataset item with 'text_demo' field

    Returns:
        Message list
    """
    return build_clean_text_prompt(item['text_demo'])
