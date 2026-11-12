"""Local, literal prompt checks using administrator-configured refusal terms."""

import re
import unicodedata


def _normalized(text: str) -> str:
    """Normalize case, compatibility forms, format characters and whitespace.

    Args:
        text: A prompt or configured refusal term.

    Returns:
        Comparable text without changing the original generation prompt.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    return " ".join(text.split())


def prompt_is_blocked(prompt: str, keywords: object, mode: str = "contains") -> bool:
    """Check the positive prompt against literal, nonempty refusal terms.

    Args:
        prompt: Parsed positive prompt, excluding command options.
        keywords: Standard configuration list; multiline text is also accepted.
        mode: Contains, exact prompt, or ASCII English word boundaries.

    Returns:
        Whether any normalized refusal term occurs in the normalized prompt.
        Empty or invalid configuration values do not add rules.
    """
    if isinstance(keywords, str):
        keywords = keywords.splitlines()
    if not isinstance(keywords, (list, tuple)):
        return False
    prompt = _normalized(prompt)
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        keyword = _normalized(keyword)
        if not keyword:
            continue
        if mode == "exact":
            matched = keyword == prompt
        elif mode == "english_word":
            matched = bool(
                re.search(
                    r"(?<![a-z0-9_])" + re.escape(keyword) + r"(?![a-z0-9_])", prompt
                )
            )
        else:
            matched = keyword in prompt
        if matched:
            return True
    return False
