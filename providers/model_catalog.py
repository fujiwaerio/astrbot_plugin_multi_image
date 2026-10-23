"""Recognize image output without submitting a generation request.

Explicit output metadata takes precedence over names. Many relays only expose
IDs, so recognizable image families remain a fallback; unknown aliases can still
be entered manually. Image inputs and tool calling alone do not imply output.
"""

from __future__ import annotations

import re

_IMAGE_TASKS = {
    "image-generation",
    "image-editing",
    "text-to-image",
    "image-to-image",
    "text2image",
    "image2image",
    "images/generations",
    "images/edits",
}
_OTHER_TASK = re.compile(
    r"(?:^|[-_/.])(?:video|vision|embedding|embeddings|caption|captioning|"
    r"ocr|clip|siglip|classification|classifier|recognition|understanding|"
    r"retrieval|search|rerank|reranker)(?:$|[-_/.])|"
    r"image[-_]to[-_]text|image[-_]text[-_]to[-_]text",
    re.IGNORECASE,
)
_IMAGE_NAME = re.compile(
    r"(?:^|[-_/.])(?:image|imagen\d*|flux|dall[-_]?e|seedream|seededit|"
    r"stable[-_]diffusion|sdxl|sd[.-]?3(?:[.-]5)?|qwen[-_]image|"
    r"hunyuan[-_]image|kolors|ideogram|recraft|cogview|jimeng|midjourney|"
    r"nano[-_]banana)(?:$|[-_/.\d])|"
    r"(?:^|[-_/.])grok[-_]imagine(?:[-_]1[.-]0)?$",
    re.IGNORECASE,
)


def _strings(value: object) -> set[str]:
    """Only accept bounded, structured metadata; never scan descriptions."""
    if not isinstance(value, list):
        return set()
    return {
        item.strip().lower().replace("_", "-")
        for item in value[:100]
        if isinstance(item, str) and len(item) <= 100
    }


def is_image_model(name: str, metadata: object = None) -> bool:
    """Return image candidates, respecting explicit output modality metadata."""
    if isinstance(metadata, dict):
        architecture = metadata.get("architecture")
        for source in (metadata, architecture):
            if isinstance(source, dict):
                outputs = _strings(source.get("output_modalities"))
                if outputs:
                    return bool(outputs & {"image", "images"})
        # Explicit capability flags override a suggestive name, including False.
        for key in ("supports_image_generation", "image_generation"):
            flag = metadata.get(key)
            if isinstance(flag, bool):
                return flag
        capabilities = metadata.get("capabilities")
        if isinstance(capabilities, dict):
            for key in ("image_generation", "image_editing"):
                if capabilities.get(key) is True:
                    return True
            if capabilities.get("image_generation") is False:
                return False
        if _strings(capabilities) & _IMAGE_TASKS:
            return True
        endpoints = _strings(metadata.get("supported_endpoint_types"))
        if any(
            endpoint.removeprefix("/v1/").lstrip("/") in _IMAGE_TASKS
            for endpoint in endpoints
        ):
            return True
    # A model that merely accepts images, or produces video, is not a candidate.
    return not _OTHER_TASK.search(name) and bool(_IMAGE_NAME.search(name))
