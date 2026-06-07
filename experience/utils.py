"""Experience-local I/O, debug, image, and prompt helpers."""

from __future__ import annotations

import base64
import io
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from PIL import Image

from .schema import (
    ExperienceLibrary,
    _load_experience_library_payload,
    _serialize_experience_library,
    format_experience_for_prompt,
    normalize_experience_library,
)


EXPERIENCE_LIBRARY_LOCK = threading.RLock()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
METADATA_FIELDS = [
    "data_source",
    "task",
    "domain",
    "subdomain",
    "category",
    "subtask",
    "prompt_category",
    "eval_focus",
    "difficulty",
    "level",
]


def _clean_tag_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def sample_metadata_tags(sample: dict[str, Any] | None) -> List[str]:
    """Return stable tags like domain_geography or category_sports."""
    if not isinstance(sample, dict):
        return []

    tags: List[str] = []
    seen = set()
    for field in METADATA_FIELDS:
        value = sample.get(field)
        if value in (None, ""):
            continue
        cleaned = _clean_tag_value(value)
        if not cleaned:
            continue
        tag = f"{field}_{cleaned}"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def append_sample_metadata_tags(
    task_tags: Iterable[Any] | None,
    sample: dict[str, Any] | None,
) -> List[str]:
    """Append dataset metadata tags while preserving existing LLM task tags."""
    tags: List[str] = []
    seen = set()
    for tag in task_tags or []:
        text = str(tag).strip()
        if text and text not in seen:
            tags.append(text)
            seen.add(text)
    for tag in sample_metadata_tags(sample):
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def sample_metadata_context(sample: dict[str, Any] | None) -> str:
    tags = sample_metadata_tags(sample)
    return "Task metadata: " + ", ".join(tags) if tags else ""


def _image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a JPEG base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _find_original_images(sample_dir: Optional[str]) -> List[Path]:
    """Return original images saved beside a sample's trajectories."""
    if not sample_dir:
        return []
    path = Path(sample_dir)
    if not path.exists() or not path.is_dir():
        return []
    return [
        image_path
        for image_path in path.glob("original_image*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _write_debug_text(debug_dir: Optional[str], filename: str, content: object) -> None:
    """Write one optional debug file and ignore filesystem errors."""
    if not debug_dir:
        return
    try:
        path = Path(debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / filename).write_text(str(content or ""), encoding="utf-8")
    except Exception:
        pass


def _write_debug_texts(debug_dir: Optional[str], files: Mapping[str, object]) -> None:
    for filename, content in files.items():
        _write_debug_text(debug_dir, filename, content)


def load_experiences(path: str) -> ExperienceLibrary:
    """Load experiences as canonical v1 records."""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _load_experience_library_payload(json.load(f))
    except FileNotFoundError:
        return {}


def save_library(path: str, experiences: Dict[str, Any]) -> None:
    """Save the canonical v1 experience library JSON payload."""
    payload = _serialize_experience_library(experiences)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_for_prompt(experiences: Dict[str, Any], max_items: int = 32) -> str:
    """Format experiences for prompt injection."""
    from prompts.experience_retrieval_templates import INJECTION_HEADER

    normalized = normalize_experience_library(experiences)
    if not normalized:
        return ""
    items = list(normalized.items())[:max_items]
    bullets = "\n".join(
        f"- [{exp_id}] {format_experience_for_prompt(exp_id, record)}"
        for exp_id, record in items
    )
    return INJECTION_HEADER.format(bullets=bullets)
