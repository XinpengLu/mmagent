"""Utility functions for the experience module."""

import os
import json
import base64
import io
from typing import Any, Dict
from PIL import Image

from .experience_schema import (
    ExperienceLibrary,
    format_experience_for_prompt,
    normalize_experience_library,
)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI.
    
    Args:
        image: PIL Image to convert
        
    Returns:
        Base64-encoded string
    """
    # 给多模态 API 使用：PIL 图片转 JPEG base64 字符串。
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --------- Experience Library I/O ---------

def load_existing(path: str) -> ExperienceLibrary:
    """Load existing experiences from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary mapping experience IDs to experience text
    """
    try:
        # 兼容两种格式：{"experiences": {...}} 或直接 {id: text}。
        data = json.load(open(path, "r", encoding="utf-8"))
        return normalize_experience_library(data)
    except Exception:
        pass
    return {}


def save_library(path: str, experiences: Dict[str, Any]):
    """Save experiences to a JSON file.
    
    Args:
        path: Path to save the JSON file
        experiences: Dictionary mapping experience IDs to experience text
    """
    # 统一保存为带 experiences 外层字段的结构化格式。
    normalized = normalize_experience_library(experiences)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 2, "experiences": normalized}, f, ensure_ascii=False, indent=2)


def load_experiences(path: str) -> ExperienceLibrary:
    """Load experiences from a JSON file.
    从 JSON 文件中加载经验数据
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary mapping experience IDs to experience text
        将经验ID与经验文本进行映射的字典
    """
    if path and os.path.exists(path):
        try:
            # 与 load_existing 保持一致，兼容旧经验库文件。
            data = json.load(open(path, "r", encoding="utf-8"))
            return normalize_experience_library(data)
        except Exception:
            pass
    return {}


def format_for_prompt(experiences: Dict[str, Any], max_items: int = 32) -> str:
    """Format experiences for injection into prompts.
    将经验格式化注入提示中

    Args:
        experiences: Dictionary mapping experience IDs to experience text
        一个将经验 ID 与经验文本映射起来的字典
        max_items: Maximum number of experiences to include 要包含的最大经验数量

    Returns:
        Formatted string for prompt injection 用于提示注入的格式化字符串bullets中
    """
    from prompts.experience_prompts_test_time import INJECTION_HEADER
    
    if not experiences:
        return ""
    # 只取前 max_items 条，避免 system prompt 过长。
    items = list(experiences.items())[:max_items]
    bullets = "\n".join([f"- [{k}] {format_experience_for_prompt(k, v)}" for k, v in items])
    return INJECTION_HEADER.format(bullets=bullets)

