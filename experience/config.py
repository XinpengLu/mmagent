"""Runtime configuration helpers for experience memory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from utils.env import env_bool, env_float, env_int


@dataclass(frozen=True)
class ExperienceLLMSettings:
    model_name: str
    max_retries: int
    base_wait: float
    max_wait: float
    api_timeout: int = 300
    image_api_timeout: int = 300
    default_max_tokens: int = 12288

    @classmethod
    def from_env(cls, model_name: Optional[str] = None) -> "ExperienceLLMSettings":
        return cls(
            model_name=model_name or os.environ.get("EXPERIENCE_MODEL_NAME", "gpt-4o"),
            max_retries=env_int("EXPERIENCE_API_MAX_RETRIES", 3),
            base_wait=env_float("EXPERIENCE_API_BASE_WAIT", 1.0),
            max_wait=env_float("EXPERIENCE_API_MAX_WAIT", 30.0),
        )


@dataclass(frozen=True)
class ExperienceGenerationSettings:
    critique_retries: int
    critique_max_tokens: int
    trajectory_retries: int
    trajectory_max_tokens: int
    trajectory_max_initial_prompt_chars: int
    trajectory_max_text_output_chars: int
    trajectory_max_tool_code_chars: int
    trajectory_max_tool_result_chars: int
    trajectory_max_other_field_chars: int
    trajectory_image_budget: int
    trajectory_tool_images_per_rollout: int
    caption_include_system_prompt: bool
    caption_tool_with_original: bool

    @classmethod
    def from_env(cls) -> "ExperienceGenerationSettings":
        return cls(
            critique_retries=env_int("EXPERIENCE_CRITIQUE_MAX_RETRIES", 3),
            critique_max_tokens=env_int("EXPERIENCE_CRITIQUE_MAX_TOKENS", 4096),
            trajectory_retries=env_int("TRAJ_SUMMARY_MAX_RETRIES", 3),
            trajectory_max_tokens=env_int("TRAJ_SUMMARY_MAX_TOKENS", 4096),
            trajectory_max_initial_prompt_chars=env_int("TRAJ_SUMMARY_MAX_INITIAL_PROMPT_CHARS", 12000),
            trajectory_max_text_output_chars=env_int("TRAJ_SUMMARY_MAX_TEXT_OUTPUT_CHARS", 6000),
            trajectory_max_tool_code_chars=env_int("TRAJ_SUMMARY_MAX_TOOL_CODE_CHARS", 6000),
            trajectory_max_tool_result_chars=env_int("TRAJ_SUMMARY_MAX_TOOL_RESULT_CHARS", 8000),
            trajectory_max_other_field_chars=env_int("TRAJ_SUMMARY_MAX_OTHER_FIELD_CHARS", 3000),
            trajectory_image_budget=env_int("TRAJ_SUMMARY_IMAGE_BUDGET", 12),
            trajectory_tool_images_per_rollout=env_int("TRAJ_SUMMARY_TOOL_IMAGES_PER_ROLLOUT", 3),
            caption_include_system_prompt=env_bool("TRAJ_SUMMARY_IMAGE_CAPTION_INCLUDE_SYSTEM_PROMPT", False),
            caption_tool_with_original=env_bool("TRAJ_SUMMARY_CAPTION_TOOL_WITH_ORIGINAL", False),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    max_retries: int
    timeout: int
    max_concurrency: int
    batch_size: int
    max_input_chars: int
    retry_http_400: bool

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        return cls(
            max_retries=env_int("EXPERIENCE_EMBEDDING_MAX_RETRIES", 3),
            timeout=env_int("EXPERIENCE_EMBEDDING_TIMEOUT", 60),
            max_concurrency=max(1, env_int("EXPERIENCE_EMBEDDING_MAX_CONCURRENCY", 4)),
            batch_size=max(1, env_int("EXPERIENCE_EMBEDDING_BATCH_SIZE", 30)),
            max_input_chars=max(256, env_int("EXPERIENCE_EMBEDDING_MAX_INPUT_CHARS", 12000)),
            retry_http_400=env_bool("EXPERIENCE_EMBEDDING_RETRY_400", True),
        )
