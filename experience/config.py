"""Runtime configuration helpers for experience memory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from configs.runtime import runtime_setting


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
            max_retries=3,
            base_wait=1.0,
            max_wait=30.0,
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
    caption_max_tokens: int
    caption_workers: int
    caption_temperature: float

    @classmethod
    def from_runtime(cls) -> "ExperienceGenerationSettings":
        return cls(
            critique_retries=3,
            critique_max_tokens=4096,
            trajectory_retries=3,
            trajectory_max_tokens=4096,
            trajectory_max_initial_prompt_chars=12000,
            trajectory_max_text_output_chars=6000,
            trajectory_max_tool_code_chars=6000,
            trajectory_max_tool_result_chars=8000,
            trajectory_max_other_field_chars=3000,
            trajectory_image_budget=int(runtime_setting("rollout.summary_image_budget", 12)),
            trajectory_tool_images_per_rollout=int(runtime_setting("rollout.summary_tool_images_per_rollout", 3)),
            caption_max_tokens=512,
            caption_workers=int(runtime_setting("experience.image_caption_workers", 4)),
            caption_temperature=0.3,
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
    def from_runtime(cls) -> "EmbeddingSettings":
        return cls(
            max_retries=3,
            timeout=60,
            max_concurrency=4,
            batch_size=30,
            max_input_chars=12000,
            retry_http_400=True,
        )


@dataclass(frozen=True)
class ExperienceRetrievalSettings:
    top_k: int
    min_similarity: float
    task_decomposition_max_tokens: int
    rewrite_max_tokens: int
    task_decomposition_temperature: float
    rewrite_temperature: float
    top_p: float
    mmr_lambda: float
    image_description_max_tokens: int
    image_description_temperature: float
    decomposition_direct_max_experiences: int

    @classmethod
    def from_runtime(cls) -> "ExperienceRetrievalSettings":
        return cls(
            top_k=max(1, int(runtime_setting("experience.retrieval.top_k", 3))),
            min_similarity=float(runtime_setting("experience.retrieval.min_similarity", 0.0)),
            task_decomposition_max_tokens=2048,
            rewrite_max_tokens=8192,
            task_decomposition_temperature=0.3,
            rewrite_temperature=0.3,
            top_p=1.0,
            mmr_lambda=float(runtime_setting("experience.retrieval.mmr_lambda", 0.7)),
            image_description_max_tokens=128,
            image_description_temperature=0.1,
            decomposition_direct_max_experiences=int(runtime_setting("experience.retrieval.direct_max_experiences", 20)),
        )


@dataclass(frozen=True)
class ExperienceLibrarySettings:
    max_retries: int
    merge_similarity_threshold: float
    reduction_similarity_threshold: float
    similarity_search_top_k: int
    reduction_max_rounds: int
    refine_max_tokens: int

    @classmethod
    def from_runtime(cls) -> "ExperienceLibrarySettings":
        return cls(
            max_retries=3,
            merge_similarity_threshold=float(runtime_setting("experience.library.merge_similarity_threshold", 0.70)),
            reduction_similarity_threshold=float(runtime_setting("experience.library.reduction_similarity_threshold", 0.60)),
            similarity_search_top_k=int(runtime_setting("experience.library.similarity_top_k", 10)),
            reduction_max_rounds=int(runtime_setting("experience.library.reduction_max_rounds", 3)),
            refine_max_tokens=16384,
        )
