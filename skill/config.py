"""Runtime configuration helpers for skill memory."""

from __future__ import annotations

from dataclasses import dataclass

from configs.runtime import runtime_setting


@dataclass(frozen=True)
class SkillSettings:
    cluster_similarity_threshold: float
    rerank_max_tokens: int
    rerank_temperature: float

    @classmethod
    def from_runtime(cls) -> "SkillSettings":
        return cls(
            cluster_similarity_threshold=float(runtime_setting("skill.cluster.similarity_threshold", 0.74)),
            rerank_max_tokens=2048,
            rerank_temperature=0.0,
        )
    
    @classmethod
    def from_env(cls) -> "SkillSettings":
        return cls.from_runtime()
