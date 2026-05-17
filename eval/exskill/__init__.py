"""Experience module for trajectory summarization and experience generation."""

# 对外暴露的统一 LLM 客户端。
from .llm_client import ExperienceLLM

# 图片级分析：给原图和工具生成图生成 caption。
from .multimodal_analysis import generate_image_captions

# 轨迹摘要：把 rollout 过程压缩成可供学习的文本。
from .trajectory_summary import summarize_rollouts

# 经验生成：从多条轨迹摘要中抽取 add/modify 操作。
from .experience_critique import intra_sample_experiences

# 经验库管理：合并、去重、压缩经验库。
from .experience_manager import batch_merge, refine_experience_library

# 经验库 I/O 和 prompt 格式化。
from .experience_utils import (
    load_existing,
    save_library,
    load_experiences,
    format_for_prompt,
)

from .experience_schema import (
    get_experience_content,
    normalize_experience_operation,
    normalize_experience_record,
)

# 多 skill 管理：聚类生成、测试时选择、经验链接和使用反馈。
from .multi_skill import (
    cluster_experiences_to_skills,
    collect_linked_experiences,
    format_selected_skills_for_prompt,
    load_skill_index,
    rerank_experiences_for_skills,
    select_skills_for_task,
    skill_index_path,
    update_memory_usage_after_sample,
)

# Skill 生成、合并、测试时适配和压缩。
from .skill_builder import (
    generate_skill_for_sample,
    merge_skills,
    adapt_skill_for_task,
    refine_skill_document,
)

# 测试时经验检索和经验重写。
from .experience_retriever import ExperienceRetriever, rewrite_experiences_for_task

# 控制 from exskill import * 时导出的公共接口。
__all__ = [
    "ExperienceLLM",
    "generate_image_captions",
    "summarize_rollouts",
    "intra_sample_experiences",
    "batch_merge",
    "refine_experience_library",
    "load_existing",
    "save_library",
    "load_experiences",
    "format_for_prompt",
    "get_experience_content",
    "normalize_experience_operation",
    "normalize_experience_record",
    "cluster_experiences_to_skills",
    "collect_linked_experiences",
    "format_selected_skills_for_prompt",
    "load_skill_index",
    "rerank_experiences_for_skills",
    "select_skills_for_task",
    "skill_index_path",
    "update_memory_usage_after_sample",
    "generate_skill_for_sample",
    "merge_skills",
    "adapt_skill_for_task",
    "refine_skill_document",
    "ExperienceRetriever",
    "rewrite_experiences_for_task",
]

