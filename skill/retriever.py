"""Embedding retriever for modular skill candidates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils.env import env_bool, env_float, env_int
from .index import load_skill_index, skill_index_path, skill_root
from .filters import (
    allow_unvalidated_recovery_inference,
    recovery_has_train_support,
    skill_selection_weight,
    skill_unvalidated_recovery,
)
from .utils import (
    cosine_scores,
    enabled_callable_tools,
    metadata_fields,
    metadata_selection_weight,
    tag_tokens,
    truncate_text,
)


def _read_skill_content(item: Dict[str, Any]) -> str:
    path = item.get("path", "") if isinstance(item, dict) else ""
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _skill_retrieval_text(skill_id: str, item: Dict[str, Any], content: Optional[str] = None) -> str:
    content = content if content is not None else _read_skill_content(item)
    fields = [
        f"skill_id: {skill_id}",
        f"description: {item.get('description', '')}",
        f"skill_type: {item.get('skill_type', '')}",
        f"recovery_status: {item.get('recovery_status', '')}",
        f"validation_quality: {item.get('validation_quality', '')}",
        f"trigger: {item.get('trigger_text', '')}",
        f"task_tags: {', '.join(item.get('task_tags', []))}",
        f"tool_tags: {', '.join(item.get('tool_tags', []))}",
        f"linked_experience_ids: {', '.join(item.get('linked_experience_ids', []))}",
        truncate_text(content, 4000),
    ]
    return "\n".join(part for part in fields if part is not None)


def _queries_from_retrieval_info(query_text: str, retrieval_info: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if retrieval_info and retrieval_info.get("decomposition_used"):
        queries = []
        for subtask in retrieval_info.get("subtasks", []):
            if not isinstance(subtask, dict):
                continue
            query = str(subtask.get("query") or "").strip()
            if query:
                queries.append({
                    "type": str(subtask.get("type") or "subtask"),
                    "query": query,
                })
        if queries:
            return queries
    return [{"type": "original", "query": query_text}]


def _metadata_first_allows(candidate_tags: Iterable[str], query_metadata_tags: Iterable[str]) -> bool:
    """Prefer skills from the same structured benchmark slice before embedding ranking."""
    query_fields = metadata_fields(query_metadata_tags or [])
    if not query_fields:
        return True

    candidate_fields = metadata_fields(candidate_tags or [])
    candidate_tokens = tag_tokens(candidate_tags or [])

    query_ds = query_fields.get("data_source")
    candidate_ds = candidate_fields.get("data_source")
    if query_ds and candidate_ds and not (query_ds & candidate_ds):
        return False

    for field in ("task", "category", "prompt_category", "subtask"):
        query_values = query_fields.get(field)
        if not query_values:
            continue
        candidate_values = candidate_fields.get(field, set())
        if field == "task" and len(candidate_values) > 1:
            return False
        if candidate_values & query_values:
            return True
        if candidate_tokens & query_values:
            return True
        return False

    return True


def _recovery_probe_enabled() -> bool:
    if env_bool("ENABLE_UNVALIDATED_RECOVERY_VALIDATION_PROBE", False) and env_bool("MMAGENT_VALIDATION_PHASE", False):
        return True
    return env_bool("ENABLE_UNVALIDATED_RECOVERY_TRAINING_PROBE", False) and env_bool("MMAGENT_TRAIN_PHASE", False)


def _recovery_probe_phase() -> str:
    if env_bool("MMAGENT_TRAIN_PHASE", False):
        return "train"
    if env_bool("MMAGENT_VALIDATION_PHASE", False):
        return "validation"
    return "unknown"


def _metadata_filter_skills(
    skills: Dict[str, Dict[str, Any]],
    query_metadata_tags: Iterable[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    info = {
        "enabled": True,
        "applied": False,
        "candidate_count_before": len(skills),
        "candidate_count_after": len(skills),
    }
    query_metadata_tags = list(query_metadata_tags or [])
    if not query_metadata_tags or not skills:
        return skills, info
    filtered = {
        skill_id: item
        for skill_id, item in skills.items()
        if _metadata_first_allows(item.get("task_tags", []), query_metadata_tags)
    }
    if filtered:
        info["applied"] = True
        info["candidate_count_after"] = len(filtered)
        return filtered, info
    info["applied"] = True
    info["candidate_count_after"] = 0
    return {}, info


def _can_probe_tool_plan(item: Dict[str, Any]) -> bool:
    if not item.get("requires_tool_plan", True):
        return False
    tools = {
        str(tool).strip()
        for tool in item.get("tool_tags", []) or []
        if str(tool).strip() in enabled_callable_tools()
    }
    if not tools:
        return False
    support = item.get("support_counts") if isinstance(item.get("support_counts"), dict) else {}
    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
    if item.get("skill_type") != "recovery_action_skill":
        return True
    if item.get("recovery_status") != "candidate_unverified":
        return True
    if int(stats.get("validation_used_count", 0) or 0) > 0:
        return True
    return (
        int(support.get("recovery", 0) or 0) > 0
        and (
            int(support.get("success_prior", 0) or 0) > 0
            or int(support.get("supervised_correction", 0) or 0) > 0
            or bool(item.get("source_success_experience_ids"))
        )
    )


def _add_validation_probe_candidates(
    candidate_by_id: Dict[str, Dict[str, Any]],
    prefiltered_skills: Dict[str, Dict[str, Any]],
    skill_contents: Dict[str, str],
    query_text: str,
    query_metadata_tags: Iterable[str],
) -> List[str]:
    """Append a few unverified recovery skills so train/validation can test them.

    Embedding retrieval is still the primary selector. This small probe
    expansion prevents newly synthesized recovery skills from being ignored
    simply because they have not accumulated enough embedding/query exposure yet.
    LLM reranking and downstream quality gates still decide whether to use them.
    """
    if not _recovery_probe_enabled():
        return []

    max_probe_extras = env_int("VALIDATION_PROBE_EXTRA_SKILL_CANDIDATES", 3)
    if max_probe_extras <= 0:
        return []

    probe_candidates = []
    for skill_id, item in prefiltered_skills.items():
        if skill_id in candidate_by_id or not skill_unvalidated_recovery(item):
            continue
        if not _can_probe_tool_plan(item):
            continue
        content = skill_contents.get(skill_id)
        if not content:
            continue
        metadata_weight = metadata_selection_weight(item.get("task_tags", []), query_metadata_tags)
        selection_weight = skill_selection_weight(item)
        if selection_weight <= 0.0:
            continue
        probe_candidates.append((
            metadata_weight * selection_weight,
            skill_id,
            item,
            content,
            metadata_weight,
            selection_weight,
        ))

    probe_candidates.sort(key=lambda row: row[0], reverse=True)
    probe_score = env_float("VALIDATION_PROBE_EXTRA_SKILL_SCORE", 0.12)
    added: List[str] = []
    for _, skill_id, item, content, metadata_weight, selection_weight in probe_candidates[:max_probe_extras]:
        candidate_by_id[skill_id] = {
            "skill_id": skill_id,
            "score": probe_score * max(0.20, min(1.5, metadata_weight)),
            "raw_score": 0.0,
            "selection_weight": selection_weight,
            "metadata_weight": metadata_weight,
            "metadata": item,
            "content": content,
            "matched_queries": [{
                "query": query_text,
                "type": f"{_recovery_probe_phase()}_recovery_probe_extra",
                "rank": 0,
                "score": probe_score,
                "raw_score": 0.0,
                "metadata_weight": metadata_weight,
            }],
        }
        added.append(skill_id)
    return added


def _empty_retrieval_info(top_k: int) -> Dict[str, Any]:
    return {
        "retrieval_method": "embedding",
        "top_k_per_query": top_k,
        "retrieved_skill_ids": [],
        "retrieval_details": [],
    }


def _prefilter_skills(
    skills: Dict[str, Dict[str, Any]],
    available_experience_ids: Optional[Iterable[str]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    available_ids = set(available_experience_ids) if available_experience_ids is not None else None
    prefiltered_skills = {}
    probe_pool_skills = {}
    skipped_by_policy = []

    for skill_id, item in skills.items():
        if isinstance(item, dict) and item.get("disabled"):
            continue
        if (
            skill_unvalidated_recovery(item)
            and not allow_unvalidated_recovery_inference()
            and not recovery_has_train_support(item)
        ):
            skipped_by_policy.append(skill_id)
            continue
        linked_ids = item.get("linked_experience_ids", []) if isinstance(item, dict) else []
        if available_ids is not None and not any(exp_id in available_ids for exp_id in linked_ids):
            continue
        probe_pool_skills[skill_id] = item
        prefiltered_skills[skill_id] = item

    return prefiltered_skills, probe_pool_skills, skipped_by_policy


def _load_skill_texts(
    prefiltered_skills: Dict[str, Dict[str, Any]],
    probe_pool_skills: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    skill_texts = {}
    skill_contents = {}
    content_source_skills = dict(probe_pool_skills)
    content_source_skills.update(prefiltered_skills)
    for skill_id, item in content_source_skills.items():
        content = _read_skill_content(item)
        if not content:
            continue
        skill_contents[skill_id] = content
        if skill_id in prefiltered_skills:
            skill_texts[skill_id] = _skill_retrieval_text(skill_id, item, content)
    return skill_texts, skill_contents


def _retriever_records(skill_texts: Dict[str, str], prefiltered_skills: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        skill_id: {
            "id": skill_id,
            "kind": "success",
            "skill_ids": [skill_id],
            "task_tags": list(item.get("task_tags", []) or []),
            "tool_tags": list(item.get("tool_tags", []) or []),
            "confidence": float(item.get("confidence", 0.7) or 0.7),
            "content": text,
            "stats": {"support_count": 1, "used_count": 0, "success_after_used": 0, "failure_after_used": 0},
        }
        for skill_id, text in skill_texts.items()
        for item in [prefiltered_skills.get(skill_id, {})]
    }


def _build_embedding_retriever(
    skill_library_path: str,
    skill_texts: Dict[str, str],
    prefiltered_skills: Dict[str, Dict[str, Any]],
    embedding_model: Optional[str],
    embedding_api_key: Optional[str],
    embedding_endpoint: Optional[str],
    enable_cache: bool,
):
    try:
        from experience.retriever import ExperienceRetriever

        return ExperienceRetriever(
            experiences=_retriever_records(skill_texts, prefiltered_skills),
            embedding_model=embedding_model or os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=embedding_api_key or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            embedding_endpoint=embedding_endpoint or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            cache_dir=str(skill_root(skill_library_path)),
            enable_cache=enable_cache,
            experience_library_path=str(skill_index_path(skill_library_path)),
        ), None
    except Exception as e:
        print(f"  [Skill Retrieval] Warning: failed to initialize embedding retriever: {e}")
        return None, str(e)


def _empty_query_detail(query_item: Dict[str, str]) -> Dict[str, Any]:
    return {
        "query": query_item["query"],
        "type": query_item["type"],
        "retrieved_skill_ids": [],
        "count": 0,
    }


def _rank_skill_matches(
    retriever,
    query_embedding,
    skills: Dict[str, Dict[str, Any]],
    query_metadata_tags: Iterable[str],
    min_similarity: float,
) -> List[Tuple[str, float, float, float]]:
    skill_ids = list(retriever._experience_embeddings.keys())
    matrix = np.vstack([retriever._experience_embeddings[sid] for sid in skill_ids])
    scores = cosine_scores(query_embedding, matrix)
    ranked = []
    for i in range(len(skill_ids)):
        skill_id = skill_ids[i]
        item = skills.get(skill_id, {})
        base_weight = skill_selection_weight(item)
        if float(scores[i]) < min_similarity or base_weight <= 0.0:
            continue
        metadata_weight = metadata_selection_weight(item.get("task_tags", []), query_metadata_tags)
        ranked.append((skill_id, float(scores[i]), float(scores[i]) * base_weight * metadata_weight, metadata_weight))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def _retrieval_detail(query_item: Dict[str, str], top_ranked: List[Tuple[str, float, float, float]]) -> Dict[str, Any]:
    return {
        "query": query_item["query"],
        "type": query_item["type"],
        "retrieved_skill_ids": [sid for sid, _, _, _ in top_ranked],
        "scores": {sid: adjusted_score for sid, _, adjusted_score, _ in top_ranked},
        "raw_scores": {sid: raw_score for sid, raw_score, _, _ in top_ranked},
        "metadata_weights": {sid: metadata_weight for sid, _, _, metadata_weight in top_ranked},
        "count": len(top_ranked),
    }


def _merge_ranked_candidates(
    candidate_by_id: Dict[str, Dict[str, Any]],
    top_ranked: List[Tuple[str, float, float, float]],
    query_item: Dict[str, str],
    skills: Dict[str, Dict[str, Any]],
    skill_contents: Dict[str, str],
) -> None:
    for rank, (skill_id, raw_score, score, metadata_weight) in enumerate(top_ranked, 1):
        item = skills[skill_id]
        entry = candidate_by_id.get(skill_id)
        if entry is None:
            entry = {
                "skill_id": skill_id,
                "score": score,
                "raw_score": raw_score,
                "selection_weight": skill_selection_weight(item),
                "metadata_weight": metadata_weight,
                "metadata": item,
                "content": skill_contents[skill_id],
                "matched_queries": [],
            }
            candidate_by_id[skill_id] = entry
        entry["score"] = max(float(entry.get("score", 0.0)), score)
        entry["raw_score"] = max(float(entry.get("raw_score", 0.0)), raw_score)
        entry["metadata_weight"] = max(float(entry.get("metadata_weight", 0.0)), metadata_weight)
        entry["matched_queries"].append({
            "query": query_item["query"],
            "type": query_item["type"],
            "rank": rank,
            "score": score,
            "raw_score": raw_score,
            "metadata_weight": metadata_weight,
        })


def _retrieve_embedding_candidates(
    retriever,
    query_text: str,
    retrieval_info: Optional[Dict[str, Any]],
    skills: Dict[str, Dict[str, Any]],
    skill_contents: Dict[str, str],
    top_k: int,
    min_similarity: float,
    query_metadata_tags: Iterable[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    details = []
    for query_item in _queries_from_retrieval_info(query_text, retrieval_info):
        query_embedding = retriever._generate_embedding(query_item["query"])
        if query_embedding is None or not retriever._experience_embeddings:
            details.append(_empty_query_detail(query_item))
            continue

        ranked = _rank_skill_matches(
            retriever, query_embedding, skills, query_metadata_tags, min_similarity
        )
        top_ranked = ranked[:top_k]
        details.append(_retrieval_detail(query_item, top_ranked))
        _merge_ranked_candidates(candidate_by_id, top_ranked, query_item, skills, skill_contents)
    return candidate_by_id, details


def _skill_retrieval_info(
    top_k: int,
    candidates: List[Dict[str, Any]],
    details: List[Dict[str, Any]],
    query_metadata_tags: List[str],
    skipped_by_policy: List[str],
    metadata_first_info: Dict[str, Any],
    probe_metadata_info: Dict[str, Any],
    embedding_error: Optional[str],
    probe_extra_ids: List[str],
) -> Dict[str, Any]:
    return {
        "retrieval_method": "embedding",
        "top_k_per_query": top_k,
        "retrieved_skill_ids": [item["skill_id"] for item in candidates],
        "retrieval_details": details,
        "total_unique_skills": len(candidates),
        "query_metadata_tags": query_metadata_tags,
        "skipped_by_policy": skipped_by_policy[:64],
        "metadata_first": metadata_first_info,
        "probe_metadata_first": probe_metadata_info,
        "embedding_error": embedding_error,
        "recovery_probe_phase": _recovery_probe_phase() if probe_extra_ids else "",
        "recovery_probe_extra_skill_ids": probe_extra_ids,
        "validation_probe_extra_skill_ids": probe_extra_ids,
    }


def _load_prefiltered_skill_pool(
    skill_library_path: str,
    available_experience_ids: Optional[Iterable[str]],
    query_metadata_tags: List[str],
    top_k: int,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Any],
    Dict[str, Any],
    List[str],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    index = load_skill_index(skill_library_path)
    skills = index.get("skills", {})
    if not skills:
        return {}, {}, {}, {}, {}, {}, [], _empty_retrieval_info(top_k)

    prefiltered_skills, probe_pool_skills, skipped_by_policy = _prefilter_skills(
        skills, available_experience_ids
    )
    prefiltered_skills, metadata_first_info = _metadata_filter_skills(prefiltered_skills, query_metadata_tags)
    probe_pool_skills, probe_metadata_info = _metadata_filter_skills(probe_pool_skills, query_metadata_tags)
    query_fields = metadata_fields(query_metadata_tags)
    if query_fields.get("task") and not prefiltered_skills and not probe_pool_skills:
        empty_info = _empty_retrieval_info(top_k)
        empty_info["metadata_first"] = metadata_first_info
        empty_info["probe_metadata_first"] = probe_metadata_info
        empty_info["query_metadata_tags"] = query_metadata_tags
        empty_info["strict_task_filter"] = True
        return skills, {}, {}, {}, metadata_first_info, probe_metadata_info, skipped_by_policy, empty_info

    skill_texts, skill_contents = _load_skill_texts(prefiltered_skills, probe_pool_skills)
    return skills, prefiltered_skills, probe_pool_skills, skill_texts, skill_contents, {
        "metadata_first": metadata_first_info,
        "probe_metadata_first": probe_metadata_info,
    }, skipped_by_policy, None


def _embedding_skill_candidates(
    *,
    skill_library_path: str,
    skill_texts: Dict[str, str],
    prefiltered_skills: Dict[str, Dict[str, Any]],
    query_text: str,
    retrieval_info: Optional[Dict[str, Any]],
    skills: Dict[str, Dict[str, Any]],
    skill_contents: Dict[str, str],
    top_k: int,
    min_similarity: float,
    query_metadata_tags: List[str],
    embedding_model: Optional[str],
    embedding_api_key: Optional[str],
    embedding_endpoint: Optional[str],
    enable_cache: bool,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    if not skill_texts:
        return {}, [], None

    retriever, embedding_error = _build_embedding_retriever(
        skill_library_path,
        skill_texts,
        prefiltered_skills,
        embedding_model,
        embedding_api_key,
        embedding_endpoint,
        enable_cache,
    )
    if retriever is None:
        return {}, [], embedding_error
    candidate_by_id, details = _retrieve_embedding_candidates(
        retriever,
        query_text,
        retrieval_info,
        skills,
        skill_contents,
        top_k,
        min_similarity,
        query_metadata_tags,
    )
    return candidate_by_id, details, embedding_error


def _finalize_skill_candidates(
    *,
    candidate_by_id: Dict[str, Dict[str, Any]],
    probe_pool_skills: Dict[str, Dict[str, Any]],
    skill_contents: Dict[str, str],
    query_text: str,
    query_metadata_tags: List[str],
    top_k: int,
    details: List[Dict[str, Any]],
    skipped_by_policy: List[str],
    metadata_infos: Dict[str, Any],
    embedding_error: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    probe_extra_ids = _add_validation_probe_candidates(
        candidate_by_id,
        probe_pool_skills,
        skill_contents,
        query_text,
        query_metadata_tags,
    )
    candidates = sorted(candidate_by_id.values(), key=lambda item: item.get("score", 0.0), reverse=True)
    return candidates, _skill_retrieval_info(
        top_k,
        candidates,
        details,
        query_metadata_tags,
        skipped_by_policy,
        metadata_infos.get("metadata_first", {}),
        metadata_infos.get("probe_metadata_first", {}),
        embedding_error,
        probe_extra_ids,
    )


def retrieve_skills_for_task(
    skill_library_path: Optional[str],
    query_text: str,
    retrieval_info: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    min_similarity: float = 0.0,
    available_experience_ids: Optional[Iterable[str]] = None,
    query_metadata_tags: Optional[Iterable[str]] = None,
    embedding_model: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    enable_cache: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Embedding-retrieve modular skills, retrieving top_k skills per subtask query."""
    empty_info = _empty_retrieval_info(top_k)
    if not skill_library_path or top_k <= 0:
        return [], empty_info

    query_metadata_tags = list(query_metadata_tags or [])
    (
        skills,
        prefiltered_skills,
        probe_pool_skills,
        skill_texts,
        skill_contents,
        metadata_infos,
        skipped_by_policy,
        early_info,
    ) = _load_prefiltered_skill_pool(
        skill_library_path,
        available_experience_ids,
        query_metadata_tags,
        top_k,
    )
    if early_info is not None:
        return [], early_info
    if not skills:
        return [], empty_info

    candidate_by_id, details, embedding_error = _embedding_skill_candidates(
        skill_library_path=skill_library_path,
        skill_texts=skill_texts,
        prefiltered_skills=prefiltered_skills,
        query_text=query_text,
        retrieval_info=retrieval_info,
        skills=skills,
        skill_contents=skill_contents,
        top_k=top_k,
        min_similarity=min_similarity,
        query_metadata_tags=query_metadata_tags,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_endpoint=embedding_endpoint,
        enable_cache=enable_cache,
    )
    return _finalize_skill_candidates(
        candidate_by_id=candidate_by_id,
        probe_pool_skills=probe_pool_skills,
        skill_contents=skill_contents,
        query_text=query_text,
        query_metadata_tags=query_metadata_tags,
        top_k=top_k,
        details=details,
        skipped_by_policy=skipped_by_policy,
        metadata_infos=metadata_infos,
        embedding_error=embedding_error,
    )
