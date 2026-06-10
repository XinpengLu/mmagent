"""Merge, reduce, prune, and refine canonical experience libraries."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from prompts.experience_generation_templates import EXPERIENCE_REFINE_PROMPT, MERGE_PROMPT

from configs.runtime import runtime_setting
from utils.json_utils import parse_llm_json_list
from .config import ExperienceLibrarySettings
from .llm_client import ExperienceLLM
from .retriever import ExperienceRetriever
from .schema import (
    ExperienceLibrary,
    format_experience_for_prompt,
    get_experience_content,
    get_experience_kind,
    mark_low_information_success,
    _make_record_for_id,
    _merge_record_metadata,
    normalize_experience_library,
)
from .utils import _write_debug_texts


def _settings() -> ExperienceLibrarySettings:
    return ExperienceLibrarySettings.from_runtime()


def _record_task_tags(record: Any) -> set:
    if not isinstance(record, dict):
        return set()
    generic = {
        "general",
        "visual",
        "vision",
        "visual_reasoning",
        "image_analysis",
        "tool_use",
        "tool",
    }
    tags = set()
    for tag in record.get("task_tags", []) or []:
        text = str(tag).strip().lower()
        if text.startswith("data_source_"):
            continue
        if text and text not in generic:
            tags.add(text)
    return tags


def _explicit_task_values(record: Any) -> set:
    if not isinstance(record, dict):
        return set()
    values = set()
    for tag in record.get("task_tags", []) or []:
        text = str(tag).strip().lower()
        if text.startswith("task_") and len(text) > len("task_"):
            values.add(text[len("task_"):])
    return values


def _task_tags_compatible(record_a: Any, record_b: Any) -> bool:
    """Prevent similarity merging across unrelated task types."""
    task_a = _explicit_task_values(record_a)
    task_b = _explicit_task_values(record_b)
    if task_a and task_b:
        if len(task_a) != 1 or len(task_b) != 1:
            return False
        return bool(task_a & task_b)
    tags_a = _record_task_tags(record_a)
    tags_b = _record_task_tags(record_b)
    if not tags_a or not tags_b:
        return False
    return bool(tags_a & tags_b)


def _is_reference_protected(record: Any) -> bool:
    """Keep experience IDs stable when other artifacts already depend on them."""
    if not isinstance(record, dict):
        return False
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    return bool(
        record.get("skill_ids")
        or record.get("recovery_status")
        or stats.get("recovery_status")
        or stats.get("recovered_from_failure")
        or stats.get("supervised_correction")
    )


def _strip_markdown_code_blocks(text: str) -> str:
    text = str(text or "")
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    if len(parts) >= 3:
        inner = parts[1]
        first_newline = inner.find("\n")
        if first_newline != -1:
            inner = inner[first_newline + 1:]
        return inner.strip()
    return (parts[0].strip() if parts else text).strip()


def _max_exp_id(exp_dict: Dict[str, Any], default: int = 0) -> int:
    nums = []
    for key in exp_dict:
        text = str(key)
        if text.startswith("E") and len(text) > 1 and text[1:].isdigit():
            nums.append(int(text[1:]))
    return max(nums, default=default)


class _ExperienceMemoryProvider:
    """Mutable view of an experience library with embedding similarity search."""

    def __init__(self, experiences: Dict[str, Any], retriever: Optional[ExperienceRetriever] = None):
        self.experiences = normalize_experience_library(experiences)
        self._retriever = retriever
        self._embedding_cache: Dict[str, np.ndarray] = {}
        if retriever is not None:
            try:
                for exp_id, emb in getattr(retriever, "_experience_embeddings", {}).items():
                    record = self.experiences.get(exp_id)
                    if record is not None and emb is not None:
                        self._embedding_cache[get_experience_content(record)] = emb
            except Exception:
                pass

    def _get_retriever(self) -> ExperienceRetriever:
        if self._retriever is None:
            self._retriever = ExperienceRetriever(
                experiences=self.experiences,
                embedding_model=os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_api_key=os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                embedding_endpoint=os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                enable_cache=False,
            )
        return self._retriever

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        try:
            retriever = self._get_retriever()
            ordered: List[Optional[np.ndarray]] = [None] * len(texts)
            missing: List[str] = []
            missing_positions: Dict[str, List[int]] = {}
            for idx, text in enumerate(texts):
                key = str(text or "")
                cached = self._embedding_cache.get(key)
                if cached is not None:
                    ordered[idx] = cached
                else:
                    if key not in missing_positions:
                        missing.append(key)
                        missing_positions[key] = []
                    missing_positions[key].append(idx)

            if missing:
                new_embeddings = retriever._generate_embeddings_batch(missing, batch_size=len(missing))
                for text, emb in zip(missing, new_embeddings):
                    if emb is None:
                        continue
                    self._embedding_cache[text] = emb
                    for idx in missing_positions.get(text, []):
                        ordered[idx] = emb
            embeddings = ordered
        except Exception as exc:
            print(f"  - Warning: embedding unavailable for merge search: {exc}")
            return np.array([], dtype=np.float32)

        valid_embeddings = [embedding for embedding in embeddings if embedding is not None]
        if not valid_embeddings or len(valid_embeddings) != len(texts):
            return np.array([], dtype=np.float32)
        if len(valid_embeddings) == 1:
            return valid_embeddings[0].reshape(1, -1)
        return np.vstack(valid_embeddings)

    def search_similar(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int,
        similarity_threshold: float,
    ) -> List[Tuple[str, Any, float]]:
        if not self.experiences:
            return []

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        exp_ids = list(self.experiences.keys())
        exp_texts = [get_experience_content(self.experiences[exp_id]) for exp_id in exp_ids]
        exp_embeddings = self._embed_texts(exp_texts)
        if exp_embeddings.size == 0:
            return []

        if len(exp_embeddings.shape) == 1:
            exp_embeddings = exp_embeddings.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, exp_embeddings)[0]
        top_indices = np.argsort(-similarities)[:top_k]

        candidates = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= similarity_threshold:
                exp_id = exp_ids[idx]
                candidates.append((exp_id, self.experiences[exp_id], score))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def add_experience(self, exp_id: str, record: Any) -> None:
        self.experiences[exp_id] = _make_record_for_id(exp_id, record)

    def remove_experience(self, exp_id: str) -> None:
        if exp_id in self.experiences:
            del self.experiences[exp_id]

    def modify_experience(self, exp_id: str, new_record: Any) -> None:
        if exp_id in self.experiences:
            self.experiences[exp_id] = _make_record_for_id(exp_id, new_record)

    def to_dict(self) -> ExperienceLibrary:
        return dict(self.experiences)

def _llm_merge_experiences(
    candidates: List[Tuple[str, Any, float]],
    llm: ExperienceLLM,
    max_retries: Optional[int] = None,
) -> Optional[str]:
    if len(candidates) < 2:
        return None
    settings = _settings()
    max_retries = settings.max_retries if max_retries is None else max_retries

    exp_list = [
        f"[{exp_id}] (similarity: {score:.3f})\n{format_experience_for_prompt(exp_id, record)}"
        for exp_id, record, score in candidates
    ]
    prompt = MERGE_PROMPT.format(experiences_text="\n\n".join(exp_list))

    for attempt in range(max_retries):
        try:
            response = _strip_markdown_code_blocks(llm.chat(prompt).strip())
            if response:
                return response
            print("  - Warning: Empty merge response, retrying...")
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"  - Warning: Merge LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            else:
                print(f"  - Error: Merge LLM call failed after {max_retries} attempts: {exc}")
                return None
    return None


def _process_add_with_merge(
    provider: _ExperienceMemoryProvider,
    operation: Dict[str, Any],
    llm: ExperienceLLM,
    similarity_threshold: Optional[float] = None,
    max_retries: Optional[int] = None,
    next_id: int = 0,
) -> Tuple[str, bool]:
    settings = _settings()
    similarity_threshold = settings.merge_similarity_threshold if similarity_threshold is None else similarity_threshold
    exp_id = f"E{next_id}"
    new_record = mark_low_information_success(_make_record_for_id(exp_id, operation))
    new_text = get_experience_content(new_record)
    if not new_text:
        return exp_id, False
    if _is_reference_protected(new_record):
        provider.add_experience(exp_id, new_record)
        return exp_id, False

    new_emb = provider._embed_texts([new_text])
    if new_emb.size == 0:
        provider.add_experience(exp_id, new_record)
        return exp_id, False

    query_emb = new_emb[0] if len(new_emb.shape) == 2 else new_emb
    candidates = provider.search_similar(
        query_emb,
        top_k=settings.similarity_search_top_k,
        similarity_threshold=similarity_threshold,
    )
    new_kind = get_experience_kind(new_record)
    candidates = [
        (cand_id, cand_record, score)
        for cand_id, cand_record, score in candidates
        if get_experience_kind(cand_record) == new_kind
        and _task_tags_compatible(new_record, cand_record)
        and not _is_reference_protected(cand_record)
    ]

    if not candidates:
        provider.add_experience(exp_id, new_record)
        return exp_id, False

    print(f"  - Found {len(candidates)} similar experiences (threshold: {similarity_threshold:.2f}), merging...")
    all_candidates = [("NEW", new_record, 1.0)] + candidates
    merged_text = _llm_merge_experiences(all_candidates, llm, max_retries=max_retries)

    if merged_text:
        for cand_id, _, _ in candidates:
            provider.remove_experience(cand_id)
        merged_record = mark_low_information_success(
            _merge_record_metadata(exp_id, [record for _, record, _ in all_candidates], merged_text)
        )
        provider.add_experience(exp_id, merged_record)
        print(f"    - Merged {len(all_candidates)} experiences into {exp_id}")
        return exp_id, True

    print("    - Warning: Merge failed, adding new experience separately")
    provider.add_experience(exp_id, new_record)
    return exp_id, False


def _reduce_existing_if_needed(
    existing: ExperienceLibrary,
    *,
    experience_limit: Optional[int],
    llm: ExperienceLLM,
    debug_dir: Optional[str],
    similarity_threshold: float,
    retriever: Optional[ExperienceRetriever],
) -> ExperienceLibrary:
    if not isinstance(experience_limit, int) or experience_limit <= 0 or len(existing) <= experience_limit:
        return existing
    try:
        capped = _reduce_experiences_to_limit(
            existing,
            experience_limit,
            llm,
            debug_dir,
            similarity_threshold=similarity_threshold,
            retriever=retriever,
        )
        if len(capped) <= experience_limit:
            print(f"  - Library reduced to limit: {len(capped)} (<= {experience_limit})")
        else:
            print(f"  - Warning: reduction kept {len(capped)} experiences (limit {experience_limit})")
        return capped
    except Exception as exc:
        print(f"  - Warning: failed to reduce existing library to limit {experience_limit}: {exc}")
        return existing


def _operation_content(operation: Dict[str, Any]) -> str:
    return get_experience_content(operation.get("experience", operation)).strip()


def _apply_batch_operations(
    provider: _ExperienceMemoryProvider,
    updates: List[dict],
    *,
    llm: ExperienceLLM,
    similarity_threshold: float,
    max_retries: Optional[int],
) -> Tuple[int, int, List[dict]]:
    max_id = _max_exp_id(provider.experiences)
    to_modify = []
    add_count = 0
    merge_count = 0

    for operation in updates:
        if not isinstance(operation, dict) or not _operation_content(operation):
            continue
        option = operation.get("option", "add")
        modified_from = operation.get("modified_from")
        if option == "modify" and modified_from and modified_from in provider.experiences:
            to_modify.append(operation)
            continue
        if option in {"add", "modify"}:
            max_id += 1
            _, was_merged = _process_add_with_merge(
                provider,
                operation,
                llm,
                similarity_threshold=similarity_threshold,
                max_retries=max_retries,
                next_id=max_id,
            )
            add_count += 1
            merge_count += int(was_merged)

    return add_count, merge_count, to_modify


def _apply_modifications(provider: _ExperienceMemoryProvider, operations: List[dict]) -> None:
    for operation in operations:
        modified_from = operation.get("modified_from")
        new_record = mark_low_information_success(_make_record_for_id(modified_from, operation)) if modified_from else None
        if modified_from and new_record and get_experience_content(new_record):
            provider.modify_experience(modified_from, new_record)
            print(f"    - Modified {modified_from}")


def _limit_experience_library(
    experiences: ExperienceLibrary,
    *,
    experience_limit: Optional[int],
    llm: ExperienceLLM,
    debug_dir: Optional[str],
    similarity_threshold: float,
    retriever: Optional[ExperienceRetriever],
) -> ExperienceLibrary:
    if not isinstance(experience_limit, int) or experience_limit <= 0 or len(experiences) <= experience_limit:
        return experiences

    overflow = len(experiences) - experience_limit
    fast_prune_enabled = bool(runtime_setting("experience.fast_prune.enabled", True))
    fast_prune_overflow = int(runtime_setting("experience.fast_prune.max_overflow", 12))
    if fast_prune_enabled and overflow <= fast_prune_overflow:
        print(
            f"  - Current library size ({len(experiences)}) exceeds limit ({experience_limit}) "
            f"by {overflow}; using quality prune before forced merge."
        )
        pruned_experiences = _prune_experiences_to_limit(experiences, experience_limit)
        if len(pruned_experiences) <= experience_limit:
            return pruned_experiences

    print(f"  - Current library size ({len(experiences)}) exceeds limit ({experience_limit}), attempting to merge...")
    try:
        reduced_experiences = _reduce_experiences_to_limit(
            experiences,
            experience_limit,
            llm,
            debug_dir,
            similarity_threshold=similarity_threshold,
            retriever=retriever,
        )
    except Exception as exc:
        print(f"  - Warning: failed to reduce experiences to limit {experience_limit}: {exc}")
        return experiences

    if len(reduced_experiences) < len(experiences):
        print(f"  - Library size after reduction: {len(reduced_experiences)} (limit {experience_limit})")
        return reduced_experiences

    print(f"  - Warning: reduction failed, library size unchanged: {len(experiences)} (limit {experience_limit})")
    pruned_experiences = _prune_experiences_to_limit(experiences, experience_limit)
    if len(pruned_experiences) < len(experiences):
        print(f"  - Library size after pruning: {len(pruned_experiences)} (limit {experience_limit})")
        return pruned_experiences
    return experiences


def batch_merge(
    existing: Dict[str, Any],
    updates: List[dict],
    llm: ExperienceLLM,
    max_retries: Optional[int] = None,
    debug_dir: Optional[str] = None,
    experience_limit: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    retriever: Optional[ExperienceRetriever] = None,
) -> ExperienceLibrary:
    """Merge generated operations into the canonical experience library."""
    existing = normalize_experience_library(existing)
    settings = _settings()
    similarity_threshold = settings.merge_similarity_threshold if similarity_threshold is None else similarity_threshold
    if not updates:
        return _reduce_existing_if_needed(
            existing,
            experience_limit=experience_limit,
            llm=llm,
            debug_dir=debug_dir,
            similarity_threshold=similarity_threshold,
            retriever=retriever,
        )

    provider = _ExperienceMemoryProvider(existing, retriever=retriever)
    add_count, merge_count, to_modify = _apply_batch_operations(
        provider,
        updates,
        llm=llm,
        similarity_threshold=similarity_threshold,
        max_retries=max_retries,
    )

    print(f"  - Batch update: {len(updates)} updates -> {len(to_modify)} modifies, {add_count} adds ({merge_count} merged)")
    _apply_modifications(provider, to_modify)

    final_experiences = provider.to_dict()
    print(f"  - Final library size: {len(final_experiences)} experiences")
    return _limit_experience_library(
        final_experiences,
        experience_limit=experience_limit,
        llm=llm,
        debug_dir=debug_dir,
        similarity_threshold=similarity_threshold,
        retriever=retriever,
    )


def _similar_pairs(current: ExperienceLibrary, provider: _ExperienceMemoryProvider, threshold: float) -> List[Tuple[str, str, float]]:
    exp_ids = list(current.keys())
    if len(exp_ids) < 2:
        return []
    exp_texts = [get_experience_content(current[exp_id]) for exp_id in exp_ids]
    embeddings = provider._embed_texts(exp_texts)
    if embeddings.size == 0:
        return []
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    similarities = normalized @ normalized.T
    pairs = []
    for i in range(len(exp_ids)):
        for j in range(i + 1, len(exp_ids)):
            score = float(similarities[i][j])
            if score >= threshold:
                pairs.append((exp_ids[i], exp_ids[j], score))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs


def _reduction_pair_is_mergeable(current: ExperienceLibrary, exp_id1: str, exp_id2: str) -> bool:
    if exp_id1 not in current or exp_id2 not in current:
        return False
    if get_experience_kind(current[exp_id1]) != get_experience_kind(current[exp_id2]):
        return False
    if not _task_tags_compatible(current[exp_id1], current[exp_id2]):
        return False
    return not (_is_reference_protected(current[exp_id1]) or _is_reference_protected(current[exp_id2]))


def _pair_merge_prompt(current: ExperienceLibrary, exp_id1: str, exp_id2: str) -> str:
    return MERGE_PROMPT.format(
        experiences_text=(
            f"[{exp_id1}]\n{format_experience_for_prompt(exp_id1, current[exp_id1])}"
            f"\n\n[{exp_id2}]\n{format_experience_for_prompt(exp_id2, current[exp_id2])}"
        )
    )


def _merge_pair_for_reduction(
    current: ExperienceLibrary,
    provider: _ExperienceMemoryProvider,
    exp_id1: str,
    exp_id2: str,
    sim_score: float,
    llm: ExperienceLLM,
) -> bool:
    if not _reduction_pair_is_mergeable(current, exp_id1, exp_id2):
        return False

    try:
        merged_text = _strip_markdown_code_blocks(llm.chat(_pair_merge_prompt(current, exp_id1, exp_id2)).strip())
        if not merged_text:
            print(f"      - Warning: Failed to generate merged text for {exp_id1} and {exp_id2}")
            return False
        provider.remove_experience(exp_id1)
        provider.remove_experience(exp_id2)
        new_id = f"E{_max_exp_id(current) + 1}"
        merged_record = mark_low_information_success(
            _merge_record_metadata(new_id, [current[exp_id1], current[exp_id2]], merged_text)
        )
        provider.add_experience(new_id, merged_record)
        del current[exp_id1]
        del current[exp_id2]
        current[new_id] = merged_record
        print(f"      - Force merged {exp_id1} and {exp_id2} (similarity: {sim_score:.3f}) into {new_id}")
        return True
    except Exception as exc:
        print(f"      - Warning: Failed to merge {exp_id1} and {exp_id2}: {exc}")
        return False


def _try_merge_reduction_pairs(
    current: ExperienceLibrary,
    provider: _ExperienceMemoryProvider,
    similar_pairs: List[Tuple[str, str, float]],
    limit: int,
    llm: ExperienceLLM,
) -> bool:
    pairs_to_merge = min(len(similar_pairs), len(current) - limit)
    merged_any = False
    for exp_id1, exp_id2, sim_score in similar_pairs[:pairs_to_merge]:
        if len(current) <= limit:
            break
        merged_any = _merge_pair_for_reduction(
            current, provider, exp_id1, exp_id2, sim_score, llm
        ) or merged_any
    return merged_any


def _reduce_experiences_to_limit(
    experiences: Dict[str, Any],
    limit: int,
    llm: ExperienceLLM,
    debug_dir: Optional[str] = None,
    max_rounds: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    retriever: Optional[ExperienceRetriever] = None,
) -> ExperienceLibrary:
    current = normalize_experience_library(experiences)
    settings = _settings()
    max_rounds = settings.reduction_max_rounds if max_rounds is None else max_rounds
    similarity_threshold = settings.reduction_similarity_threshold if similarity_threshold is None else similarity_threshold
    round_idx = 0
    print(f"    - Starting reduction: {len(current)} experiences, target <= {limit}")
    provider = _ExperienceMemoryProvider(current, retriever=retriever)

    while len(current) > limit and round_idx < max_rounds:
        round_idx += 1
        print(f"    - Round {round_idx}: Current count: {len(current)}, target: {limit}")
        similar_pairs = _similar_pairs(current, provider, similarity_threshold)
        if not similar_pairs:
            print(f"    - Round {round_idx}: No similar pairs found (threshold: {similarity_threshold:.2f}), stopping")
            break

        merged_any = _try_merge_reduction_pairs(current, provider, similar_pairs, limit, llm)
        if not merged_any:
            print(f"    - Round {round_idx}: No merges applied, stopping")
            break
        print(f"    - Round {round_idx}: After merging, {len(current)} experiences remain")
        provider = _ExperienceMemoryProvider(current, retriever=retriever)

    return normalize_experience_library(current)


def _experience_prune_rank_key(exp_id: str, record: Dict[str, Any]) -> Tuple[Any, ...]:
    if _is_reference_protected(record):
        return (1,)
    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    val_used = int(stats.get("validation_used_count", 0) or 0)
    validation_avg = float(stats.get("validation_average_score", 0.0) or 0.0)
    validation_pass = float(stats.get("validation_pass_rate", 0.0) or 0.0)
    return (
        0,
        not stats.get("low_information_success"),
        record.get("kind") == "success",
        bool(record.get("tool_sequence")),
        val_used > 0 and (validation_avg > 0.0 or validation_pass > 0.0),
        bool(record.get("validation_checks") or record.get("fallback_plan")),
        _experience_sort_key(exp_id),
    )


def _prune_experiences_to_limit(experiences: Dict[str, Any], limit: int) -> ExperienceLibrary:
    current = normalize_experience_library(experiences)
    if not isinstance(limit, int) or limit <= 0 or len(current) <= limit:
        return current

    removable = [
        (exp_id, _experience_prune_rank_key(exp_id, record))
        for exp_id, record in current.items()
        if not _is_reference_protected(record)
    ]
    remove_count = min(len(removable), len(current) - limit)
    if remove_count <= 0:
        return current

    removable.sort(key=lambda item: (item[1], item[0]))
    for exp_id, _ in removable[:remove_count]:
        del current[exp_id]
    print(f"  - Pruned {remove_count} unreferenced low-value experience(s) to enforce limit {limit}.")
    return normalize_experience_library(current)


def _experience_sort_key(exp_id: str) -> int:
    return int(exp_id[1:]) if str(exp_id)[1:].isdigit() else 0


def _refine_prompt(experiences: ExperienceLibrary) -> str:
    exp_lines = [
        f"[{exp_id}] {format_experience_for_prompt(exp_id, experiences[exp_id])}"
        for exp_id in sorted(experiences.keys(), key=_experience_sort_key)
    ]
    return EXPERIENCE_REFINE_PROMPT.format(
        exp_count=len(experiences),
        experiences="\n\n".join(exp_lines),
    )


def _chat_with_retries(
    llm: ExperienceLLM,
    prompt: str,
    *,
    max_retries: int,
    label: str,
    max_tokens: Optional[int] = None,
):
    settings = _settings()
    max_tokens = settings.refine_max_tokens if max_tokens is None else max_tokens
    for attempt in range(max_retries):
        try:
            return llm.chat(prompt, max_tokens=max_tokens)
        except Exception as exc:
            if attempt < max_retries - 1:
                print(f"  [{label}] LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}")
                time.sleep(2 ** attempt)
            else:
                print(f"  [{label}] LLM call failed after {max_retries} attempts: {exc}")
    return None


def _parse_refine_operations(response: str) -> Optional[List[dict]]:
    try:
        operations = parse_llm_json_list(response)
    except Exception as exc:
        print(f"  [Refine] Failed to parse operations: {exc}")
        return None
    if not isinstance(operations, list):
        print("  [Refine] Invalid response format, skipping")
        return None
    return operations


def _valid_merge_sources(refined: ExperienceLibrary, merged_from: Any) -> List[str]:
    valid_sources = [exp_id for exp_id in merged_from or [] if exp_id in refined]
    if len(valid_sources) < 2:
        return []
    source_kinds = {get_experience_kind(refined[exp_id]) for exp_id in valid_sources}
    if len(source_kinds) != 1:
        return []
    if any(_is_reference_protected(refined[exp_id]) for exp_id in valid_sources):
        return []
    if not all(
        _task_tags_compatible(refined[valid_sources[0]], refined[exp_id])
        for exp_id in valid_sources[1:]
    ):
        return []
    return valid_sources


def _apply_refine_merge(refined: ExperienceLibrary, op: Dict[str, Any]) -> bool:
    merged_text = get_experience_content(op.get("experience", op)).strip()
    valid_sources = _valid_merge_sources(refined, op.get("merged_from", []))
    if not merged_text or not valid_sources:
        return False

    source_records = [refined[exp_id] for exp_id in valid_sources]
    for exp_id in valid_sources:
        del refined[exp_id]
    new_id = f"E{_max_exp_id(refined, default=-1) + 1}"
    refined[new_id] = _merge_record_metadata(new_id, source_records, merged_text)
    print(f"    - Merged {valid_sources} -> {new_id}")
    return True


def _apply_refine_delete(refined: ExperienceLibrary, op: Dict[str, Any]) -> bool:
    deleted_id = op.get("deleted_id", "")
    if deleted_id and deleted_id in refined and not _is_reference_protected(refined[deleted_id]):
        del refined[deleted_id]
        print(f"    - Deleted {deleted_id}")
        return True
    return False


def _apply_refine_operations(experiences: ExperienceLibrary, operations: List[dict]) -> Tuple[ExperienceLibrary, int, int]:
    refined = dict(experiences)
    merge_count = 0
    delete_count = 0
    for op in operations:
        if not isinstance(op, dict):
            continue
        option = op.get("option", "")
        if option == "merge" and _apply_refine_merge(refined, op):
            merge_count += 1
        elif option == "delete" and _apply_refine_delete(refined, op):
            delete_count += 1
    return normalize_experience_library(refined), merge_count, delete_count


def refine_experience_library(
    experiences: Dict[str, Any],
    llm: ExperienceLLM,
    debug_dir: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> ExperienceLibrary:
    """Run a global merge/delete refinement pass when the library is large."""
    experiences = normalize_experience_library(experiences)
    if not experiences or len(experiences) < 10:
        return experiences
    settings = _settings()
    max_retries = settings.max_retries if max_retries is None else max_retries

    print(f"  [Refine] Starting library refinement: {len(experiences)} experiences")
    prompt = _refine_prompt(experiences)
    _write_debug_texts(debug_dir, {"exp_refine_prompt.txt": prompt})

    response = _chat_with_retries(llm, prompt, max_retries=max_retries, label="Refine")
    if not response:
        return experiences
    _write_debug_texts(debug_dir, {"exp_refine_resp.txt": response})

    operations = _parse_refine_operations(response)
    if operations is None:
        return experiences

    final, merge_count, delete_count = _apply_refine_operations(experiences, operations)
    print(f"  [Refine] Complete: {len(experiences)} -> {len(final)} ({merge_count} merges, {delete_count} deletes)")
    return final
