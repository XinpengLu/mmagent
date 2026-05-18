"""Cluster-based multi-skill management."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from prompts.skill_prompts import GENERATE_CLUSTER_SKILL_PROMPT

from .experience_schema import (
    ExperienceLibrary,
    format_experience_for_prompt,
    get_experience_content,
    normalize_experience_library,
    normalize_experience_record,
)


SkillIndex = Dict[str, Any]


def _load_memory_prompt(prompt_key: str = "memory_rerank") -> str:
    """Load memory prompts from YAML so prompt text is not embedded in code."""
    prompt_path = Path(
        os.environ.get("MEMORY_PROMPTS_PATH", "eval/prompts/memory_prompts.yaml")
    )
    try:
        with prompt_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        prompt = data.get(prompt_key)
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    except Exception as e:
        print(f"  [Memory Prompt] Warning: failed to load {prompt_path}: {e}")

    # Minimal fallback keeps runs alive if the external prompt file is missing.
    return (
        "Select directly useful memory for the task.\n"
        "Task:\n{query}\n\nSkills:\n{skills}\n\nExperiences:\n{experiences}\n\n"
        "Return JSON with skill_ids, experience_ids, and notes. "
        "Use at most {max_skills} skills and {max_experiences} experiences."
    )


def _slugify(parts: Iterable[str], fallback: str = "general") -> str:
    text = "_".join(part for part in parts if part)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:48] or fallback


def _skill_root(skill_library_path: str) -> Path:
    return Path(skill_library_path).expanduser().resolve().parent


def skill_index_path(skill_library_path: str) -> Path:
    return _skill_root(skill_library_path) / "skill_index.json"


def skills_dir(skill_library_path: str) -> Path:
    return _skill_root(skill_library_path) / "skills"


def load_skill_index(skill_library_path: Optional[str]) -> SkillIndex:
    if not skill_library_path:
        return {"schema_version": 1, "skills": {}}
    path = skill_index_path(skill_library_path)
    if not path.exists():
        return {"schema_version": 1, "skills": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("skills"), dict):
            return data
    except Exception:
        pass
    return {"schema_version": 1, "skills": {}}


def save_skill_index(skill_library_path: str, index: SkillIndex) -> None:
    path = skill_index_path(skill_library_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_index = {"schema_version": index.get("schema_version", 1), "skills": {}}
    for skill_id, item in (index.get("skills", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        compact_item = {}
        for key in (
            "skill_id",
            "path",
            "task_tags",
            "tool_tags",
            "linked_experience_ids",
            "trigger_text",
            "updated_at",
        ):
            value = item.get(key)
            if value not in (None, "", [], {}):
                compact_item[key] = value
        compact_index["skills"][skill_id] = compact_item
    with path.open("w", encoding="utf-8") as f:
        json.dump(compact_index, f, ensure_ascii=False, indent=2)


def _record_signature(record: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    task_tags = tuple(sorted(record.get("task_tags", []) or ["general"]))
    tool_tags = tuple(sorted(record.get("tool_tags", []) or ["reasoning"]))
    return task_tags, tool_tags


def _exp_sort_key(exp_id: str) -> Tuple[int, str]:
    if isinstance(exp_id, str) and exp_id.startswith("E") and exp_id[1:].isdigit():
        return int(exp_id[1:]), exp_id
    return 10**9, str(exp_id)


def _try_content_embeddings(records: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    api_key = os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    endpoint = os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE")
    if not api_key or not endpoint:
        return None
    try:
        from .experience_retriever import ExperienceRetriever

        retriever = ExperienceRetriever(
            experiences={},
            embedding_model=os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=api_key,
            embedding_endpoint=endpoint,
            enable_cache=False,
        )
        texts = [get_experience_content(record) for record in records]
        embeddings = retriever._generate_embeddings_batch(texts, batch_size=max(1, len(texts)))
        valid = [emb for emb in embeddings if emb is not None]
        if len(valid) == len(records):
            return np.vstack(valid)
    except Exception as e:
        print(f"  [Skill Cluster] Embedding clustering unavailable, falling back to tag clustering: {e}")
    return None


def _majority_tags(records: List[Dict[str, Any]], key: str) -> List[str]:
    """Return the most common tags across cluster members."""
    from collections import Counter
    counter: Counter = Counter()
    for record in records:
        for tag in record.get(key, []):
            counter[tag] += 1
    if not counter:
        return ["general"] if key == "task_tags" else ["reasoning"]
    threshold = len(records) * 0.3
    return sorted(tag for tag, cnt in counter.items() if cnt >= threshold) or [counter.most_common(1)[0][0]]


def _cluster_by_tags_and_content(
    experiences: ExperienceLibrary,
    similarity_threshold: float = 0.74,
) -> List[Dict[str, Any]]:
    """Cluster experiences by embedding similarity (embedding-first), with tag-based fallback."""
    normalized = normalize_experience_library(experiences)
    items = sorted(normalized.items(), key=lambda x: _exp_sort_key(x[0]))
    if not items:
        return []

    records = [record for _, record in items]
    embeddings = _try_content_embeddings(records)

    if embeddings is None:
        groups: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], List[Tuple[str, Dict[str, Any]]]] = {}
        for exp_id, record in items:
            groups.setdefault(_record_signature(record), []).append((exp_id, record))
        return [{"signature": sig, "items": group_items} for sig, group_items in groups.items()]

    similarities = _cosine_similarity_matrix(embeddings)
    assigned = [False] * len(items)
    clusters = []
    for i in range(len(items)):
        if assigned[i]:
            continue
        cluster_items = [items[i]]
        assigned[i] = True
        for j in range(i + 1, len(items)):
            if not assigned[j] and similarities[i][j] >= similarity_threshold:
                cluster_items.append(items[j])
                assigned[j] = True
        cluster_records = [r for _, r in cluster_items]
        task_tags = tuple(_majority_tags(cluster_records, "task_tags"))
        tool_tags = tuple(_majority_tags(cluster_records, "tool_tags"))
        clusters.append({
            "signature": (task_tags, tool_tags),
            "items": cluster_items,
        })
    return clusters


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


def _cluster_skill_id(cluster: Dict[str, Any]) -> str:
    task_tags, tool_tags = cluster["signature"]
    exp_ids = sorted([exp_id for exp_id, _ in cluster["items"]], key=_exp_sort_key)
    representative = exp_ids[0] if exp_ids else "empty"
    digest_input = "|".join(list(task_tags) + list(tool_tags) + [representative])
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:8]
    return f"skill_{_slugify(task_tags)}_{_slugify(tool_tags)}_{digest}"


def _read_rollout_evidence(records: List[Dict[str, Any]], max_rollouts: int = 4, max_chars: int = 12000) -> str:
    chunks = []
    seen_paths = set()
    for record in records:
        for rollout in record.get("evidence_rollouts", []):
            path = rollout.get("traj_path")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            if len(chunks) >= max_rollouts:
                break
            text = ""
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = "(trajectory file unavailable)"
            if len(text) > max_chars // max(1, max_rollouts):
                text = text[: max_chars // max(1, max_rollouts)] + "\n...[truncated]"
            chunks.append(
                "### {rollout_id} | success={success} | score={score}\n{content}".format(
                    rollout_id=rollout.get("rollout_id", "unknown_rollout"),
                    success=rollout.get("success"),
                    score=rollout.get("accuracy_score"),
                    content=text,
                )
            )
    return "\n\n".join(chunks) or "(No rollout evidence available.)"


_EVIDENCE_BOUNDED_TERMS = (
    "object detection",
    "detector",
    "ocr",
    "segmentation",
    "bfs",
    "a*",
    "astar",
    "dijkstra",
)


def _sanitize_evidence_bounded_skill(skill_content: str, support_text: str) -> str:
    """Remove named methods that the LLM introduced without supporting evidence."""
    if not skill_content:
        return skill_content
    support_lower = (support_text or "").lower()
    lines = []
    replaced = False
    replacement_cache = set()

    for line in skill_content.splitlines():
        line_lower = line.lower()
        unsupported = [
            term for term in _EVIDENCE_BOUNDED_TERMS
            if term in line_lower and term not in support_lower
        ]
        if unsupported:
            prefix_match = re.match(r"^(\s*(?:[-*]\s*)?)", line)
            prefix = prefix_match.group(1) if prefix_match else ""
            replacement = (
                prefix
                + "Use only extraction or validation steps supported by the linked experiences; "
                + "keep unsupported methods as placeholders."
            )
            if replacement not in replacement_cache:
                lines.append(replacement)
                replacement_cache.add(replacement)
            replaced = True
        else:
            lines.append(line)

    if replaced:
        print("  [Skill Cluster] Sanitized unsupported named methods from generated skill.")
    return "\n".join(lines).strip()


def _fallback_skill_content(skill_id: str, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    linked = metadata.get("linked_experience_ids", [])
    task_tags = metadata.get("task_tags", [])
    tool_tags = metadata.get("tool_tags", [])
    experience_lines = "\n".join(
        f"- {format_experience_for_prompt(record.get('id', f'E{i}'), record)}"
        for i, record in enumerate(records)
    )
    return f"""---
name: {skill_id}
description: |
  Auto-generated fallback skill for {', '.join(task_tags) or 'general'} tasks.
version: 1.0.0
task_tags: {task_tags}
tool_tags: {tool_tags}
linked_experience_ids: {linked}
---

# {skill_id}

## When to Use
Use this skill when the task matches task tags `{task_tags}` or tool tags `{tool_tags}`.

## Workflow
1. Inspect the target visual state and identify whether a tool-backed check is needed.
2. Derive any grid, table, bounding box, coordinate, recognized text, object list, or option state from the current image and exact problem text before using it.
3. Follow the linked experiences below as concrete constraints.
4. If the task is ambiguous, use the tagged tools `{tool_tags}` before answering.

## Failure Checks
- Do not hand-write task-specific visual structures from memory or from a previous case.
- Treat a tool result as invalid if its structured input was not derived from the current image/problem text.

## Linked Experiences
{experience_lines}
"""


def _failure_guardrail_skill_content(skill_id: str, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    """For all-failure clusters, create guardrails instead of inventing a workflow."""
    linked = metadata.get("linked_experience_ids", [])
    task_tags = metadata.get("task_tags", [])
    tool_tags = metadata.get("tool_tags", [])
    experience_lines = []
    for i, record in enumerate(records):
        exp_id = record.get("id") or (linked[i] if i < len(linked) else f"E{i}")
        experience_lines.append(f"- [{exp_id}] {record.get('content', '')}")
    return f"""---
name: {skill_id}
description: |
  Failure guardrail skill for avoiding repeated errors in {', '.join(task_tags) or 'similar'} tasks.
version: 1.0.0
task_tags: {task_tags}
tool_tags: {tool_tags}
linked_experience_ids: {linked}
---

# Failure Guardrails

## When to Use
- Use only when the task matches these tags: `{task_tags}`.
- This skill is not a recipe. It records failure patterns to avoid.

## Guardrails
{os.linesep.join(experience_lines)}

## How to Apply
1. Check whether your planned step matches one of the failure conditions.
2. If it matches, revise that step before executing it.
3. Do not introduce new algorithms, tools, coordinates, object lists, or constants that are not supported by the current image/problem text.
"""


def cluster_experiences_to_skills(
    experiences: ExperienceLibrary,
    skill_library_path: str,
    llm: Any,
    args: Any = None,
) -> ExperienceLibrary:
    """Generate/update modular skills from experience clusters and link records back."""
    normalized = normalize_experience_library(experiences)
    if not normalized or not skill_library_path:
        return normalized

    index = load_skill_index(skill_library_path)
    index.setdefault("schema_version", 1)
    index.setdefault("skills", {})

    out_dir = skills_dir(skill_library_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_clusters = int(getattr(args, "skill_cluster_max_clusters", 16) or 16)
    min_exps = int(getattr(args, "skill_cluster_min_experiences", 2) or 2)
    clusters = _cluster_by_tags_and_content(normalized)
    clusters = [cluster for cluster in clusters if len(cluster["items"]) >= min_exps]
    clusters.sort(key=lambda c: (-len(c["items"]), str(c["signature"])))
    clusters = clusters[:max_clusters]

    updated = {}
    for exp_id, record in normalized.items():
        cleaned = dict(record)
        cleaned["skill_ids"] = []
        updated[exp_id] = normalize_experience_record(exp_id, cleaned)
    active_skill_ids = set()
    for cluster in clusters:
        skill_id = _cluster_skill_id(cluster)
        active_skill_ids.add(skill_id)
        items = cluster["items"]
        exp_ids = [exp_id for exp_id, _ in items]
        records = [updated[exp_id] for exp_id in exp_ids if exp_id in updated]
        all_failure_cluster = all(record.get("kind") == "failure" for record in records)
        task_tags = sorted({tag for record in records for tag in record.get("task_tags", [])})
        tool_tags = sorted({tag for record in records for tag in record.get("tool_tags", [])})
        confidence = round(sum(record.get("confidence", 0.5) for record in records) / max(1, len(records)), 3)

        metadata = {
            "skill_id": skill_id,
            "path": str(out_dir / f"{skill_id}.md"),
            "task_tags": task_tags,
            "tool_tags": tool_tags,
            "linked_experience_ids": exp_ids,
            "confidence": confidence,
            "trigger_text": " ".join(task_tags + tool_tags),
            "updated_at": datetime.now().isoformat(),
        }

        exp_text = "\n\n".join(
            f"[{exp_id}] {format_experience_for_prompt(exp_id, updated[exp_id])}"
            for exp_id in exp_ids
            if exp_id in updated
        )
        rollout_evidence = _read_rollout_evidence(records)
        support_text = f"{exp_text}\n\n{rollout_evidence}"
        prompt = GENERATE_CLUSTER_SKILL_PROMPT.format(
            skill_id=skill_id,
            task_tags=", ".join(task_tags) or "general",
            tool_tags=", ".join(tool_tags) or "reasoning",
            experience_ids=", ".join(exp_ids),
            experiences=exp_text,
            rollout_evidence=rollout_evidence,
        )

        if all_failure_cluster:
            skill_content = _failure_guardrail_skill_content(skill_id, records, metadata)
        else:
            try:
                skill_content = llm.chat(prompt, max_tokens=4096).strip()
                skill_content = re.sub(r"^```markdown\n", "", skill_content)
                skill_content = re.sub(r"^```\n", "", skill_content)
                skill_content = re.sub(r"\n```$", "", skill_content).strip()
            except Exception as e:
                print(f"  [Skill Cluster] Warning: failed to generate {skill_id}: {e}")
                skill_content = ""

        if not skill_content:
            skill_content = _fallback_skill_content(skill_id, records, metadata)
        skill_content = _sanitize_evidence_bounded_skill(skill_content, support_text)

        skill_path = Path(metadata["path"])
        skill_path.write_text(skill_content, encoding="utf-8")

        index["skills"][skill_id] = metadata

        for exp_id in exp_ids:
            record = dict(updated[exp_id])
            skill_ids = list(dict.fromkeys(record.get("skill_ids", []) + [skill_id]))
            record["skill_ids"] = skill_ids
            updated[exp_id] = normalize_experience_record(exp_id, record)

    _prune_stale_skills(index, active_skill_ids, out_dir)
    save_skill_index(skill_library_path, index)
    _write_skill_manifest(skill_library_path, index)
    print(f"  [Skill Cluster] Generated/updated {len(clusters)} modular skills in {out_dir}")
    return updated


def _prune_stale_skills(index: SkillIndex, active_skill_ids: set, out_dir: Path) -> None:
    """Remove skills that no longer correspond to current experience clusters."""
    skills = index.get("skills", {})
    if not isinstance(skills, dict):
        index["skills"] = {}
        return

    out_dir_resolved = out_dir.resolve()
    for skill_id, item in list(skills.items()):
        if skill_id in active_skill_ids:
            continue
        path = item.get("path") if isinstance(item, dict) else None
        if path:
            try:
                skill_path = Path(path).resolve()
                skill_path.relative_to(out_dir_resolved)
                skill_path.unlink(missing_ok=True)
            except Exception:
                pass
        del skills[skill_id]


def _write_skill_manifest(skill_library_path: str, index: SkillIndex) -> None:
    lines = [
        "---",
        "name: modular_skill_library",
        "description: |",
        "  Manifest for modular skills generated from clustered experiences.",
        "version: 1.0.0",
        "---",
        "",
        "# Modular Skill Library",
        "",
        "Use `skill_index.json` for selection. Individual skills live under `skills/`.",
        "",
    ]
    for skill_id, item in sorted(index.get("skills", {}).items()):
        rel_path = os.path.relpath(item.get("path", ""), _skill_root(skill_library_path))
        lines.append(f"- **{skill_id}** (`{rel_path}`): {item.get('trigger_text', '')}")
    Path(skill_library_path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


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
        f"trigger: {item.get('trigger_text', '')}",
        f"task_tags: {', '.join(item.get('task_tags', []))}",
        f"tool_tags: {', '.join(item.get('tool_tags', []))}",
        f"linked_experience_ids: {', '.join(item.get('linked_experience_ids', []))}",
        _truncate_text(content, 4000),
    ]
    return "\n".join(part for part in fields if part is not None)


def _cosine_scores(query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_embedding.reshape(1, -1)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    m_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    m_norm[m_norm == 0] = 1.0
    return ((q / q_norm) @ (matrix / m_norm).T)[0]


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


def retrieve_skills_for_task(
    skill_library_path: Optional[str],
    query_text: str,
    retrieval_info: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    min_similarity: float = 0.0,
    available_experience_ids: Optional[Iterable[str]] = None,
    embedding_model: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    enable_cache: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Embedding-retrieve modular skills, retrieving top_k skills per subtask query."""
    empty_info = {
        "retrieval_method": "embedding",
        "top_k_per_query": top_k,
        "retrieved_skill_ids": [],
        "retrieval_details": [],
    }
    if not skill_library_path or top_k <= 0:
        return [], empty_info

    index = load_skill_index(skill_library_path)
    skills = index.get("skills", {})
    if not skills:
        return [], empty_info

    available_ids = set(available_experience_ids) if available_experience_ids is not None else None
    skill_texts = {}
    skill_contents = {}
    for skill_id, item in skills.items():
        linked_ids = item.get("linked_experience_ids", []) if isinstance(item, dict) else []
        if available_ids is not None and not any(exp_id in available_ids for exp_id in linked_ids):
            continue
        content = _read_skill_content(item)
        if not content:
            continue
        skill_contents[skill_id] = content
        skill_texts[skill_id] = _skill_retrieval_text(skill_id, item, content)

    if not skill_texts:
        return [], empty_info

    try:
        from .experience_retriever import ExperienceRetriever

        retriever = ExperienceRetriever(
            experiences=skill_texts,
            embedding_model=embedding_model or os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=embedding_api_key or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            embedding_endpoint=embedding_endpoint or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            cache_dir=str(_skill_root(skill_library_path)),
            enable_cache=enable_cache,
            experience_library_path=str(skill_index_path(skill_library_path)),
        )
    except Exception as e:
        print(f"  [Skill Retrieval] Warning: failed to initialize embedding retriever: {e}")
        empty_info["error"] = str(e)
        return [], empty_info

    queries = _queries_from_retrieval_info(query_text, retrieval_info)
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    details = []
    for query_item in queries:
        query = query_item["query"]
        query_embedding = retriever._generate_embedding(query)
        if query_embedding is None or not retriever._experience_embeddings:
            details.append({
                "query": query,
                "type": query_item["type"],
                "retrieved_skill_ids": [],
                "count": 0,
            })
            continue

        skill_ids = list(retriever._experience_embeddings.keys())
        matrix = np.vstack([retriever._experience_embeddings[sid] for sid in skill_ids])
        scores = _cosine_scores(query_embedding, matrix)
        ranked = [
            (skill_ids[i], float(scores[i]))
            for i in range(len(skill_ids))
            if float(scores[i]) >= min_similarity
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        top_ranked = ranked[:top_k]
        details.append({
            "query": query,
            "type": query_item["type"],
            "retrieved_skill_ids": [sid for sid, _ in top_ranked],
            "scores": {sid: score for sid, score in top_ranked},
            "count": len(top_ranked),
        })

        for rank, (skill_id, score) in enumerate(top_ranked, 1):
            item = skills[skill_id]
            entry = candidate_by_id.get(skill_id)
            if entry is None:
                entry = {
                    "skill_id": skill_id,
                    "score": score,
                    "metadata": item,
                    "content": skill_contents[skill_id],
                    "matched_queries": [],
                }
                candidate_by_id[skill_id] = entry
            entry["score"] = max(float(entry.get("score", 0.0)), score)
            entry["matched_queries"].append({
                "query": query,
                "type": query_item["type"],
                "rank": rank,
                "score": score,
            })

    candidates = sorted(candidate_by_id.values(), key=lambda item: item.get("score", 0.0), reverse=True)
    info = {
        "retrieval_method": "embedding",
        "top_k_per_query": top_k,
        "retrieved_skill_ids": [item["skill_id"] for item in candidates],
        "retrieval_details": details,
        "total_unique_skills": len(candidates),
    }
    return candidates, info


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _format_skill_candidate_for_rerank(skill: Dict[str, Any], max_chars: int = 1800) -> str:
    metadata = skill.get("metadata", {}) if isinstance(skill, dict) else {}
    skill_id = skill.get("skill_id", "") if isinstance(skill, dict) else ""
    fields = [
        f"[{skill_id}] score={float(skill.get('score', 0.0) or 0.0):.3f}",
        f"description: {metadata.get('description', '')}",
        f"task_tags: {', '.join(metadata.get('task_tags', []) or [])}",
        f"tool_tags: {', '.join(metadata.get('tool_tags', []) or [])}",
        f"linked_experience_ids: {', '.join(metadata.get('linked_experience_ids', []) or [])}",
        "content:",
        _truncate_text(skill.get("content", ""), max_chars),
    ]
    return "\n".join(fields)


def _format_experience_candidate_for_rerank(exp_id: str, record: Dict[str, Any], max_chars: int = 1200) -> str:
    text = format_experience_for_prompt(exp_id, record)
    return f"[{exp_id}]\n{_truncate_text(text, max_chars)}"


def rerank_memory_with_llm(
    query_text: str,
    candidate_skills: List[Dict[str, Any]],
    candidate_experiences: ExperienceLibrary,
    llm: Any,
    max_skills: int = 2,
    max_experiences: int = 8,
    debug_dir: Optional[str] = None,
    question_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], ExperienceLibrary, Dict[str, Any]]:
    """Use an LLM to select useful skills/experiences from embedding candidates."""
    max_skills = max(0, int(max_skills or 0))
    max_experiences = max(0, int(max_experiences or 0))
    normalized_experiences = normalize_experience_library(candidate_experiences)

    skill_by_id = {
        str(skill.get("skill_id")): skill
        for skill in candidate_skills
        if isinstance(skill, dict) and skill.get("skill_id")
    }
    exp_by_id = {str(exp_id): record for exp_id, record in normalized_experiences.items()}

    fallback_skills = candidate_skills[:max_skills]
    fallback_experiences = dict(list(exp_by_id.items())[:max_experiences])
    info: Dict[str, Any] = {
        "rerank_used": False,
        "question_id": question_id,
        "candidate_skill_ids": list(skill_by_id.keys()),
        "candidate_experience_ids": list(exp_by_id.keys()),
        "selected_skill_ids": [skill.get("skill_id") for skill in fallback_skills],
        "selected_experience_ids": list(fallback_experiences.keys()),
        "fallback": False,
    }

    if not skill_by_id and not exp_by_id:
        return [], {}, info
    if llm is None:
        info["fallback"] = True
        info["error"] = "llm_not_available"
        return fallback_skills, fallback_experiences, info

    skill_text = "\n\n".join(
        _format_skill_candidate_for_rerank(skill)
        for skill in candidate_skills
        if skill.get("skill_id") in skill_by_id
    ) or "(none)"
    exp_text = "\n\n".join(
        _format_experience_candidate_for_rerank(exp_id, record)
        for exp_id, record in exp_by_id.items()
    ) or "(none)"
    prompt = _load_memory_prompt("memory_rerank").format(
        query=query_text,
        skills=skill_text,
        experiences=exp_text,
        max_skills=max_skills,
        max_experiences=max_experiences,
    )

    try:
        response = llm.chat(prompt, max_tokens=2048, temperature=0.0)
        parsed = _extract_json_object(response)
        if "skill_ids" not in parsed or "experience_ids" not in parsed:
            raise ValueError("rerank response missing skill_ids or experience_ids")
        raw_skill_ids = parsed.get("skill_ids", [])
        raw_exp_ids = parsed.get("experience_ids", [])
        if not isinstance(raw_skill_ids, list) or not isinstance(raw_exp_ids, list):
            raise ValueError("rerank response skill_ids/experience_ids must be lists")

        skill_ids = [
            str(skill_id)
            for skill_id in raw_skill_ids
            if str(skill_id) in skill_by_id
        ][:max_skills]
        exp_ids = [
            str(exp_id)
            for exp_id in raw_exp_ids
            if str(exp_id) in exp_by_id
        ][:max_experiences]

        selected_skills = [skill_by_id[skill_id] for skill_id in skill_ids]
        selected_experiences = {exp_id: exp_by_id[exp_id] for exp_id in exp_ids}

        info.update({
            "rerank_used": True,
            "selected_skill_ids": skill_ids,
            "selected_experience_ids": exp_ids,
            "notes": parsed.get("notes", ""),
            "fallback": False,
        })
        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            Path(debug_dir, "tt_memory_rerank.json").write_text(
                json.dumps({"prompt": prompt, "response": response, "info": info}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return selected_skills, selected_experiences, info
    except Exception as e:
        info.update({
            "fallback": True,
            "error": str(e),
            "selected_skill_ids": [skill.get("skill_id") for skill in fallback_skills],
            "selected_experience_ids": list(fallback_experiences.keys()),
        })
        print(f"  [Memory Rerank] Warning: LLM rerank failed, using embedding order: {e}")
        return fallback_skills, fallback_experiences, info


def collect_linked_experiences(selected_skills: List[Dict[str, Any]], experiences: ExperienceLibrary) -> ExperienceLibrary:
    normalized = normalize_experience_library(experiences)
    linked = {}
    for skill in selected_skills:
        for exp_id in skill.get("metadata", {}).get("linked_experience_ids", []):
            if exp_id in normalized:
                linked[exp_id] = normalized[exp_id]
    return linked


def format_selected_skills_for_prompt(selected_skills: List[Dict[str, Any]]) -> str:
    chunks = []
    for skill in selected_skills:
        chunks.append(f"<!-- {skill['skill_id']} score={skill['score']:.3f} -->\n{skill['content']}")
    return "\n\n".join(chunks)


def update_memory_usage_after_sample(
    experience_library_path: Optional[str],
    skill_library_path: Optional[str],
    sample_dir: str,
    sample: Dict[str, Any],
    rollout_results: List[Dict[str, Any]],
) -> None:
    usage_path = Path(sample_dir) / "tt_memory_usage.json"
    if not usage_path.exists() or not experience_library_path:
        return
    try:
        usage = json.loads(usage_path.read_text(encoding="utf-8"))
    except Exception:
        return

    from .experience_utils import EXPERIENCE_LIBRARY_LOCK, load_experiences, save_library

    with EXPERIENCE_LIBRARY_LOCK:
        experiences = load_experiences(experience_library_path)
        selected_skill_ids = usage.get("selected_skill_ids", [])
        used_exp_ids = usage.get("retrieved_experience_ids", [])
        similarity_scores = usage.get("similarity_scores", {})
        success = any(float(result.get("accuracy_score", 0.0) or 0.0) == 1.0 for result in rollout_results if isinstance(result, dict))

        valid_used_exp_ids = []
        for exp_id in used_exp_ids:
            if exp_id not in experiences:
                continue
            valid_used_exp_ids.append(exp_id)
            record = dict(experiences[exp_id])
            stats = dict(record.get("stats", {}))
            stats["used_count"] = int(stats.get("used_count", 0) or 0) + 1
            key = "success_after_used" if success else "failure_after_used"
            stats[key] = int(stats.get(key, 0) or 0) + 1
            record["stats"] = stats
            # Similarity-weighted confidence: higher similarity → larger update
            conf = float(record.get("confidence", 0.5) or 0.5)
            sim = float(similarity_scores.get(exp_id, 0.5))
            if success:
                conf = min(1.0, conf + 0.05 * sim)
            else:
                conf = max(0.0, conf - 0.1 * sim)
            record["confidence"] = round(conf, 4)
            experiences[exp_id] = normalize_experience_record(exp_id, record)

        save_library(experience_library_path, experiences)

        if skill_library_path and selected_skill_ids:
            index = load_skill_index(skill_library_path)
            for skill_id in selected_skill_ids:
                item = index.get("skills", {}).get(skill_id)
                if not item:
                    continue
                stats = dict(item.get("stats", {}))
                stats["used_count"] = int(stats.get("used_count", 0) or 0) + 1
                key = "success_after_used" if success else "failure_after_used"
                stats[key] = int(stats.get(key, 0) or 0) + 1
                item["stats"] = stats
                if not success:
                    item["needs_refine"] = True
                    item["linked_experience_ids"] = list(dict.fromkeys(item.get("linked_experience_ids", []) + valid_used_exp_ids))
                index["skills"][skill_id] = item
            save_skill_index(skill_library_path, index)
