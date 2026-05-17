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

from prompts.skill_prompts import GENERATE_CLUSTER_SKILL_PROMPT

from .experience_schema import (
    ExperienceLibrary,
    format_experience_for_prompt,
    get_experience_content,
    normalize_experience_library,
    normalize_experience_record,
)


SkillIndex = Dict[str, Any]


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
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _record_signature(record: Dict[str, Any]) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    task_tags = tuple(sorted(record.get("task_tags", []) or ["general"]))
    tool_tags = tuple(sorted(record.get("tool_tags", []) or ["reasoning"]))
    kind = record.get("kind", "success")
    return kind, task_tags, tool_tags


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


def _cluster_by_tags_and_content(
    experiences: ExperienceLibrary,
    similarity_threshold: float = 0.74,
) -> List[Dict[str, Any]]:
    normalized = normalize_experience_library(experiences)
    groups: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], List[Tuple[str, Dict[str, Any]]]] = {}
    for exp_id, record in normalized.items():
        groups.setdefault(_record_signature(record), []).append((exp_id, record))

    clusters = []
    for signature, items in groups.items():
        if len(items) <= 1:
            clusters.append({"signature": signature, "items": items})
            continue

        records = [record for _, record in items]
        embeddings = _try_content_embeddings(records)
        if embeddings is None:
            clusters.append({"signature": signature, "items": items})
            continue

        assigned = [False] * len(items)
        similarities = _cosine_similarity_matrix(embeddings)
        for i, item in enumerate(items):
            if assigned[i]:
                continue
            cluster_items = [item]
            assigned[i] = True
            for j in range(i + 1, len(items)):
                if not assigned[j] and similarities[i][j] >= similarity_threshold:
                    cluster_items.append(items[j])
                    assigned[j] = True
            clusters.append({"signature": signature, "items": cluster_items})
    return clusters


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return normalized @ normalized.T


def _cluster_skill_id(cluster: Dict[str, Any]) -> str:
    _, task_tags, tool_tags = cluster["signature"]
    exp_ids = [exp_id for exp_id, _ in cluster["items"]]
    digest_input = "|".join(list(task_tags) + list(tool_tags) + exp_ids)
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


def _fallback_skill_content(skill_id: str, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    linked = metadata.get("linked_experience_ids", [])
    task_tags = metadata.get("task_tags", [])
    tool_tags = metadata.get("tool_tags", [])
    visual_tags = metadata.get("visual_tags", [])
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
visual_tags: {visual_tags}
linked_experience_ids: {linked}
---

# {skill_id}

## When to Use
Use this skill when the task matches task tags `{task_tags}` or visual tags `{visual_tags}`.

## Workflow
1. Inspect the target visual state and identify whether a tool-backed check is needed.
2. Follow the linked experiences below as concrete constraints.
3. If the task is ambiguous, use the tagged tools `{tool_tags}` before answering.

## Linked Experiences
{experience_lines}
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
    min_exps = int(getattr(args, "skill_cluster_min_experiences", 1) or 1)
    clusters = _cluster_by_tags_and_content(normalized)
    clusters = [cluster for cluster in clusters if len(cluster["items"]) >= min_exps]
    clusters.sort(key=lambda c: (-len(c["items"]), str(c["signature"])))
    clusters = clusters[:max_clusters]

    updated = dict(normalized)
    for cluster in clusters:
        skill_id = _cluster_skill_id(cluster)
        items = cluster["items"]
        exp_ids = [exp_id for exp_id, _ in items]
        records = [updated[exp_id] for exp_id in exp_ids if exp_id in updated]
        task_tags = sorted({tag for record in records for tag in record.get("task_tags", [])})
        tool_tags = sorted({tag for record in records for tag in record.get("tool_tags", [])})
        visual_tags = sorted({tag for record in records for tag in record.get("visual_tags", [])})
        failure_modes = sorted({record.get("failure_mode") for record in records if record.get("failure_mode")})
        confidence = round(sum(record.get("confidence", 0.5) for record in records) / max(1, len(records)), 3)
        evidence_rollouts = []
        for record in records:
            for rollout in record.get("evidence_rollouts", []):
                if rollout not in evidence_rollouts:
                    evidence_rollouts.append(rollout)

        metadata = {
            "skill_id": skill_id,
            "path": str(out_dir / f"{skill_id}.md"),
            "task_tags": task_tags,
            "tool_tags": tool_tags,
            "visual_tags": visual_tags,
            "failure_modes": failure_modes,
            "linked_experience_ids": exp_ids,
            "confidence": confidence,
            "evidence_rollouts": evidence_rollouts[:8],
            "trigger_text": " ".join(task_tags + tool_tags + visual_tags + failure_modes),
            "updated_at": datetime.now().isoformat(),
        }

        exp_text = "\n\n".join(
            f"[{exp_id}] {format_experience_for_prompt(exp_id, updated[exp_id])}"
            for exp_id in exp_ids
            if exp_id in updated
        )
        rollout_evidence = _read_rollout_evidence(records)
        prompt = GENERATE_CLUSTER_SKILL_PROMPT.format(
            skill_id=skill_id,
            task_tags=", ".join(task_tags) or "general",
            tool_tags=", ".join(tool_tags) or "reasoning",
            visual_tags=", ".join(visual_tags) or "unspecified",
            failure_modes=", ".join(failure_modes) or "none",
            experience_ids=", ".join(exp_ids),
            experiences=exp_text,
            rollout_evidence=rollout_evidence,
        )

        try:
            skill_content = llm.chat(prompt, max_tokens=8192).strip()
            skill_content = re.sub(r"^```markdown\n", "", skill_content)
            skill_content = re.sub(r"^```\n", "", skill_content)
            skill_content = re.sub(r"\n```$", "", skill_content).strip()
        except Exception as e:
            print(f"  [Skill Cluster] Warning: failed to generate {skill_id}: {e}")
            skill_content = ""

        if not skill_content:
            skill_content = _fallback_skill_content(skill_id, records, metadata)

        skill_path = Path(metadata["path"])
        skill_path.write_text(skill_content, encoding="utf-8")

        old_stats = index["skills"].get(skill_id, {}).get("stats", {})
        metadata["stats"] = {
            "used_count": int(old_stats.get("used_count", 0) or 0),
            "success_after_used": int(old_stats.get("success_after_used", 0) or 0),
            "failure_after_used": int(old_stats.get("failure_after_used", 0) or 0),
        }
        index["skills"][skill_id] = metadata

        for exp_id in exp_ids:
            record = dict(updated[exp_id])
            skill_ids = list(dict.fromkeys(record.get("skill_ids", []) + [skill_id]))
            record["skill_ids"] = skill_ids
            updated[exp_id] = normalize_experience_record(exp_id, record)

    save_skill_index(skill_library_path, index)
    _write_skill_manifest(skill_library_path, index)
    print(f"  [Skill Cluster] Generated/updated {len(clusters)} modular skills in {out_dir}")
    return updated


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


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()))


def select_skills_for_task(
    skill_library_path: Optional[str],
    query_text: str,
    retrieval_info: Optional[Dict[str, Any]] = None,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    index = load_skill_index(skill_library_path)
    skills = index.get("skills", {})
    if not skills:
        return []

    subtask_texts = []
    if retrieval_info:
        for subtask in retrieval_info.get("subtasks", []):
            if isinstance(subtask, dict):
                subtask_texts.append(str(subtask.get("query") or subtask.get("description") or ""))
            else:
                subtask_texts.append(str(subtask))
    query_tokens = _tokenize(" ".join([query_text] + subtask_texts))

    scored = []
    for skill_id, item in skills.items():
        trigger = " ".join(
            [item.get("trigger_text", "")]
            + item.get("task_tags", [])
            + item.get("tool_tags", [])
            + item.get("visual_tags", [])
            + item.get("failure_modes", [])
        )
        skill_tokens = _tokenize(trigger)
        overlap = len(query_tokens & skill_tokens)
        confidence = float(item.get("confidence", 0.5) or 0.5)
        linked_count = len(item.get("linked_experience_ids", []))
        score = overlap * 2.0 + confidence + min(linked_count, 5) * 0.1
        if score > 0:
            scored.append((score, skill_id, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, skill_id, item in scored[:top_k]:
        path = item.get("path", "")
        content = ""
        try:
            content = Path(path).read_text(encoding="utf-8")
        except Exception:
            continue
        selected.append({"skill_id": skill_id, "score": score, "metadata": item, "content": content})
    return selected


def collect_linked_experiences(selected_skills: List[Dict[str, Any]], experiences: ExperienceLibrary) -> ExperienceLibrary:
    normalized = normalize_experience_library(experiences)
    linked = {}
    for skill in selected_skills:
        for exp_id in skill.get("metadata", {}).get("linked_experience_ids", []):
            if exp_id in normalized:
                linked[exp_id] = normalized[exp_id]
    return linked


def rerank_experiences_for_skills(
    direct_experiences: ExperienceLibrary,
    linked_experiences: ExperienceLibrary,
    selected_skill_ids: List[str],
    max_items: int = 8,
) -> ExperienceLibrary:
    combined = normalize_experience_library({**direct_experiences, **linked_experiences})
    scored = []
    for exp_id, record in combined.items():
        linked_bonus = 1.0 if any(skill_id in record.get("skill_ids", []) for skill_id in selected_skill_ids) else 0.0
        failure_bonus = 0.2 if record.get("kind") == "failure" else 0.0
        score = linked_bonus + failure_bonus + float(record.get("confidence", 0.5) or 0.5)
        scored.append((score, exp_id, record))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {exp_id: record for _, exp_id, record in scored[:max_items]}


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

    from .experience_utils import load_experiences, save_library

    experiences = load_experiences(experience_library_path)
    selected_skill_ids = usage.get("selected_skill_ids", [])
    used_exp_ids = usage.get("retrieved_experience_ids", [])
    success = any(float(result.get("accuracy_score", 0.0) or 0.0) == 1.0 for result in rollout_results if isinstance(result, dict))

    for exp_id in used_exp_ids:
        if exp_id not in experiences:
            continue
        record = dict(experiences[exp_id])
        stats = dict(record.get("stats", {}))
        stats["used_count"] = int(stats.get("used_count", 0) or 0) + 1
        key = "success_after_used" if success else "failure_after_used"
        stats[key] = int(stats.get(key, 0) or 0) + 1
        record["stats"] = stats
        experiences[exp_id] = normalize_experience_record(exp_id, record)

    failure_exp_id = None
    if not success and selected_skill_ids:
        next_id = _next_experience_id(experiences)
        failure_exp_id = next_id
        task = sample.get("task") or "general"
        evidence_rollouts = []
        for idx, result in enumerate(rollout_results):
            if not isinstance(result, dict):
                continue
            rollout_idx = result.get("_rollout_idx", idx)
            rollout_dir = Path(sample_dir) / f"rollout_{rollout_idx}" if len(rollout_results) > 1 else Path(sample_dir)
            evidence_rollouts.append({
                "doc_id": sample.get("doc_id", sample.get("question_id", "")),
                "rollout_id": f"{sample.get('doc_id', sample.get('question_id', 'sample'))}:rollout_{rollout_idx}",
                "rollout_idx": rollout_idx,
                "traj_path": str(rollout_dir / "traj.jsonl"),
                "accuracy_score": result.get("accuracy_score"),
                "success": False,
                "final_answer": result.get("final_answer"),
            })
        experiences[next_id] = normalize_experience_record(
            next_id,
            {
                "kind": "failure",
                "skill_ids": selected_skill_ids,
                "task_tags": [task],
                "tool_tags": [],
                "visual_tags": [],
                "failure_mode": "failed_after_skill_use",
                "confidence": 0.6,
                "action_policy": {
                    "when": "selected skills and experiences still fail",
                    "do": "revisit the linked rollout evidence and add a stricter verification step",
                },
                "evidence_rollouts": evidence_rollouts,
                "content": "When selected skills still lead to a wrong answer, add an explicit verification step instead of trusting the adapted workflow.",
            },
        )
        used_exp_ids.append(next_id)

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
                item["linked_experience_ids"] = list(dict.fromkeys(item.get("linked_experience_ids", []) + used_exp_ids))
                _append_skill_feedback(item, sample, failure_exp_id)
            index["skills"][skill_id] = item
        save_skill_index(skill_library_path, index)


def _append_skill_feedback(item: Dict[str, Any], sample: Dict[str, Any], failure_exp_id: Optional[str]) -> None:
    path = item.get("path")
    if not path or not failure_exp_id:
        return
    feedback = (
        "\n\n## Failure Feedback\n"
        f"- Linked failure experience `{failure_exp_id}` from `{sample.get('doc_id', sample.get('question_id', 'unknown'))}`. "
        "Add an explicit verification step before reusing this skill on similar cases.\n"
    )
    try:
        skill_path = Path(path)
        existing = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
        marker = f"Linked failure experience `{failure_exp_id}`"
        if marker not in existing:
            skill_path.write_text(existing.rstrip() + feedback, encoding="utf-8")
    except Exception:
        pass


def _next_experience_id(experiences: ExperienceLibrary) -> str:
    nums = []
    for exp_id in experiences:
        if isinstance(exp_id, str) and exp_id.startswith("E") and exp_id[1:].isdigit():
            nums.append(int(exp_id[1:]))
    return f"E{max(nums, default=-1) + 1}"
