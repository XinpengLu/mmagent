"""Train-time memory update and snapshot utilities."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from experience import (
    EXPERIENCE_LIBRARY_LOCK,
    ExperienceLLM,
    ExperienceRetriever,
    generate_experience_for_sample,
    load_experiences as exp_load_experiences,
    batch_merge as exp_batch_merge,
    save_library as exp_save_library,
    refine_experience_library,
)
from main_flow.batching import experience_update_interval_samples
from skill import cluster_experiences_to_skills, skill_generation_is_current
from utils.artifacts import (
    ROLLOUT_METRICS_FILENAME,
    TRAJECTORY_FILENAME,
    copy_file_contents,
    copy_tree_contents,
)


EXPERIENCE_GENERATION_MARKER = ".experience_generation_complete.json"
EXPERIENCE_GENERATION_MARKER_VERSION = 2
EXPERIENCE_OPERATIONS_CACHE = ".experience_operations.json"
EXPERIENCE_OPERATIONS_CACHE_VERSION = 2


def _sample_marker_path(sample_info):
    return Path(sample_info["sample_dir"]) / EXPERIENCE_GENERATION_MARKER


def _sample_operations_cache_path(sample_info):
    return Path(sample_info["sample_dir"]) / EXPERIENCE_OPERATIONS_CACHE


def _sample_rollout_signature_payload(sample_info, path_style="posix"):
    sample_dir = Path(sample_info["sample_dir"])
    files = []
    for rollout_dir in sorted(sample_dir.glob("rollout_*")):
        if not rollout_dir.is_dir():
            continue
        for filename in (TRAJECTORY_FILENAME, ROLLOUT_METRICS_FILENAME):
            path = rollout_dir / filename
            if path.exists():
                stat = path.stat()
                relative = path.relative_to(sample_dir).as_posix()
                if path_style == "windows":
                    relative = relative.replace("/", "\\")
                files.append((relative, stat.st_size, stat.st_mtime_ns))
    if not files:
        for filename in (TRAJECTORY_FILENAME, ROLLOUT_METRICS_FILENAME):
            path = sample_dir / filename
            if path.exists():
                stat = path.stat()
                files.append((filename, stat.st_size, stat.st_mtime_ns))
    return files


def _signature_from_payload(files):
    payload = json.dumps(files, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() if files else ""


def _sample_rollout_signature(sample_info):
    return _signature_from_payload(_sample_rollout_signature_payload(sample_info, "posix"))


def _legacy_sample_rollout_signatures(sample_info):
    return {
        _signature_from_payload(_sample_rollout_signature_payload(sample_info, path_style))
        for path_style in ("posix", "windows")
    } - {""}


def _record_question_ids(record):
    ids = set()
    for evidence in record.get("evidence_rollouts") or []:
        if not isinstance(evidence, dict):
            continue
        for key in ("doc_id", "question_id"):
            value = str(evidence.get(key) or "").strip()
            if value:
                ids.add(value)
        rollout_id = str(evidence.get("rollout_id") or "").strip()
        if ":rollout_" in rollout_id:
            ids.add(rollout_id.split(":rollout_", 1)[0])
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    for value in (
        record.get("source_question_id"),
        source.get("doc_id"),
        source.get("question_id"),
    ):
        text = str(value or "").strip()
        if text:
            ids.add(text)
    return ids


def _memory_question_ids(args):
    if not args.experience_library or not os.path.exists(args.experience_library):
        return set()
    try:
        experiences = exp_load_experiences(args.experience_library)
    except Exception:
        return set()
    ids = set()
    for record in experiences.values():
        if isinstance(record, dict):
            ids.update(_record_question_ids(record))
    return ids


def _read_sample_marker(sample_info):
    path = _sample_marker_path(sample_info)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_cached_operations(sample_info):
    path = _sample_operations_cache_path(sample_info)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    version = int(payload.get("version", 0))
    signature = str(payload.get("rollout_signature") or "")
    if version == EXPERIENCE_OPERATIONS_CACHE_VERSION:
        valid_signature = signature == _sample_rollout_signature(sample_info)
    elif version == 1:
        valid_signature = signature in _legacy_sample_rollout_signatures(sample_info)
    else:
        valid_signature = False
    if not valid_signature:
        return None
    operations = payload.get("experience_ops")
    return operations if isinstance(operations, list) else None


def _write_cached_operations(sample_info, operations):
    path = _sample_operations_cache_path(sample_info)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": EXPERIENCE_OPERATIONS_CACHE_VERSION,
        "question_id": sample_info["question_id"],
        "rollout_signature": _sample_rollout_signature(sample_info),
        "experience_ops": list(operations or []),
        "generated_at": datetime.now().isoformat(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _write_sample_marker(sample_info, operation_count, *, migrated=False):
    path = _sample_marker_path(sample_info)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": EXPERIENCE_GENERATION_MARKER_VERSION,
        "question_id": sample_info["question_id"],
        "rollout_signature": _sample_rollout_signature(sample_info),
        "operation_count": max(0, int(operation_count)),
        "migrated_from_memory_evidence": bool(migrated),
        "completed_at": datetime.now().isoformat(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _marker_is_current(sample_info, marker):
    if not marker:
        return False
    version = int(marker.get("version", 0))
    signature = str(marker.get("rollout_signature") or "")
    if version == EXPERIENCE_GENERATION_MARKER_VERSION:
        current = _sample_rollout_signature(sample_info)
        return bool(current and signature == current)
    if version == 1:
        return signature in _legacy_sample_rollout_signatures(sample_info)
    return False


def _filter_completed_memory_samples(sample_infos, args):
    memory_ids = _memory_question_ids(args)
    pending = []
    skipped = 0
    for sample_info in sample_infos:
        marker = _read_sample_marker(sample_info)
        if _marker_is_current(sample_info, marker):
            if int(marker.get("version", 0)) != EXPERIENCE_GENERATION_MARKER_VERSION:
                _write_sample_marker(
                    sample_info,
                    marker.get("operation_count", 0),
                    migrated=bool(marker.get("migrated_from_memory_evidence")),
                )
            skipped += 1
            continue
        question_id = sample_info["question_id"]
        if marker is None and question_id in memory_ids and _sample_rollout_signature(sample_info):
            _write_sample_marker(sample_info, 1, migrated=True)
            skipped += 1
            continue
        pending.append(sample_info)
    if skipped:
        print(f"  [Large Batch] Skipped experience generation for {skipped} completed sample(s).")
    return pending


def _next_snapshot_batch_idx(args):
    snapshot_root = Path(args.output_dir) / "snapshots"
    indices = []
    for path in snapshot_root.glob("batch_*") if snapshot_root.exists() else []:
        if not (path / "metadata.json").exists():
            continue
        suffix = path.name.removeprefix("batch_")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def _experience_count(path):
    if not os.path.exists(path):
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        exp_data = json.load(f)
    if isinstance(exp_data, list):
        return len(exp_data)
    if isinstance(exp_data, dict):
        inner = exp_data.get("experiences", {})
        return len(inner) if isinstance(inner, (list, dict)) else 0
    return 0


def _snapshot_experience_library(args, snapshot_dir):
    snapshot_exp_path = os.path.join(snapshot_dir, "experiences.json")
    if args.experience_library and os.path.exists(args.experience_library):
        copy_file_contents(args.experience_library, snapshot_exp_path)
    return _experience_count(snapshot_exp_path)


def _snapshot_skill_memory(args, snapshot_dir):
    metadata = {"skill_word_count": 0, "skill_count": 0}
    if args.skill_library and os.path.exists(args.skill_library):
        copy_file_contents(args.skill_library, os.path.join(snapshot_dir, os.path.basename(args.skill_library)))
        with open(args.skill_library, 'r', encoding='utf-8') as f:
            metadata["skill_word_count"] = len(f.read().split())

    if not args.skill_library:
        return metadata

    skill_root = os.path.dirname(args.skill_library)
    skill_index_src = os.path.join(skill_root, "skill_index.json")
    if os.path.exists(skill_index_src):
        skill_index_dst = os.path.join(snapshot_dir, "skill_index.json")
        copy_file_contents(skill_index_src, skill_index_dst)
        with open(skill_index_dst, 'r', encoding='utf-8') as f:
            skill_index = json.load(f)
        metadata["skill_count"] = len(skill_index.get("skills", {}))

    skills_src = os.path.join(skill_root, "skills")
    if os.path.isdir(skills_src):
        copy_tree_contents(
            skills_src,
            os.path.join(snapshot_dir, "skills"),
            dirs_exist_ok=True,
        )
    return metadata


def _save_knowledge_snapshot(args, batch_idx):
    from datetime import datetime

    snapshot_dir = os.path.join(args.output_dir, "snapshots", f"batch_{batch_idx:03d}")
    os.makedirs(snapshot_dir, exist_ok=True)

    metadata = {
        "batch_idx": batch_idx,
        "timestamp": datetime.now().isoformat(),
    }

    metadata["experience_count"] = _snapshot_experience_library(args, snapshot_dir)
    metadata.update(_snapshot_skill_memory(args, snapshot_dir))

    with open(os.path.join(snapshot_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  [Snapshot] Saved batch_{batch_idx:03d} snapshot (exp: {metadata['experience_count']}, skills: {metadata['skill_count']})")


def _generation_worker_count(args, num_tasks):
    generation_workers = args.experience_generation_workers or args.num_workers
    return max(1, min(num_tasks, int(generation_workers), 16))


def _generate_batch_experiences(samples_info, args):
    all_experience_ops = []
    successful_samples = []
    pending_samples = []
    reused_samples = 0
    for sample_info in samples_info:
        cached_ops = _read_cached_operations(sample_info)
        if cached_ops is None:
            pending_samples.append(sample_info)
            continue
        all_experience_ops.extend(cached_ops)
        successful_samples.append((sample_info, len(cached_ops)))
        reused_samples += 1
    if reused_samples:
        print(f"  [Large Batch] Reused cached experience operations for {reused_samples} sample(s).")

    num_tasks = len(pending_samples)
    if not num_tasks:
        return all_experience_ops, successful_samples
    generation_workers = _generation_worker_count(args, num_tasks)
    print(f"  [Large Batch] Experience generation workers: {generation_workers}/{num_tasks}")

    with ThreadPoolExecutor(max_workers=generation_workers) as executor:
        exp_futures = {
            executor.submit(generate_experience_for_sample, sample_info, args): sample_info
            for sample_info in pending_samples
        }
        for future in as_completed(exp_futures):
            sample_info = exp_futures[future]
            try:
                result = future.result()
                if result['success']:
                    experience_ops = result.get('experience_ops') or []
                    _write_cached_operations(sample_info, experience_ops)
                    all_experience_ops.extend(experience_ops)
                    successful_samples.append((sample_info, len(experience_ops)))
                else:
                    print(f"  Warning: Generation failed for {result['sample_id']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  Warning: Generation failed for {sample_info['question_id']}: {e}")
    return all_experience_ops, successful_samples


def _maybe_refine_experiences(merged, args, is_final):
    if args.phase != "train":
        return merged
    max_items = args.experience_max_items
    if not (is_final or len(merged) > max_items):
        return merged

    refine_start = time.time()
    merged = refine_experience_library(merged, ExperienceLLM(), debug_dir=args.output_dir)
    refine_time = time.time() - refine_start
    print(f"  [Timing] Experience refine{'(final)' if is_final else ''}: {refine_time:.1f}s")
    return merged


def _merge_experience_ops(all_experience_ops, args, is_final):
    if not (all_experience_ops and args.phase == "train" and args.experience_library):
        return None

    with EXPERIENCE_LIBRARY_LOCK:
        existing = exp_load_experiences(args.experience_library)
        retriever = ExperienceRetriever(
            experiences=existing,
            embedding_model=os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_api_key=os.environ.get("EXPERIENCE_EMBEDDING_API_KEY"),
            embedding_endpoint=os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT"),
            cache_dir=os.path.dirname(args.experience_library),
            enable_cache=True,
            experience_library_path=args.experience_library,
        )
        merge_start = time.time()
        merged = exp_batch_merge(
            existing,
            all_experience_ops,
            ExperienceLLM(),
            debug_dir=None,
            experience_limit=args.experience_max_items,
            retriever=retriever,
        )
        merge_time = time.time() - merge_start
        merged = _maybe_refine_experiences(merged, args, is_final)
        retriever.update_experiences(merged, incremental=True)
        os.makedirs(os.path.dirname(args.experience_library), exist_ok=True)
        exp_save_library(args.experience_library, merged)
        print(f"  [Large Batch] Merged {len(all_experience_ops)} experience operations into library (final size: {len(merged)})")
        print(f"  [Timing] Experience merge: {merge_time:.1f}s")
        return merged


def _maybe_cluster_skills(merged, args):
    if not (merged is not None and args.skill_library and getattr(args, "ablation_skill_generation", True)):
        return
    skill_start = time.time()
    merged = cluster_experiences_to_skills(merged, args.skill_library, ExperienceLLM(), args)
    exp_save_library(args.experience_library, merged)
    print(f"  [Timing] Skill clustering: {time.time() - skill_start:.1f}s")


def _update_memory_library(all_experience_ops, args, is_final):
    try:
        if not all_experience_ops:
            return True
        merged = _merge_experience_ops(all_experience_ops, args, is_final)
        if merged is None:
            return False
    except Exception as e:
        print(f"  Warning: Failed to merge experiences into library: {e}")
        return False

    try:
        _maybe_cluster_skills(merged, args)
    except Exception as e:
        print(
            "  [Skill Checkpoint] Warning: skill update failed after the experience library "
            f"was saved; it will be reconciled on restart: {e}"
        )
    return True


def _reconcile_skill_library_after_resume(args, memory_runtime):
    if (
        not getattr(args, "skill_library", None)
        or not getattr(args, "experience_library", None)
        or not getattr(args, "ablation_skill_generation", True)
        or not os.path.exists(args.experience_library)
    ):
        return False

    try:
        experiences = exp_load_experiences(args.experience_library)
        if not experiences:
            return False
        if skill_generation_is_current(experiences, args.skill_library, args=args):
            print("  [Skill Checkpoint] Skill library is current; skipping regeneration.")
            return False

        print("  [Skill Checkpoint] Skill library is incomplete or stale; reconciling incrementally...")
        updated = cluster_experiences_to_skills(
            experiences,
            args.skill_library,
            ExperienceLLM(),
            args,
        )
        exp_save_library(args.experience_library, updated)
        memory_runtime.reload_after_training_update()
        print("  [Skill Checkpoint] Skill library reconciliation completed.")
        return True
    except Exception as e:
        print(f"  [Skill Checkpoint] Warning: failed to reconcile skill library: {e}")
        return False


def _process_large_batch_experiences(samples_info, args, batch_idx=0, is_final=False):
    if not samples_info:
        return 0

    start_time = time.time()
    print(f"\n[Large Batch {batch_idx}] Processing experience generation for {len(samples_info)} samples...")

    _save_knowledge_snapshot(args, batch_idx)

    gen_start = time.time()
    all_experience_ops, successful_samples = _generate_batch_experiences(samples_info, args)
    gen_time = time.time() - gen_start
    print(f"  [Timing] Experience generation: {gen_time:.1f}s")
    memory_updated = _update_memory_library(all_experience_ops, args, is_final)
    if memory_updated:
        for sample_info, operation_count in successful_samples:
            _write_sample_marker(sample_info, operation_count)

    total_time = time.time() - start_time
    print(f"[Large Batch] Completed: {len(successful_samples)}/{len(samples_info)} samples processed successfully (total: {total_time:.1f}s)")
    return len(successful_samples)

class TrainingMemoryUpdater:
    """Handles train-time memory generation and refresh boundaries."""
    def __init__(self, args, memory_runtime):
        self.args = args
        self.memory_runtime = memory_runtime
        self.queue = []
        self.batch_idx = _next_snapshot_batch_idx(args)
        self.enabled = args.phase == "train" and getattr(args, "ablation_experience_generation", True)
        self.update_interval_samples = (
            experience_update_interval_samples(args) if self.enabled else 1
        )
        _reconcile_skill_library_after_resume(args, memory_runtime)

    def extend(self, sample_infos):
        if not self.enabled or not sample_infos:
            return
        self.queue.extend(_filter_completed_memory_samples(sample_infos, self.args))
        self.flush_if_ready()

    def append(self, sample_info):
        if not self.enabled or not sample_info:
            return
        pending = _filter_completed_memory_samples([sample_info], self.args)
        self.queue.extend(pending)
        self.flush_if_ready()

    def flush_if_ready(self):
        if len(self.queue) >= self.update_interval_samples:
            print(f"\n[Large Batch] Triggering experience generation for {len(self.queue)} samples...")
            self._process_queue(is_final=False)

    def flush_final(self):
        if self.enabled and self.queue:
            print(f"\n[Final Batch] Processing remaining {len(self.queue)} samples...")
            self._process_queue(is_final=True)

    def _process_queue(self, is_final):
        _process_large_batch_experiences(self.queue, self.args, batch_idx=self.batch_idx, is_final=is_final)
        self.batch_idx += 1
        self.queue.clear()
        self.memory_runtime.reload_after_training_update()
