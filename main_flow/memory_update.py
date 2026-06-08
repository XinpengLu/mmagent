"""Train-time memory update and snapshot utilities."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from experience import (
    EXPERIENCE_LIBRARY_LOCK,
    ExperienceLLM,
    generate_experience_for_sample,
    load_experiences as exp_load_experiences,
    batch_merge as exp_batch_merge,
    save_library as exp_save_library,
    refine_experience_library,
)
from skill import cluster_experiences_to_skills


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
    import shutil

    snapshot_exp_path = os.path.join(snapshot_dir, "experiences.json")
    if args.experience_library and os.path.exists(args.experience_library):
        shutil.copy2(args.experience_library, snapshot_exp_path)
    return _experience_count(snapshot_exp_path)


def _snapshot_skill_memory(args, snapshot_dir):
    import shutil

    metadata = {"skill_word_count": 0, "skill_count": 0}
    if getattr(args, 'skill_library', None) and os.path.exists(args.skill_library):
        shutil.copy2(args.skill_library, os.path.join(snapshot_dir, os.path.basename(args.skill_library)))
        with open(args.skill_library, 'r', encoding='utf-8') as f:
            metadata["skill_word_count"] = len(f.read().split())

    if not getattr(args, 'skill_library', None):
        return metadata

    skill_root = os.path.dirname(args.skill_library)
    skill_index_src = os.path.join(skill_root, "skill_index.json")
    if os.path.exists(skill_index_src):
        skill_index_dst = os.path.join(snapshot_dir, "skill_index.json")
        shutil.copy2(skill_index_src, skill_index_dst)
        with open(skill_index_dst, 'r', encoding='utf-8') as f:
            skill_index = json.load(f)
        metadata["skill_count"] = len(skill_index.get("skills", {}))

    skills_src = os.path.join(skill_root, "skills")
    if os.path.isdir(skills_src):
        shutil.copytree(skills_src, os.path.join(snapshot_dir, "skills"), dirs_exist_ok=True)
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
    generation_workers = getattr(args, 'experience_generation_workers', None)
    if generation_workers is None or generation_workers <= 0:
        generation_workers = getattr(args, 'num_workers', 4) or 4
    return max(1, min(num_tasks, int(generation_workers), 16))


def _generate_batch_experiences(samples_info, args):
    all_experience_ops = []
    successful_samples = 0
    num_tasks = len(samples_info)
    generation_workers = _generation_worker_count(args, num_tasks)
    print(f"  [Large Batch] Experience generation workers: {generation_workers}/{num_tasks}")

    with ThreadPoolExecutor(max_workers=generation_workers) as executor:
        exp_futures = {
            executor.submit(generate_experience_for_sample, sample_info, args): sample_info
            for sample_info in samples_info
        }
        for future in as_completed(exp_futures):
            sample_info = exp_futures[future]
            try:
                result = future.result()
                if result['success']:
                    all_experience_ops.extend(result['experience_ops'])
                    successful_samples += 1
                else:
                    print(f"  Warning: Generation failed for {result['sample_id']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  Warning: Generation failed for {sample_info['question_id']}: {e}")
    return all_experience_ops, successful_samples


def _maybe_refine_experiences(merged, args, is_final):
    if not getattr(args, 'experience_refine', False):
        return merged
    max_items = getattr(args, 'experience_max_items', 256)
    if not (is_final or len(merged) > max_items):
        return merged

    refine_start = time.time()
    merged = refine_experience_library(merged, ExperienceLLM(), debug_dir=args.output_dir)
    refine_time = time.time() - refine_start
    print(f"  [Timing] Experience refine{'(final)' if is_final else ''}: {refine_time:.1f}s")
    return merged


def _merge_experience_ops(all_experience_ops, args, is_final):
    if not (all_experience_ops and getattr(args, 'experience_library_update', False) and args.experience_library):
        return None

    with EXPERIENCE_LIBRARY_LOCK:
        existing = exp_load_experiences(args.experience_library)
        merge_start = time.time()
        merged = exp_batch_merge(
            existing,
            all_experience_ops,
            ExperienceLLM(),
            debug_dir=None,
            experience_limit=getattr(args, 'experience_max_items', 16),
        )
        merge_time = time.time() - merge_start
        merged = _maybe_refine_experiences(merged, args, is_final)

        os.makedirs(os.path.dirname(args.experience_library), exist_ok=True)
        exp_save_library(args.experience_library, merged)
        print(f"  [Large Batch] Merged {len(all_experience_ops)} experience operations into library (final size: {len(merged)})")
        print(f"  [Timing] Experience merge: {merge_time:.1f}s")
        return merged


def _maybe_cluster_skills(merged, args):
    if not (merged is not None and getattr(args, 'skill_enable', False) and getattr(args, 'skill_library', None)):
        return
    skill_start = time.time()
    merged = cluster_experiences_to_skills(merged, args.skill_library, ExperienceLLM(), args)
    exp_save_library(args.experience_library, merged)
    print(f"  [Timing] Skill clustering: {time.time() - skill_start:.1f}s")


def _update_memory_library(all_experience_ops, args, is_final):
    try:
        merged = _merge_experience_ops(all_experience_ops, args, is_final)
        _maybe_cluster_skills(merged, args)
    except Exception as e:
        print(f"  Warning: Failed to merge experiences into library: {e}")


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

    _update_memory_library(all_experience_ops, args, is_final)

    total_time = time.time() - start_time
    print(f"[Large Batch] Completed: {successful_samples}/{len(samples_info)} samples processed successfully (total: {total_time:.1f}s)")
    return successful_samples

class TrainingMemoryUpdater:
    """Handles train-time memory generation and refresh boundaries."""
    def __init__(self, args, memory_runtime):
        self.args = args
        self.memory_runtime = memory_runtime
        self.queue = []
        self.batch_idx = 0
        self.enabled = getattr(args, 'experience_online_generate', False)

    def extend(self, sample_infos):
        if not self.enabled or not sample_infos:
            return
        self.queue.extend(sample_infos)
        self.flush_if_ready()

    def append(self, sample_info):
        if not self.enabled or not sample_info:
            return
        self.queue.append(sample_info)
        self.flush_if_ready()

    def flush_if_ready(self):
        if len(self.queue) >= self.args.experience_samples_per_large_batch:
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
