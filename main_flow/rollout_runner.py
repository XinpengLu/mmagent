"""Rollout execution and runtime memory state."""

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from engine.api_processors import process_single_sample, set_global_prompts as set_processors_prompts
from skill import update_memory_usage_after_sample
from utils.env import env_flag, env_int
from utils.transient import (
    result_has_transient_failure,
    text_has_transient_failure,
    rollout_dir_has_transient_failure,
)
from main_flow.memory import (
    initialize_experience_retriever,
    get_sample_metadata,
    prepare_sample_args,
    reload_experiences,
    retrieve_experiences_for_sample,
)
from main_flow.metrics import compute_and_save_sample_summary
from utils.artifacts import (
    ROLLOUT_METRICS_FILENAME,
    SAMPLE_SUMMARY_FILENAME,
    TRAJECTORY_FILENAME,
    clean_rollout_artifacts,
)


logger = logging.getLogger(__name__)


@dataclass
class _PipelineState:
    active_futures: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    rollout_counts: Dict[int, int] = field(default_factory=dict)
    completed_samples: set = field(default_factory=set)
    next_sample_idx: int = 0
    active_retrieval_count: int = 0


def _float_or_default(value, default):
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _apply_validated_memory_sampling_policy(args, sampling_params):
    if not env_flag("ENABLE_VALIDATED_MEMORY_POLICY_LOW_TEMPERATURE", True):
        return
    if getattr(args, "_memory_policy_status", "") != "validated_policy":
        return

    policy_temperature = (
        getattr(args, "_memory_policy_temperature", None)
        if getattr(args, "_memory_policy_temperature", None) is not None
        else os.environ.get("VALIDATED_MEMORY_POLICY_TEMPERATURE", "0.2")
    )
    policy_temperature = _float_or_default(policy_temperature, 0.2)
    sampling_params["temperature"] = min(
        _float_or_default(sampling_params.get("temperature"), policy_temperature),
        policy_temperature,
    )
    if os.environ.get("VALIDATED_MEMORY_POLICY_TOP_P"):
        top_p = _float_or_default(os.environ.get("VALIDATED_MEMORY_POLICY_TOP_P"), None)
        if top_p is not None:
            sampling_params["top_p"] = min(
                _float_or_default(sampling_params.get("top_p"), 1.0),
                top_p,
            )


def _run_single_rollout(sample, args, sampling_params, rollout_idx):
    """Run one rollout for one sample."""
    import random

    rollout_args = prepare_sample_args(args)
    seed = args.seed_base + rollout_idx
    random.seed(seed)

    rollout_sampling_params = sampling_params.copy()
    if 'seed' in rollout_sampling_params:
        rollout_sampling_params['seed'] = seed

    _apply_validated_memory_sampling_policy(rollout_args, rollout_sampling_params)

    result = process_single_sample(sample, rollout_args, rollout_sampling_params, rollout_idx=rollout_idx)
    return result


def _rollout_output_dir(output_dir, question_id, rollout_idx, rollouts_per_sample):
    sample_dir = os.path.join(output_dir, str(question_id))
    if int(rollouts_per_sample or 1) > 1:
        return os.path.join(sample_dir, f"rollout_{rollout_idx}")
    return sample_dir


def _sample_rollout_info(sample, sample_idx, question_id, sample_dir, sample_rollout_results):
    return {
        'sample': sample,
        'sample_idx': sample_idx,
        'question_id': question_id,
        'sample_dir': sample_dir,
        'sample_rollout_results': sample_rollout_results,
    }


def _record_sample_info(all_results, sample_info):
    if not sample_info or not sample_info.get('sample_rollout_results'):
        return False
    all_results.extend(sample_info['sample_rollout_results'])
    return True


def _run_rollout_until_valid(sample, args, sampling_params, rollout_idx, run_single_rollout_func=_run_single_rollout):
    """Run one rollout index until the saved trajectory is free of transient failures."""
    question_id = sample.get("doc_id", sample.get("question_id", "N/A"))
    max_attempts = 3
    wait_seconds = 5.0
    last_reason = "transient rollout failure"

    for attempt in range(1, max_attempts + 1):
        try:
            result = run_single_rollout_func(sample, args, sampling_params, rollout_idx)
        except Exception as exc:
            if not text_has_transient_failure(str(exc)):
                raise
            result = None
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = "empty result"
            if isinstance(result, dict):
                reason = result.get("failure_type") or "transient rollout failure"
        last_reason = reason

        if result and not result_has_transient_failure(result):
            result["_rollout_idx"] = rollout_idx
            return result

        if attempt < max_attempts:
            clean_rollout_artifacts(_rollout_output_dir(args.output_dir, question_id, rollout_idx, args.rollouts_per_sample))
            print(
                f"  [Pipeline] Sample {question_id} rollout {rollout_idx}: "
                f"{reason}; rerunning ({attempt + 1}/{max_attempts})..."
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    clean_rollout_artifacts(_rollout_output_dir(args.output_dir, question_id, rollout_idx, args.rollouts_per_sample))
    raise RuntimeError(
        f"Sample {question_id} rollout {rollout_idx} still has transient failure "
        f"after {max_attempts} attempt(s): {last_reason}"
    )


def _load_completed_rollout_result(sample_dir, rollout_dir, rollout_idx=None):
    traj_file = os.path.join(rollout_dir, TRAJECTORY_FILENAME)
    metrics_file = os.path.join(rollout_dir, ROLLOUT_METRICS_FILENAME)
    if not (os.path.exists(traj_file) and os.path.exists(metrics_file)):
        return None

    max_turn_idx = -1
    has_turn = False
    try:
        with open(traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    traj_data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.debug(f"Error parsing trajectory line: {e}")
                    continue
                turn_idx = traj_data.get('turn_idx')
                if turn_idx is not None:
                    has_turn = True
                    if turn_idx > max_turn_idx:
                        max_turn_idx = turn_idx
    except Exception as e:
        logger.debug(f"Error reading trajectory file: {e}")
        return None

    if not has_turn:
        return None

    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    except Exception as e:
        logger.debug(f"Error loading existing metrics: {e}")
        return None

    if rollout_dir_has_transient_failure(rollout_dir, metrics):
        logger.info(f"Existing rollout has transient failure and will be rerun: {rollout_dir}")
        clean_rollout_artifacts(rollout_dir)
        return None

    num_turns = max_turn_idx + 1 if max_turn_idx >= 0 else 0
    result = {
        'question_id': os.path.basename(sample_dir),
        'accuracy_score': metrics.get('accuracy_score', 0.0),
        'conversation_history': [{"role": "assistant", "content": ""} for _ in range(num_turns)] if num_turns > 0 else [],
    }
    if rollout_idx is not None:
        result['_rollout_idx'] = rollout_idx
    return result


def _load_reusable_rollout_results(sample_dir, args):
    """Return clean completed rollout results plus rollout indices that need rerun."""
    rollouts_per_sample = int(getattr(args, 'rollouts_per_sample', 1) or 1)
    if rollouts_per_sample <= 1:
        result = _load_completed_rollout_result(sample_dir, sample_dir, 0)
        return ([result] if result else []), ([] if result else [0])

    reusable_results = []
    missing_indices = []
    for rollout_idx in range(rollouts_per_sample):
        rollout_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}")
        result = _load_completed_rollout_result(sample_dir, rollout_dir, rollout_idx)
        if result is None:
            missing_indices.append(rollout_idx)
        else:
            reusable_results.append(result)
    return reusable_results, missing_indices


def _sample_summary_is_reusable(sample_dir):
    summary_path = os.path.join(sample_dir, SAMPLE_SUMMARY_FILENAME)
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        logger.info(f"Existing sample summary is unreadable and will be regenerated: {summary_path} ({exc})")
        return False
    if isinstance(summary, dict) and summary.get("has_transient_failure"):
        logger.info(f"Existing sample summary has transient failure and will be regenerated: {summary_path}")
        os.remove(summary_path)
        return False
    return True


def _check_sample_completed(sample_dir, args):
    """Check whether a sample is complete and recover minimal result records."""
    if not getattr(args, 'skip_completed', False):
        return None

    if not _sample_summary_is_reusable(sample_dir):
        return None

    sample_rollout_results, missing_indices = _load_reusable_rollout_results(sample_dir, args)
    if missing_indices:
        return None
    return {'sample_rollout_results': sample_rollout_results}


def _pipeline_rollout_state(question_id, sample_dir, args):
    completed_rollouts = []
    pending_rollout_indices = list(range(args.rollouts_per_sample))
    skip_sample = False

    if not getattr(args, 'skip_completed', False):
        return completed_rollouts, pending_rollout_indices, skip_sample

    skip_sample = True
    try:
        completed_rollouts, pending_rollout_indices = _load_reusable_rollout_results(sample_dir, args)
        if pending_rollout_indices:
            skip_sample = False
            print(
                f"  [Pipeline] Sample {question_id}: reusing "
                f"{len(completed_rollouts)} valid rollout(s), "
                f"rerunning rollout(s) {pending_rollout_indices}..."
            )
        else:
            compute_and_save_sample_summary(
                question_id, completed_rollouts, args.rollouts_per_sample, sample_dir
            )
    except Exception:
        skip_sample = False
        print(f"  [Pipeline] Sample {question_id}: existing rollout files are unreadable, rerunning...")

    if skip_sample:
        print(f"  [Pipeline] Sample {question_id}: already has {SAMPLE_SUMMARY_FILENAME}, skipping...")

    return completed_rollouts, pending_rollout_indices, skip_sample


def _pending_pipeline_samples(samples, sample_indices, sample_results_dict, args, all_results_list):
    pending_samples = []
    for local_idx, sample in enumerate(samples):
        actual_sample_idx = sample_indices[local_idx]
        question_id = sample_results_dict[actual_sample_idx]['question_id']
        sample_dir = sample_results_dict[actual_sample_idx]['sample_dir']
        completed_rollouts, pending_rollout_indices, skip_sample = _pipeline_rollout_state(
            question_id, sample_dir, args
        )
        if skip_sample:
            continue

        if completed_rollouts:
            sample_results_dict[actual_sample_idx]['results'].extend(completed_rollouts)
            if all_results_list is not None:
                all_results_list.extend(completed_rollouts)
        pending_samples.append({
            'sample_idx': actual_sample_idx,
            'sample': sample,
            'sample_args': prepare_sample_args(args),
            'question_id': question_id,
            'sample_dir': sample_dir,
            'pending_rollout_indices': pending_rollout_indices,
        })
    return pending_samples


def _finalize_pipeline_sample_if_complete(sample_idx, rollout_counts, completed_samples, sample_results_dict, args):
    if sample_idx in completed_samples or rollout_counts[sample_idx] < args.rollouts_per_sample:
        return

    completed_samples.add(sample_idx)
    sample_info = sample_results_dict[sample_idx]
    if not sample_info['results']:
        return

    compute_and_save_sample_summary(
        sample_info['question_id'],
        sample_info['results'],
        args.rollouts_per_sample,
        sample_info['sample_dir'],
    )
    if not getattr(args, 'disable_memory_usage_update', False):
        update_memory_usage_after_sample(
            getattr(args, 'experience_library', None),
            getattr(args, 'skill_library', None),
            sample_info['sample_dir'],
            sample_info['sample'],
            sample_info['results'],
        )


def _schedule_item_rollouts(
    executor,
    state: _PipelineState,
    item,
    sampling_params,
    run_single_rollout_func,
    base_system_prompt,
):
    sample_args = item['sample_args']
    sample_args._sample_system_prompt = getattr(sample_args, "_sample_system_prompt", base_system_prompt)
    for rollout_idx in item.get('pending_rollout_indices') or []:
        future = executor.submit(
            _run_rollout_until_valid,
            item['sample'],
            sample_args,
            sampling_params,
            rollout_idx,
            run_single_rollout_func,
        )
        state.active_futures[future] = {
            'type': 'rollout',
            'sample_idx': item['sample_idx'],
            'rollout_idx': rollout_idx,
        }


def _schedule_next_retrieval(
    executor,
    state: _PipelineState,
    pending_samples,
    experience_retriever,
    base_system_prompt,
):
    if state.next_sample_idx >= len(pending_samples):
        return
    item = pending_samples[state.next_sample_idx]
    future = executor.submit(
        retrieve_experiences_for_sample,
        item['sample'],
        item['sample_args'],
        experience_retriever,
        base_system_prompt,
        item['question_id'],
        item['sample_dir'],
        update_global_prompts=False,
    )
    state.active_futures[future] = {
        'type': 'retrieve',
        'sample_idx': item['sample_idx'],
        'sample': item['sample'],
        'sample_args': item['sample_args'],
        'question_id': item['question_id'],
        'sample_dir': item['sample_dir'],
        'pending_rollout_indices': item.get('pending_rollout_indices') or [],
    }
    state.active_retrieval_count += 1
    state.next_sample_idx += 1


def _schedule_initial_pipeline_work(
    executor,
    state: _PipelineState,
    pending_samples,
    *,
    args,
    experience_retriever,
    base_system_prompt,
    sampling_params,
    run_single_rollout_func,
):
    skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
    if experience_retriever is None and not skill_enabled:
        for item in pending_samples:
            _schedule_item_rollouts(
                executor,
                state,
                item,
                sampling_params,
                run_single_rollout_func,
                base_system_prompt,
            )
        state.next_sample_idx = len(pending_samples)
        return

    max_concurrent_retrievals = args.num_workers
    while state.next_sample_idx < len(pending_samples) and state.active_retrieval_count < max_concurrent_retrievals:
        _schedule_next_retrieval(executor, state, pending_samples, experience_retriever, base_system_prompt)


def _handle_retrieval_future(
    future,
    info,
    executor,
    state: _PipelineState,
    pending_samples,
    *,
    experience_retriever,
    base_system_prompt,
    sampling_params,
    run_single_rollout_func,
):
    state.active_retrieval_count -= 1
    sample_args = info['sample_args']
    question_id = info['question_id']
    pending_rollout_indices = info.get('pending_rollout_indices') or []

    try:
        future.result()
        print(
            f"  [Pipeline] Sample {question_id}: Retrieve completed, "
            f"submitting {len(pending_rollout_indices)} rollout(s)..."
        )
    except Exception as e:
        print(f"  [Pipeline] Error retrieving for sample {question_id}: {e}. Falling back to base prompt.")
        sample_args._sample_system_prompt = base_system_prompt

    _schedule_item_rollouts(
        executor,
        state,
        info,
        sampling_params,
        run_single_rollout_func,
        base_system_prompt,
    )
    _schedule_next_retrieval(executor, state, pending_samples, experience_retriever, base_system_prompt)


def _handle_rollout_future(
    future,
    info,
    state: _PipelineState,
    *,
    sample_results_dict,
    all_results_list,
    args,
):
    actual_sample_idx = info['sample_idx']
    rollout_idx = info['rollout_idx']
    try:
        result = future.result()
        if result:
            result['_rollout_idx'] = rollout_idx
            sample_results_dict[actual_sample_idx]['results'].append(result)
            if all_results_list is not None:
                all_results_list.append(result)
    except Exception as e:
        raise RuntimeError(
            f"Rollout {rollout_idx} for sample {actual_sample_idx} failed: {e}"
        ) from e

    state.rollout_counts.setdefault(actual_sample_idx, 0)
    state.rollout_counts[actual_sample_idx] += 1
    _finalize_pipeline_sample_if_complete(
        actual_sample_idx,
        state.rollout_counts,
        state.completed_samples,
        sample_results_dict,
        args,
    )


def _prepare_pipeline_run(samples, sample_indices, sample_results_dict, args, all_results_list, progress_desc):
    pending_samples = _pending_pipeline_samples(
        samples, sample_indices, sample_results_dict, args, all_results_list
    )
    scheduled_rollouts = sum(len(item.get('pending_rollout_indices') or []) for item in pending_samples)
    skipped_rollouts = len(samples) * args.rollouts_per_sample - scheduled_rollouts
    if skipped_rollouts:
        print(f"  [Pipeline] Skipped {skipped_rollouts} completed rollout(s).")
    if scheduled_rollouts == 0:
        print("  [Pipeline] All samples in this batch are already completed.")
        return pending_samples, None, None

    state = _PipelineState()
    state.rollout_counts = {
        item['sample_idx']: len(sample_results_dict[item['sample_idx']]['results'])
        for item in pending_samples
    }
    pbar = tqdm(total=scheduled_rollouts, desc=progress_desc or "Processing rollouts")
    return pending_samples, state, pbar


def _run_pipeline_event_loop(
    executor,
    state: _PipelineState,
    pending_retrieval_samples,
    *,
    args,
    experience_retriever,
    base_system_prompt,
    sampling_params,
    run_single_rollout_func,
    sample_results_dict,
    all_results_list,
    pbar,
) -> None:
    while state.active_futures:
        done_futures, _ = wait(state.active_futures.keys(), return_when=FIRST_COMPLETED)
        for future in done_futures:
            if future not in state.active_futures:
                continue
            info = state.active_futures.pop(future)
            if info['type'] == 'retrieve':
                _handle_retrieval_future(
                    future,
                    info,
                    executor,
                    state,
                    pending_retrieval_samples,
                    experience_retriever=experience_retriever,
                    base_system_prompt=base_system_prompt,
                    sampling_params=sampling_params,
                    run_single_rollout_func=run_single_rollout_func,
                )
            elif info['type'] == 'rollout':
                _handle_rollout_future(
                    future,
                    info,
                    state,
                    sample_results_dict=sample_results_dict,
                    all_results_list=all_results_list,
                    args=args,
                )
                pbar.update(1)


def _execute_pipeline_parallel_processing(
    samples: List[Dict],
    sample_indices: List[int],
    sample_results_dict: Dict,
    args: Any,
    experience_retriever: Optional[Any],
    base_system_prompt: str,
    sampling_params: Dict,
    run_single_rollout_func: Callable,
    progress_desc: Optional[str] = None,
    all_results_list: Optional[List] = None
) -> None:
    """Pipeline-parallel retrieval and rollout scheduling for multi-rollout mode."""
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        pending_retrieval_samples, state, pbar = _prepare_pipeline_run(
            samples, sample_indices, sample_results_dict, args, all_results_list, progress_desc
        )
        if state is None:
            return

        _schedule_initial_pipeline_work(
            executor,
            state,
            pending_retrieval_samples,
            args=args,
            experience_retriever=experience_retriever,
            base_system_prompt=base_system_prompt,
            sampling_params=sampling_params,
            run_single_rollout_func=run_single_rollout_func,
        )

        try:
            _run_pipeline_event_loop(
                executor,
                state,
                pending_retrieval_samples,
                args=args,
                experience_retriever=experience_retriever,
                base_system_prompt=base_system_prompt,
                sampling_params=sampling_params,
                run_single_rollout_func=run_single_rollout_func,
                sample_results_dict=sample_results_dict,
                all_results_list=all_results_list,
                pbar=pbar,
            )
        finally:
            pbar.close()

class MemoryRuntime:
    """Owns the mutable memory prompt/retriever state used during an experiment."""

    def __init__(self, args, base_system_prompt):
        self.args = args
        self.base_system_prompt = base_system_prompt
        self.current_system_prompt = base_system_prompt
        self.experience_retriever = None

    def initialize(self):
        self.experience_retriever, experience_injection_text = initialize_experience_retriever(self.args)
        if experience_injection_text:
            self.current_system_prompt = (
                experience_injection_text + "\n\n" + self.base_system_prompt
                if self.base_system_prompt else experience_injection_text
            )
            set_processors_prompts(self.current_system_prompt)
        return self

    def reload_after_training_update(self):
        self.current_system_prompt = reload_experiences(
            self.args, self.base_system_prompt, self.experience_retriever
        )
        return self.current_system_prompt

class RolloutRunner:
    """Runs samples; training mode periodically delegates memory updates."""

    def __init__(self, config, memory_runtime, training_updater):
        self.config = config
        self.args = config.args
        self.data = config.data
        self.sampling_params = config.sampling_params
        self.memory_runtime = memory_runtime
        self.training_updater = training_updater
        self.all_results = []

    def run(self):
        if self.args.num_workers > 1 and self.args.rollouts_per_sample > 1:
            self._run_parallel_multi_rollout()
        elif self.args.num_workers > 1:
            self._run_parallel_single_rollout()
        else:
            self._run_sequential()
        self.training_updater.flush_final()
        return self.all_results

    def _process_single_sample(self, sample, sample_idx):
        sample_args = prepare_sample_args(self.args)
        question_id, sample_dir = get_sample_metadata(sample, sample_idx, sample_args.output_dir)
        os.makedirs(sample_dir, exist_ok=True)

        retrieve_experiences_for_sample(
            sample,
            sample_args,
            self.memory_runtime.experience_retriever,
            self.memory_runtime.current_system_prompt,
            question_id,
            sample_dir,
            update_global_prompts=False,
        )

        completed_info = _check_sample_completed(sample_dir, sample_args)
        if completed_info:
            print(f"[Sample {sample_idx + 1}/{len(self.data)}] {question_id} - Already completed, skipping...")
            return _sample_rollout_info(
                sample, sample_idx, question_id, sample_dir, completed_info.get('sample_rollout_results', [])
            )

        sample_rollout_results = []
        pending_rollout_indices = list(range(sample_args.rollouts_per_sample))
        if getattr(sample_args, 'skip_completed', False):
            sample_rollout_results, pending_rollout_indices = _load_reusable_rollout_results(sample_dir, sample_args)
            if sample_rollout_results:
                print(
                    f"[Sample {sample_idx + 1}/{len(self.data)}] {question_id} - "
                    f"Reusing {len(sample_rollout_results)} valid rollout(s), "
                    f"running {len(pending_rollout_indices)} rollout(s)..."
                )

        if sample_args.rollouts_per_sample > 1:
            print(f"\n[Sample {sample_idx + 1}/{len(self.data)}] {question_id} - Running {len(pending_rollout_indices)} rollout(s)...")
            max_workers = min(sample_args.num_workers, sample_args.rollouts_per_sample)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rollout_idx = {
                    executor.submit(
                        _run_rollout_until_valid,
                        sample,
                        sample_args,
                        self.sampling_params,
                        rollout_idx,
                    ): rollout_idx
                    for rollout_idx in pending_rollout_indices
                }
                for future in as_completed(future_to_rollout_idx):
                    rollout_idx = future_to_rollout_idx[future]
                    result = future.result()
                    if result:
                        result['_rollout_idx'] = rollout_idx
                        sample_rollout_results.append(result)
        else:
            if pending_rollout_indices:
                result = _run_rollout_until_valid(sample, sample_args, self.sampling_params, rollout_idx=0)
                if result:
                    result['_rollout_idx'] = 0
                    sample_rollout_results.append(result)

        self._finalize_sample_rollouts(sample, sample_args, question_id, sample_dir, sample_rollout_results)

        return _sample_rollout_info(sample, sample_idx, question_id, sample_dir, sample_rollout_results)

    def _finalize_sample_rollouts(self, sample, sample_args, question_id, sample_dir, sample_rollout_results):
        compute_and_save_sample_summary(
            question_id,
            sample_rollout_results,
            sample_args.rollouts_per_sample,
            sample_dir,
        )
        if sample_rollout_results:
            update_memory_usage_after_sample(
                getattr(sample_args, 'experience_library', None),
                getattr(sample_args, 'skill_library', None),
                sample_dir,
                sample,
                sample_rollout_results,
            )

    def _pipeline_batch(self, batch_samples, batch_indices, progress_desc):
        batch_sample_results = {}
        for local_idx, sample in enumerate(batch_samples):
            actual_sample_idx = batch_indices[local_idx]
            question_id, sample_dir = get_sample_metadata(sample, actual_sample_idx, self.args.output_dir)
            batch_sample_results[actual_sample_idx] = {
                'results': [],
                'sample': sample,
                'question_id': question_id,
                'sample_dir': sample_dir
            }

        _execute_pipeline_parallel_processing(
            samples=batch_samples,
            sample_indices=batch_indices,
            sample_results_dict=batch_sample_results,
            args=self.args,
            experience_retriever=self.memory_runtime.experience_retriever,
            base_system_prompt=self.memory_runtime.current_system_prompt,
            sampling_params=self.sampling_params,
            run_single_rollout_func=_run_single_rollout,
            progress_desc=progress_desc,
            all_results_list=self.all_results,
        )

        batch_sample_infos = []
        for actual_sample_idx in batch_indices:
            if actual_sample_idx in batch_sample_results:
                sample_info = batch_sample_results[actual_sample_idx]
                if not sample_info.get('results'):
                    continue
                batch_sample_infos.append(
                    _sample_rollout_info(
                        sample_info['sample'],
                        actual_sample_idx,
                        sample_info['question_id'],
                        sample_info['sample_dir'],
                        sample_info['results'],
                    )
                )
        return batch_sample_infos

    def _run_parallel_multi_rollout(self):
        print(f"\n[Parallel Mode] Processing {len(self.data)} samples with {self.args.num_workers} workers...")
        print(f"  Rollouts per sample: {self.args.rollouts_per_sample}")

        if self.training_updater.enabled:
            max_concurrent_samples = max(1, self.args.num_workers // self.args.rollouts_per_sample)
            print(f"  Max concurrent samples per batch: {max_concurrent_samples} (limited by {self.args.num_workers} workers)")
            sample_idx = 0
            while sample_idx < len(self.data):
                batch_size = min(max_concurrent_samples, len(self.data) - sample_idx)
                batch_samples = self.data[sample_idx:sample_idx + batch_size]
                batch_indices = list(range(sample_idx, sample_idx + batch_size))
                print(f"\n[Batch] Processing samples {sample_idx + 1}-{sample_idx + batch_size} ({batch_size} samples)...")
                batch_infos = self._pipeline_batch(batch_samples, batch_indices, f"Batch {sample_idx // batch_size + 1}")
                self.training_updater.extend(batch_infos)
                sample_idx += batch_size
        else:
            total_rollouts = len(self.data) * self.args.rollouts_per_sample
            print(f"  Total rollouts: {total_rollouts} (no batch limit)")
            self._pipeline_batch(self.data, list(range(len(self.data))), "Processing rollouts")

    def _run_parallel_single_rollout(self):
        print(f"\n[Parallel Mode] Processing {len(self.data)} samples with {self.args.num_workers} workers...")

        if self.training_updater.enabled:
            batch_size = min(self.args.experience_samples_per_large_batch, self.args.num_workers)
            sample_idx = 0
            while sample_idx < len(self.data):
                actual_batch_size = min(batch_size, len(self.data) - sample_idx)
                batch_samples = self.data[sample_idx:sample_idx + actual_batch_size]
                batch_indices = list(range(sample_idx, sample_idx + actual_batch_size))
                print(f"\n[Batch] Processing samples {sample_idx + 1}-{sample_idx + actual_batch_size} ({actual_batch_size} samples)...")
                batch_sample_infos = []
                with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
                    futures = {
                        executor.submit(self._process_single_sample, sample, actual_idx): actual_idx
                        for actual_idx, sample in zip(batch_indices, batch_samples)
                    }
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {sample_idx // batch_size + 1}"):
                        actual_idx = futures[future]
                        try:
                            sample_info = future.result()
                            if _record_sample_info(self.all_results, sample_info):
                                batch_sample_infos.append(sample_info)
                        except Exception as e:
                            raise RuntimeError(f"Sample {actual_idx} failed after rollout retries: {e}") from e
                self.training_updater.extend(batch_sample_infos)
                sample_idx += actual_batch_size
        else:
            with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
                futures = {
                    executor.submit(self._process_single_sample, sample, sample_idx): sample_idx
                    for sample_idx, sample in enumerate(self.data)
                }
                for future in tqdm(as_completed(futures), total=len(self.data), desc="Processing samples"):
                    sample_idx = futures[future]
                    try:
                        sample_info = future.result()
                        _record_sample_info(self.all_results, sample_info)
                    except Exception as e:
                        raise RuntimeError(f"Sample {sample_idx} failed after rollout retries: {e}") from e

    def _run_sequential(self):
        for sample_idx, sample in enumerate(tqdm(self.data, desc="Processing samples")):
            sample_info = self._process_single_sample(sample, sample_idx)
            if _record_sample_info(self.all_results, sample_info):
                self.training_updater.append(sample_info)
