"""API-based inference script."""

import argparse
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 结果保存与汇总
from utils.result_utils import save_results, print_summary
import yaml

# XSkill 经验/技能学习模块
from exskill import (
    ExperienceLLM,
    summarize_rollouts,
    intra_sample_experiences,
    load_existing as exp_load_existing,
    batch_merge as exp_batch_merge,
    save_library as exp_save_library,
    refine_experience_library,
    cluster_experiences_to_skills,
    compact_experience_operation,
    normalize_experience_operation,
    update_memory_usage_after_sample,
)
from exskill.experience_utils import EXPERIENCE_LIBRARY_LOCK


def _operation_kind(operation):
    """Return the explicit kind on an LLM-produced experience operation, if any."""
    if not isinstance(operation, dict):
        return None
    payload = operation.get("experience")
    if isinstance(payload, dict):
        value = payload.get("kind") or payload.get("outcome")
    else:
        value = operation.get("kind") or operation.get("outcome")
    return str(value).strip().lower() if value is not None else None


_UNSUPPORTED_MEMORY_TERMS = (
    "ocr",
    "handwriting-specific",
    "object detection",
    "detector",
    "segmentation",
    "bfs",
    "a*",
    "astar",
    "dijkstra",
)


def _text_has_unsupported_memory_term(text, support_text):
    text_lower = str(text or "").lower()
    support_lower = str(support_text or "").lower()
    return any(term in text_lower and term not in support_lower for term in _UNSUPPORTED_MEMORY_TERMS)


def _sanitize_evidence_bounded_operation(operation, support_text):
    """Remove unsupported named methods from LLM-produced memory operations."""
    if not isinstance(operation, dict):
        return operation
    op = dict(operation)
    payload = op.get("experience")
    target = dict(payload) if isinstance(payload, dict) else op

    content = target.get("content") or target.get("experience") or target.get("text")
    if _text_has_unsupported_memory_term(content, support_text):
        target["content"] = (
            "When a visual extraction attempt fails, use only available tools and "
            "validated intermediate evidence; do not assume unavailable specialized tools or methods."
        )

    if isinstance(payload, dict):
        op["experience"] = target
    else:
        op.update(target)
    return op



# 单样本 agent 执行入口
from engine.api_processors import process_single_sample, set_global_prompts as set_processors_prompts

# 推理流程中的检索、并行和指标工具
from infer_api_utils import (
    initialize_experience_retriever,
    get_used_original_experiences,
    retrieve_experiences_for_sample,
    compute_and_save_sample_summary,
    check_sample_completed,
    reload_experiences,
    prepare_sample_args,
    get_sample_metadata,
    compute_dataset_summary,
    execute_pipeline_parallel_processing,
)


# 单次 agent 执行单元。一个样本可以跑多个 rollout，用于 pass@k 和后续经验总结
def run_single_rollout(sample, args, sampling_params, rollout_idx):
    """
    Run a single rollout for a given sample.
    对给定的样本进行一次单独的测试运行

    Args:
        sample: Data sample to process
        args: Command line arguments
        sampling_params: Sampling parameters dictionary
        rollout_idx: Rollout index (0-based)

    Returns:
        Result dictionary with trajectory information
        返回包含轨迹信息的结果字典
    """
    # 论文 Phase I/II 的基础执行单元：一次独立 rollout。
    import random

    # 每个 rollout 使用独立 args 副本，避免并行时互相覆盖 output_dir 等状态。
    rollout_args = prepare_sample_args(args)

    # 每个 rollout 使用不同随机种子，便于 pass@k 统计。
    seed = args.seed_base + rollout_idx
    random.seed(seed)

    # 如果后端支持 seed，就写入采样参数。
    rollout_sampling_params = sampling_params.copy()
    if 'seed' in rollout_sampling_params:
        rollout_sampling_params['seed'] = seed

    result = process_single_sample(sample, rollout_args, rollout_sampling_params, rollout_idx=rollout_idx)
    return result


# 对一个样本的轨迹做总结，然后做 cross-rollout critique，生成可复用经验，并可选生成 skill 片段
def generate_experience_for_sample(sample_info, args):
    """
    Generate experience for a single sample (extracted from process_single_sample_with_experience).
    This function can be called in parallel for multiple samples.
    为单个样本生成经验（该经验是从“process_single_sample_with_experience”函数中提取出来的）。
    此函数可并行调用以处理多个样本。

    Args:
        sample_info: Dict containing:
            - 'sample': Data sample
            - 'sample_idx': Sample index
            - 'question_id': Question ID
            - 'sample_dir': Sample directory path
            - 'sample_rollout_results': List of rollout results
        args: Command line arguments

    Returns:
        Dict with:
            - 'sample_id': Question ID
            - 'success': bool
            - 'experience_ops': List of experience operations (normalized)
            - 'error': str or None
    """
    # 对应 Phase I: Rollout Summary + Cross-Rollout Critique。
    question_id = sample_info['question_id']
    sample_dir = sample_info['sample_dir']
    sample_rollout_results = sample_info['sample_rollout_results']

    result = {
        'sample_id': question_id,
        'success': False,
        'experience_ops': [],
        'skill_content': '',
        'error': None
    }

    try:
        llm = ExperienceLLM()

        traj_paths = []
        for rollout_idx in range(len(sample_rollout_results)):
            if args.rollouts_per_sample > 1:
                rollout_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}")
            else:
                rollout_dir = sample_dir

            traj_path = os.path.join(rollout_dir, 'traj.jsonl')
            if os.path.exists(traj_path):
                traj_paths.append(traj_path)

        # 汇总所有 rollout 轨迹，得到每次尝试的关键步骤和失败/成功原因。
        if not traj_paths:
            result['error'] = "No trajectory files found"
            print(f"  Warning: No trajectory files found for {question_id}")
            return result

        merged_summaries = summarize_rollouts(traj_paths, llm, sample_dir=sample_dir)

        # 如果轨迹摘要失败，区分“没有执行轮次”和“摘要生成失败”两类情况。
        if not merged_summaries:
            has_any_turns = False
            for traj_path in traj_paths:
                try:
                    with open(traj_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                rec = json.loads(line)
                                # Check if this record has turn_idx (indicates a turn, not just metadata)
                                if 'turn_idx' in rec and rec.get('turn_idx') is not None:
                                    has_any_turns = True
                                    break
                            except (json.JSONDecodeError, Exception):
                                continue
                    if has_any_turns:
                        break
                except Exception:
                    continue

            if not has_any_turns:
                result['error'] = "No turns in trajectories (empty trajectories)"
                print(f"  Warning: Skipping experience generation for {question_id} - trajectories contain no turns (only initial_prompt)")
            else:
                result['error'] = "Failed to generate summary"
                print(f"  Warning: Failed to generate summary for {question_id}")
            return result

        question = merged_summaries.get('question', '') or sample_rollout_results[0].get('prompt', '')
        ground_truth = merged_summaries.get('ground_truth', '') or sample_rollout_results[0].get('ground_truth', '')
        system_prompt_text = merged_summaries.get('system_prompt', '')

        # 基于同一样本的多个 rollout 做批判，总结可迁移的经验。
        # 只传实际 rollout 摘要，不把 question/ground_truth/system_prompt 当作 rollout。
        summaries_only = {k: v for k, v in merged_summaries.items()
                         if k not in ['question', 'ground_truth', 'system_prompt']}

        # 读取本样本推理时用过的原始经验，便于生成“修改/强化旧经验”的建议。
        used_experiences = get_used_original_experiences(sample_dir)

        raw_ops = intra_sample_experiences(
            question, ground_truth, summaries_only, llm,
            max_ops=getattr(args, 'experience_max_ops', 2),
            debug_dir=sample_dir,
            system_prompt=system_prompt_text,
            used_experiences=used_experiences
        )

        # 将 LLM 输出统一整理成结构化 experience ops，后续检索仍只使用 content 字段。
        score_list = [
            r.get("accuracy_score")
            for r in sample_rollout_results
            if isinstance(r, dict) and r.get("accuracy_score") is not None
        ]
        evidence_rollouts = []
        for fallback_idx, rollout_result in enumerate(sample_rollout_results):
            if not isinstance(rollout_result, dict):
                continue
            rollout_idx = rollout_result.get("_rollout_idx", fallback_idx)
            rollout_dir = os.path.join(sample_dir, f"rollout_{rollout_idx}") if args.rollouts_per_sample > 1 else sample_dir
            traj_path = os.path.join(rollout_dir, "traj.jsonl")
            score = rollout_result.get("accuracy_score")
            evidence_rollouts.append({
                "doc_id": question_id,
                "rollout_id": f"{question_id}:rollout_{rollout_idx}",
                "rollout_idx": rollout_idx,
                "traj_path": traj_path,
                "accuracy_score": score,
                "success": bool(score == 1.0),
                "final_answer": rollout_result.get("final_answer"),
            })
        success_count = sum(1 for score in score_list if float(score) == 1.0)
        total_rollouts = max(1, len(score_list))
        has_success = success_count > 0
        reliable_success = success_count > (total_rollouts / 2)
        default_kind = "success" if reliable_success else "failure"
        all_rollouts_failed = not has_success
        source = {
            "doc_id": question_id,
            "task": sample_info.get("sample", {}).get("task"),
            "accuracy_scores": score_list,
            "evidence_rollouts": evidence_rollouts,
        }

        norm_ops = []
        if isinstance(raw_ops, str):
            try:
                raw_ops = json.loads(raw_ops)
            except Exception:
                pass

        if isinstance(raw_ops, dict) and 'experiences' in raw_ops:
            raw_iter = raw_ops['experiences'].values()
        elif isinstance(raw_ops, list):
            raw_iter = raw_ops
        elif isinstance(raw_ops, dict):
            raw_iter = [raw_ops]
        else:
            raw_iter = []
        support_text = json.dumps(
            {
                "question": question,
                "summaries": summaries_only,
                "used_experiences": used_experiences,
            },
            ensure_ascii=False,
        )

        for op in raw_iter:
            if isinstance(op, str):
                op = {"experience": op}
            elif isinstance(op, dict):
                op = _sanitize_evidence_bounded_operation(op, support_text)
                if not op.get("task_tags") and sample_info.get("sample", {}).get("task"):
                    op["task_tags"] = [sample_info["sample"]["task"]]
                if _operation_kind(op) == "success" and not has_success:
                    print(
                        f"  [Experience] Dropping success memory for {question_id} "
                        f"because no rollout was successful"
                    )
                    continue
                if all_rollouts_failed:
                    op["kind"] = "failure"
                    if op.get("confidence") is not None:
                        try:
                            op["confidence"] = min(float(op["confidence"]), 0.65)
                        except (TypeError, ValueError):
                            op["confidence"] = 0.6
            normalized = normalize_experience_operation(
                op,
                default_kind=default_kind,
            )
            if normalized:
                norm_ops.append(normalized)

        # 保存样本级经验，便于调试和复盘。
        if norm_ops:
            with open(os.path.join(sample_dir, 'exp_items.json'), 'w', encoding='utf-8') as f:
                json.dump([compact_experience_operation(op) for op in norm_ops], f, ensure_ascii=False, indent=2)

        # 多 skill 版本不再按单样本直接生成 skill。
        # skill 会在 batch 级别由 experience 聚类后统一生成，并回写 skill_ids。

        result['success'] = True
        result['experience_ops'] = norm_ops

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


# 每次大批次更新前保存当前 experiences.json 和 SKILL.md 快照，方便追踪持续学习过程。
def save_knowledge_snapshot(args, batch_idx):
    """
    Save a snapshot of current experience library and skill document before batch processing.
    在进行批量处理之前，先保存当前经验库和技能文档的当前状态的快照。

    Args:
        args: Command line arguments
        batch_idx: Current batch index (0-based)
    """
    # 记录每个大批次前的知识库状态，便于追踪持续学习过程。
    import shutil
    from datetime import datetime

    snapshot_dir = os.path.join(args.output_dir, "snapshots", f"batch_{batch_idx:03d}")
    os.makedirs(snapshot_dir, exist_ok=True)

    metadata = {
        "batch_idx": batch_idx,
        "timestamp": datetime.now().isoformat(),
    }

    # 复制当前经验库快照；metadata 按快照文件计数，避免源路径异常时计数不一致。
    snapshot_exp_path = os.path.join(snapshot_dir, "experiences.json")
    if args.experience_library and os.path.exists(args.experience_library):
        shutil.copy2(args.experience_library, snapshot_exp_path)
    # 从快照文件统计经验条数。
    if os.path.exists(snapshot_exp_path):
        try:
            with open(snapshot_exp_path, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)
            if isinstance(exp_data, list):
                metadata["experience_count"] = len(exp_data)
            elif isinstance(exp_data, dict):
                inner = exp_data.get("experiences", {})
                metadata["experience_count"] = len(inner) if isinstance(inner, (list, dict)) else 0
            else:
                metadata["experience_count"] = 0
        except Exception:
            metadata["experience_count"] = -1
    else:
        metadata["experience_count"] = 0

    # 复制当前技能文档快照。
    if getattr(args, 'skill_library', None) and os.path.exists(args.skill_library):
        shutil.copy2(args.skill_library, os.path.join(snapshot_dir, "SKILL.md"))
        try:
            with open(args.skill_library, 'r', encoding='utf-8') as f:
                skill_content = f.read()
            metadata["skill_word_count"] = len(skill_content.split())
        except Exception:
            metadata["skill_word_count"] = -1
    else:
        metadata["skill_word_count"] = 0

    metadata["skill_count"] = 0
    if getattr(args, 'skill_library', None):
        skill_root = os.path.dirname(args.skill_library)
        skill_index_src = os.path.join(skill_root, "skill_index.json")
        if os.path.exists(skill_index_src):
            skill_index_dst = os.path.join(snapshot_dir, "skill_index.json")
            shutil.copy2(skill_index_src, skill_index_dst)
            try:
                with open(skill_index_dst, 'r', encoding='utf-8') as f:
                    skill_index = json.load(f)
                metadata["skill_count"] = len(skill_index.get("skills", {}))
            except Exception:
                metadata["skill_count"] = -1

        skills_src = os.path.join(skill_root, "skills")
        if os.path.isdir(skills_src):
            shutil.copytree(skills_src, os.path.join(snapshot_dir, "skills"), dirs_exist_ok=True)

    # 保存快照元信息。
    with open(os.path.join(snapshot_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  [Snapshot] Saved batch_{batch_idx:03d} snapshot (exp: {metadata['experience_count']}, skills: {metadata['skill_count']})")


# Phase I 的知识合并部分：批量生成经验、合并经验库、合并 skill 文档，并按需 refine。
def process_large_batch_experiences(samples_info, args, batch_idx=0, is_final=False):
    """
    Process experience generation for a large batch of samples in parallel,
    then merge all experiences into the library.
    对大量样本进行并行处理以生成过程经验，然后将所有经验合并到知识库中。

    Args:
        samples_info: List of sample_info dicts (from generate_experience_for_sample)
        args: Command line arguments
        batch_idx: Current batch index for snapshot naming
        is_final: If True, force refine regardless of threshold (for final batch)

    Returns:
        Number of successfully processed samples
    """
    # 对应 Phase I: Knowledge Consolidation，批量合并经验与技能。
    if not samples_info:
        return 0
    start_time = time.time()
    print(f"\n[Large Batch {batch_idx}] Processing experience generation for {len(samples_info)} samples...")

    # 批处理前先保存知识库快照，便于比较更新前后变化。
    save_knowledge_snapshot(args, batch_idx)

    # 对大批次内的样本并行生成经验；skill 后续由经验聚类统一生成。
    all_experience_ops = []
    successful_samples = 0

    gen_start = time.time()
    # 技能生成放在单样本任务内部执行，避免并行写同一个 SKILL.md。
    num_tasks = len(samples_info)
    generation_workers = getattr(args, 'experience_generation_workers', None)
    if generation_workers is None or generation_workers <= 0:
        generation_workers = getattr(args, 'num_workers', 4) or 4
    generation_workers = max(1, min(num_tasks, int(generation_workers), 16))
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

    gen_time = time.time() - gen_start
    print(f"  [Timing] Experience generation: {gen_time:.1f}s")

    # 将本批次的新经验合并进全局经验库。
    if all_experience_ops and getattr(args, 'experience_library_update', False) and args.experience_library:
        try:
            with EXPERIENCE_LIBRARY_LOCK:
                existing = exp_load_existing(args.experience_library)
                merge_start = time.time()
                merged = exp_batch_merge(
                    existing, all_experience_ops, ExperienceLLM(),
                    debug_dir=None,
                    experience_limit=getattr(args, 'experience_max_items', 16)
                )
                merge_time = time.time() - merge_start

                # 经验库过长或最终批次时，调用 LLM 去重、压缩和重写。
                if getattr(args, 'experience_refine', False):
                    max_items = getattr(args, 'experience_max_items', 256)
                    if is_final or len(merged) > max_items:
                        refine_start = time.time()
                        refine_llm = ExperienceLLM()
                        merged = refine_experience_library(merged, refine_llm, debug_dir=args.output_dir)
                        refine_time = time.time() - refine_start
                        print(f"  [Timing] Experience refine{'(final)' if is_final else ''}: {refine_time:.1f}s")

                os.makedirs(os.path.dirname(args.experience_library), exist_ok=True)
                exp_save_library(args.experience_library, merged)
                print(f"  [Large Batch] Merged {len(all_experience_ops)} experience operations into library (final size: {len(merged)})")
                print(f"  [Timing] Experience merge: {merge_time:.1f}s")

                if getattr(args, 'skill_enable', False) and getattr(args, 'skill_library', None):
                    # 从合并后的结构化 experience 中聚类生成多个 skill，并把 skill_ids 回写到 experience。
                    skill_start = time.time()
                    skill_llm = ExperienceLLM()
                    merged = cluster_experiences_to_skills(merged, args.skill_library, skill_llm, args)
                    exp_save_library(args.experience_library, merged)
                    print(f"  [Timing] Skill clustering: {time.time() - skill_start:.1f}s")
        except Exception as e:
            print(f"  Warning: Failed to merge experiences into library: {e}")

    total_time = time.time() - start_time
    print(f"[Large Batch] Completed: {successful_samples}/{len(samples_info)} samples processed successfully (total: {total_time:.1f}s)")
    return successful_samples


def main(args):
    """
    Main function to run the API-based inference process.
    """
    # 主入口：串起 Phase I 的积累流程和 Phase II 的推理流程。
    # 从环境变量读取主推理模型配置。
    args.model_name = os.environ.get("REASONING_MODEL_NAME")
    if not args.model_name:
        raise ValueError("REASONING_MODEL_NAME environment variable must be set")

    print(f"Starting API-based greedy inference...")
    print(f"Reasoning model: {args.model_name}")

    if getattr(args, 'experience_online_generate', False):
        # ROLLOUTS_PER_SAMPLE: 每个样本跑几次agent
        small_batch = args.rollouts_per_sample
        # EXPERIENCE_LARGE_BATCH：累计多少个 rollout 后做一次经验/技能更新
        large_batch = getattr(args, 'experience_large_batch', None) or small_batch
        args.experience_large_batch = large_batch

        # 校验在线经验生成的批次设置：大批次必须是 rollout 数的整数倍。
        if large_batch % small_batch != 0:
            raise ValueError(
                f"experience_large_batch ({large_batch}) must be a multiple of "
                f"rollouts_per_sample ({small_batch})"
            )

        # 计算每次更新经验的样本数量
        args.experience_samples_per_large_batch = large_batch // small_batch

        print(f"\n{'='*80}")
        print(f"Experience Generation Batching:")
        print(f"  Small batch (rollouts per sample): {small_batch}")
        print(f"  Large batch (rollouts per batch): {large_batch}")
        print(f"  Samples per large batch: {args.experience_samples_per_large_batch}")
        print(f"{'='*80}\n")

    # 多 rollout 用于 pass@k / average@k 统计，也为经验批判提供多条轨迹。
    if args.rollouts_per_sample > 1:
        print(f"\n{'='*80}")
        print(f"Multi-rollout mode enabled: {args.rollouts_per_sample} rollouts per sample")
        print(f"Base seed: {args.seed_base}")
        print(f"This will enable pass@k and average@k evaluation")
        print(f"{'='*80}\n")

    # 从 YAML 加载基础 system prompt，用 system_prompt_key 区分
    # system-prompt-key 的区别，本质是告诉模型“你有哪些工具、应该按什么方式解题”
    try:
        with open(args.inference_prompts_path, 'r', encoding='utf-8') as f:
            prompts_yaml = yaml.safe_load(f)
        SYSTEM_PROMPT = prompts_yaml['system_prompts'][args.system_prompt_key]
        print("Inference prompts loaded successfully.")
    except Exception as e:
        print(f"Error loading inference prompts from {args.inference_prompts_path}: {e}")
        return

    # 保存不含经验/技能注入的原始 prompt，后续每个样本都从它重新构造。
    BASE_SYSTEM_PROMPT = SYSTEM_PROMPT

    # 同步到 api_processors，全局默认先使用基础 prompt。
    set_processors_prompts(SYSTEM_PROMPT)


    # 加载工具配置
    try:
        tool_config_path = getattr(args, 'tool_config_path', 'eval/configs/tool_configs.yaml')
        if os.path.exists(tool_config_path):
            with open(tool_config_path, 'r', encoding='utf-8') as f:
                args.tool_configs = yaml.safe_load(f) or {}
            print(f"Loaded tool configs from {tool_config_path}")
        else:
            args.tool_configs = {}
            print(f"Tool config file not found at {tool_config_path}, using defaults")
    except Exception as e:
        print(f"Error loading tool configs: {e}")
        args.tool_configs = {}

    # 读取输入数据，支持 JSON list 或 JSONL
    if args.input_file.endswith('.jsonl'):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    original_count = len(data)

    # 调试时可限制样本数量
    if args.max_samples is not None and args.max_samples > 0:
        data = data[:args.max_samples]
        print(f"Loaded {original_count} samples, limiting to {len(data)} samples (--max-samples={args.max_samples}).")
    else:
        print(f"Loaded {len(data)} samples.")

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_completion_tokens,
    }
    if args.max_total_tokens <= args.max_completion_tokens:
        adjusted_total = args.max_completion_tokens + 4096
        print(
            f"Warning: max_total_tokens ({args.max_total_tokens}) must be larger than "
            f"max_completion_tokens ({args.max_completion_tokens}); using {adjusted_total}."
        )
        args.max_total_tokens = adjusted_total

    # 初始化经验检索器；若未启用检索，也可能退化为全量经验注入。
    # 对应 Phase II: Decomposition/Retrieval 的初始化。
    experience_retriever, experience_injection_text = initialize_experience_retriever(args)
    current_system_prompt = BASE_SYSTEM_PROMPT
    if experience_injection_text:
        # Traditional mode: inject all experiences (or up to max_items)
        current_system_prompt = (experience_injection_text + "\n\n" + BASE_SYSTEM_PROMPT) if BASE_SYSTEM_PROMPT else experience_injection_text
        set_processors_prompts(current_system_prompt)

    all_results = []

    # 在线经验生成的大批次队列；样本先推理，积累到阈值后统一总结/合并。
    large_batch_queue = []
    large_batch_idx = 0
    use_experience_batching = getattr(args, 'experience_online_generate', False)

    def process_single_sample_with_experience(sample, sample_idx):
        """Process a single sample including experience generation."""
        # 单样本流水线：先检索/注入知识，再执行 agent rollout。
        sample_args = prepare_sample_args(args)

        question_id, sample_dir = get_sample_metadata(sample, sample_idx, sample_args.output_dir)
        os.makedirs(sample_dir, exist_ok=True)

        retrieve_experiences_for_sample(
            sample, sample_args, experience_retriever, current_system_prompt,
            question_id, sample_dir,
            update_global_prompts=False
        )

        # 如果开启断点续跑且样本已完成，则直接复用已有结果。
        completed_info = check_sample_completed(sample_dir, sample_args)
        if completed_info:
            print(f"[Sample {sample_idx + 1}/{len(data)}] {question_id} - Already completed, skipping...")
            return {
                'sample': sample,
                'sample_idx': sample_idx,
                'question_id': question_id,
                'sample_dir': sample_dir,
                'sample_rollout_results': completed_info.get('sample_rollout_results', [])
            }

        sample_rollout_results = []

        if sample_args.rollouts_per_sample > 1:
            print(f"\n[Sample {sample_idx + 1}/{len(data)}] {question_id} - Running {sample_args.rollouts_per_sample} rollouts...")
            max_workers = min(sample_args.num_workers, sample_args.rollouts_per_sample)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交此样本的所有rollout
                future_to_rollout_idx = {
                    executor.submit(run_single_rollout, sample, sample_args, sampling_params, rollout_idx): rollout_idx
                    for rollout_idx in range(sample_args.rollouts_per_sample)
                }
                for future in as_completed(future_to_rollout_idx):
                    rollout_idx = future_to_rollout_idx[future]
                    result = future.result()
                    if result:
                        result['_rollout_idx'] = rollout_idx
                        sample_rollout_results.append(result)

            # 多 rollout 模式下保存样本级 pass@k / average@k 指标。
            compute_and_save_sample_summary(
                question_id, sample_rollout_results, sample_args.rollouts_per_sample, sample_dir
            )

        else:
            result = process_single_sample(sample, sample_args, sampling_params, rollout_idx=None)
            if result:
                sample_rollout_results.append(result)

        if sample_rollout_results:
            # 推理结束后记录本题使用过的 skill/experience，并在失败时生成 failure experience。
            update_memory_usage_after_sample(
                getattr(sample_args, 'experience_library', None),
                getattr(sample_args, 'skill_library', None),
                sample_dir,
                sample,
                sample_rollout_results,
            )

        # 返回样本执行信息；经验生成在大批次层统一触发。
        return {
            'sample': sample,
            'sample_idx': sample_idx,
            'question_id': question_id,
            'sample_dir': sample_dir,
            'sample_rollout_results': sample_rollout_results
        }

    if args.num_workers > 1 and args.rollouts_per_sample > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")
        print(f"  Rollouts per sample: {args.rollouts_per_sample}")

        if use_experience_batching:
            max_concurrent_samples = max(1, args.num_workers // args.rollouts_per_sample)
            print(f"  Max concurrent samples per batch: {max_concurrent_samples} (limited by {args.num_workers} workers)")

            # 并行 pipeline 模式：按批提交样本和 rollout，兼顾吞吐与经验更新节奏。
            sample_idx = 0
            while sample_idx < len(data):
                # 每批最多处理 max_concurrent_samples 个样本。
                batch_size = min(max_concurrent_samples, len(data) - sample_idx)
                batch_samples = data[sample_idx:sample_idx + batch_size]
                batch_indices = list(range(sample_idx, sample_idx + batch_size))

                print(f"\n[Batch] Processing samples {sample_idx + 1}-{sample_idx + batch_size} ({batch_size} samples)...")

                batch_sample_results = {}
                for local_idx, sample in enumerate(batch_samples):
                    actual_sample_idx = batch_indices[local_idx]
                    question_id, sample_dir = get_sample_metadata(sample, actual_sample_idx, args.output_dir)

                    batch_sample_results[actual_sample_idx] = {
                        'results': [],
                        'sample': sample,
                        'question_id': question_id,
                        'sample_dir': sample_dir
                    }

                # 执行检索、rollout、结果收集的流水线并行。
                execute_pipeline_parallel_processing(
                    samples=batch_samples,
                    sample_indices=batch_indices,
                    sample_results_dict=batch_sample_results,
                    args=args,
                    experience_retriever=experience_retriever,
                    base_system_prompt=current_system_prompt,
                    sampling_params=sampling_params,
                    run_single_rollout_func=run_single_rollout,
                    progress_desc=f"Batch {sample_idx // batch_size + 1}",
                    all_results_list=all_results
                )

                batch_sample_infos = []
                for actual_sample_idx in batch_indices:
                    if actual_sample_idx in batch_sample_results:
                        sample_info = batch_sample_results[actual_sample_idx]
                        sample_info_dict = {
                            'sample': sample_info['sample'],
                            'sample_idx': actual_sample_idx,
                            'question_id': sample_info['question_id'],
                            'sample_dir': sample_info['sample_dir'],
                            'sample_rollout_results': sample_info['results']
                        }
                        batch_sample_infos.append(sample_info_dict)

                # 将本批结果加入大批次经验队列。
                large_batch_queue.extend(batch_sample_infos)

                # 达到阈值后触发经验/技能更新，并刷新检索器。
                if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                    print(f"\n[Large Batch] Triggering experience generation for {len(large_batch_queue)} samples...")
                    process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                    large_batch_idx += 1
                    large_batch_queue.clear()
                    current_system_prompt = reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)
                sample_idx += batch_size
        else:
            # 不做在线经验更新时，尽量一次性提交所有 rollout 提高利用率。
            total_rollouts = len(data) * args.rollouts_per_sample
            print(f"  Total rollouts: {total_rollouts} (no batch limit)")

            # 按样本收集多个 rollout 的结果。
            sample_results = {}

            # 预先建立样本结果容器。
            for sample_idx, sample in enumerate(data):
                question_id, sample_dir = get_sample_metadata(sample, sample_idx, args.output_dir)

                sample_results[sample_idx] = {
                    'results': [],
                    'sample': sample,
                    'question_id': question_id,
                    'sample_dir': sample_dir
                }

            execute_pipeline_parallel_processing(
                samples=data,
                sample_indices=list(range(len(data))),
                sample_results_dict=sample_results,
                args=args,
                experience_retriever=experience_retriever,
                base_system_prompt=current_system_prompt,
                sampling_params=sampling_params,
                run_single_rollout_func=run_single_rollout,
                progress_desc="Processing rollouts",
                all_results_list=all_results
            )

    elif args.num_workers > 1:
        print(f"\n[Parallel Mode] Processing {len(data)} samples with {args.num_workers} workers...")

        if use_experience_batching:
            # 单 rollout 并行 + 在线经验更新：按批推理，批后总结经验。
            max_concurrent_samples = args.num_workers
            # 批大小受 worker 数和经验大批次阈值共同限制。
            batch_size = min(args.experience_samples_per_large_batch, max_concurrent_samples)

            sample_idx = 0
            while sample_idx < len(data):
                # 最后一批可能不足 batch_size。
                actual_batch_size = min(batch_size, len(data) - sample_idx)
                batch_samples = data[sample_idx:sample_idx + actual_batch_size]
                batch_indices = list(range(sample_idx, sample_idx + actual_batch_size))

                print(f"\n[Batch] Processing samples {sample_idx + 1}-{sample_idx + actual_batch_size} ({actual_batch_size} samples)...")
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = {
                        executor.submit(process_single_sample_with_experience, sample, actual_idx): actual_idx
                        for local_idx, (actual_idx, sample) in enumerate(zip(batch_indices, batch_samples))
                    }

                    # 等待本批所有样本完成后再进入经验生成，保证知识更新顺序清晰。
                    batch_sample_infos = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {sample_idx // batch_size + 1}"):
                        actual_idx = futures[future]
                        try:
                            sample_info = future.result()
                            if sample_info and sample_info.get('sample_rollout_results'):
                                all_results.extend(sample_info['sample_rollout_results'])
                                batch_sample_infos.append(sample_info)
                        except Exception as e:
                            print(f"  Error processing sample {actual_idx}: {e}")

                if use_experience_batching and batch_sample_infos:
                    large_batch_queue.extend(batch_sample_infos)

                    # 达到经验大批次阈值后触发知识库更新。
                    if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                        print(f"\n[Large Batch] Triggering experience generation for {len(large_batch_queue)} samples...")
                        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                        large_batch_idx += 1
                        large_batch_queue.clear()
                        current_system_prompt = reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)
                sample_idx += actual_batch_size
            if use_experience_batching and large_batch_queue:
                print(f"\n[Final Batch] Processing remaining {len(large_batch_queue)} samples...")
                process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx, is_final=True)
                large_batch_idx += 1
                large_batch_queue.clear()
        else:
            # 纯推理模式：所有样本并行提交，不做批后经验更新。
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                # 提交所有样本。
                futures = {
                    executor.submit(process_single_sample_with_experience, sample, sample_idx): sample_idx
                    for sample_idx, sample in enumerate(data)
                }

                for future in tqdm(as_completed(futures), total=len(data), desc="Processing samples"):
                    sample_idx = futures[future]
                    try:
                        sample_info = future.result()
                        if sample_info and sample_info.get('sample_rollout_results'):
                            all_results.extend(sample_info['sample_rollout_results'])
                    except Exception as e:
                        print(f"  Error processing sample {sample_idx}: {e}")

    else:
        # 单 worker 顺序执行，最适合 smoke test 和调试。
        for sample_idx, sample in enumerate(tqdm(data, desc="Processing samples")):
            sample_info = process_single_sample_with_experience(sample, sample_idx)
            if sample_info and sample_info.get('sample_rollout_results'):
                all_results.extend(sample_info['sample_rollout_results'])
                if use_experience_batching:
                    large_batch_queue.append(sample_info)
                    if len(large_batch_queue) >= args.experience_samples_per_large_batch:
                        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx)
                        large_batch_idx += 1
                        large_batch_queue.clear()
                        # 经验库更新后刷新注入内容/检索缓存。
                        current_system_prompt = reload_experiences(args, BASE_SYSTEM_PROMPT, experience_retriever)

    # 处理未达到大批次阈值的剩余样本。
    if use_experience_batching and large_batch_queue:
        print(f"\n[Final Batch] Processing remaining {len(large_batch_queue)} samples...")
        process_large_batch_experiences(large_batch_queue, args, batch_idx=large_batch_idx, is_final=True)
        large_batch_idx += 1
        large_batch_queue.clear()

    save_results(all_results, args.output_dir)
    print_summary(all_results, args.output_dir)
    if args.rollouts_per_sample > 1:
        print(f"\n{'='*80}")
        print(f"Aggregating pass@k and average@k metrics across all samples")
        print(f"{'='*80}\n")
        compute_dataset_summary(data, args)

    print("API-based inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API-based inference script.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output results.")
    parser.add_argument("--skip-completed", action='store_true',
                       help="Skip samples that are already completed (have traj.jsonl with turn_idx and metrics.json).")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: None, process all samples).")

    # 采样参数
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling.")
    parser.add_argument("--max-completion-tokens", type=int, default=8192, help="Maximum tokens for single model response (aligned with Baseline framework).")

    # 推理控制参数
    parser.add_argument("--max-turns", type=int, default=16, help="Maximum number of turns for inference.")
    parser.add_argument("--max-images", type=int, default=16, help="Maximum number of images per sample.")
    parser.add_argument("--max-total-tokens", type=int, default=32768, help="Maximum total tokens for context.")
    parser.add_argument("--max-pixels", type=int, default=2000000, help="Maximum pixels for image processing.")
    parser.add_argument("--min-pixels", type=int, default=40000, help="Minimum pixels for image processing.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing.")
    parser.add_argument("--rollouts-per-sample", type=int, default=4, help="Number of independent rollouts per sample for pass@k evaluation.")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed for reproducibility. Each rollout uses seed_base + rollout_index.")

    # Prompt 配置
    parser.add_argument("--inference-prompts-path", type=str, default="eval/prompts/inference_prompts.yaml",
                       help="Path to the inference prompts YAML file.")
    parser.add_argument("--system-prompt-key", type=str, default="multi_tool_agent_search",
                       choices=['direct_cot', 'agent_zoom', 'multi_tool_agent', 'multi_tool_agent_search', 'multi_tool_agent_code'],
                       help="The key for the system prompt to use from the inference prompts YAML file.")

    # 工具参数
    parser.add_argument("--tool-config-path", type=str, default="eval/configs/tool_configs.yaml",
                       help="Path to the tool configuration YAML file.")
    parser.add_argument("--image-search-max-calls", type=int, default=3,
                       help="Maximum number of image_search tool calls per sample (default: 3)")
    parser.add_argument("--web-search-max-calls", type=int, default=5,
                       help="Maximum number of web_search tool calls per sample (default: 5)")

    # 经验生成/更新参数
    parser.add_argument("--experience-enable", action='store_true',
                       help="Enable experience injection from library")
    parser.add_argument("--experience-library", type=str, default=None,
                       help="Path to consolidated experiences JSON")
    parser.add_argument("--experience-online-generate", action='store_true',
                       help="Generate per-sample experiences after inference")
    parser.add_argument("--experience-library-update", action='store_true',
                       help="Merge per-sample experiences back into the library")
    parser.add_argument("--experience-max-ops", type=int, default=2,
                       help="Max operations per sample during critique")
    parser.add_argument("--experience-temperature", type=float, default=0.6,
                       help="Temperature for experience summarization/merge (default 0.6)")
    parser.add_argument("--experience-large-batch", type=int, default=None,
                        help="Large batch size for experience generation (number of rollouts to trigger batch processing). "
                             "Must be a multiple of rollouts_per_sample. If None, equals rollouts_per_sample (no batching).")
    parser.add_argument("--experience-max-items", type=int, default=120,
                        help="Max number of experiences to inject/keep in library")
    parser.add_argument("--experience-refine", action='store_true',
                       help="Enable experience library refinement after merge (uses --experience-max-items as threshold)")
    parser.add_argument("--experience-generation-workers", type=int, default=None,
                       help="Maximum workers for parallel experience generation inside each large batch")

    # 技能生成/注入参数
    parser.add_argument("--skill-enable", action='store_true',
                       help="Enable dynamic skill generation")
    parser.add_argument("--skill-library", type=str, default=None,
                       help="Path to the global skill library (SKILL.md)")
    parser.add_argument("--skill-inference", action='store_true',
                       help="Enable skill injection during inference")
    parser.add_argument("--no-skill-adaptation", action='store_false', dest='skill_adaptation',
                       default=True, help="Disable skill adaptation (use raw skill directly, default: adapt)")
    parser.add_argument("--skill-refine", action='store_true',
                       help="Enable skill refinement after merge (trim and consolidate)")
    parser.add_argument("--skill-max-length", type=int, default=1000,
                       help="Word count threshold to trigger skill refinement (default: 1000)")
    parser.add_argument("--skill-cluster-max-clusters", type=int, default=16,
                       help="Maximum number of modular skill clusters to update per experience batch")
    parser.add_argument("--skill-cluster-min-experiences", type=int, default=2,
                       help="Minimum number of experiences required to form a skill cluster")
    parser.add_argument("--skill-retrieval-top-k", type=int, default=3,
                       help="Number of modular skills retrieved per query/subtask by embedding")
    parser.add_argument("--skill-retrieval-min-similarity", type=float, default=0.0,
                       help="Minimum embedding similarity for modular skill retrieval")
    parser.add_argument("--memory-rerank-top-skills", type=int, default=2,
                       help="Maximum number of retrieved skills kept after LLM rerank")
    parser.add_argument("--memory-rerank-top-experiences", type=int, default=8,
                       help="Maximum number of direct+linked experiences kept after LLM rerank")

    # 经验检索参数
    parser.add_argument("--experience-retrieval", action='store_true',
                        help="Enable retrieval-based experience injection (default: False, uses all experiences)")
    parser.add_argument("--experience-retrieval-top-k", type=int, default=3,
                        help="Number of top experiences to retrieve per query (default: 3)")
    parser.add_argument("--experience-retrieval-min-similarity", type=float, default=0.0,
                        help="Minimum similarity threshold for retrieved experiences (0.0 to 1.0, default: 0.0)")
    parser.add_argument("--experience-embedding-api-key", type=str, default=None,
                        help="API key for embedding service (defaults to EXPERIENCE_EMBEDDING_API_KEY or OPENAI_API_KEY)")
    parser.add_argument("--experience-embedding-endpoint", type=str, default=None,
                        help="API endpoint for embedding service (defaults to EXPERIENCE_EMBEDDING_ENDPOINT or OPENAI_API_BASE)")
    parser.add_argument("--no-experience-embedding-cache", action='store_false', dest='experience_embedding_cache_enable',
                       default=True, help="Disable disk caching of experience embeddings (default: enabled)")
    parser.add_argument("--experience-retrieval-decomposition", action='store_true',
                        help="Enable task decomposition for retrieval (uses LLM to decompose task into subtasks, then retrieves for each) (default: False)")
    parser.add_argument("--experience-retrieval-rewrite", action='store_true',
                        help="Enable experience rewrite to adapt retrieved experiences to the current task (uses LLM to rewrite experiences) (default: False)")


    args = parser.parse_args()
    main(args)
