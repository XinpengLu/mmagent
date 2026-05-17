"""
Utility functions for infer_api.py to reduce code duplication and improve maintainability.
"""

import os
import re
import json
import logging
import copy
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from exskill import ExperienceLLM, ExperienceRetriever, load_experiences, format_for_prompt, rewrite_experiences_for_task, adapt_skill_for_task
from prompts.skill_prompts_test_time import SKILL_INJECTION_HEADER
from engine.api_processors import set_global_prompts as set_processors_prompts
from qwen_vl_utils import fetch_image
from utils.context_utils import process_image

logger = logging.getLogger(__name__)

# 输出文件名：检索日志、单题指标、数据集指标。
RETRIEVAL_INFO_FILENAME = "tt_decomposition_info.txt"
SAMPLE_SUMMARY_FILENAME = "metrics_sample.json"
DATASET_SUMMARY_FILENAME = "metrics_at_k.json"


def initialize_experience_retriever(args):
    """
    Initialize experience retriever if retrieval mode is enabled.
    如果检索模式已启用，则初始化经验检索器
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (experience_retriever, system_prompt)
            - experience_retriever: ExperienceRetriever instance or None
            经验检索器：经验检索器实例或 None
            - system_prompt: Updated system prompt with experiences injected (if not using retrieval)
            系统提示：已注入经验后的更新系统提示（若未使用检索功能）
    """
    # 对应 Phase II: 初始化经验库检索器和 embedding 缓存。
    experience_retriever = None
    system_prompt = None
    
    if getattr(args, 'experience_enable', False) and args.experience_library:
        try:
            # 读入经验库，后续按配置选择“向量检索”或“全量注入”。
            exps = load_experiences(args.experience_library)
            
            # 检查检索模式是否已启用
            use_retrieval = getattr(args, 'experience_retrieval', False)
            
            if use_retrieval:
                # 检索模式：为每道题按相似度取少量相关经验
                print(f"\n{'='*80}")
                print(f"Initializing Experience Retriever (retrieval mode enabled)...")
                print(f"{'='*80}\n")
                print(f"Loading experience library: {args.experience_library}")
                print(f"Total experiences in library: {len(exps)}")
                
                try:
                    # embedding 缓存默认放在经验库同目录，避免重复生成向量
                    enable_cache = getattr(args, 'experience_embedding_cache_enable', True)
                    cache_dir = os.path.dirname(args.experience_library) if args.experience_library else None
                    embedding_model = os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small")
                    embedding_api_key = getattr(args, 'experience_embedding_api_key', None) or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
                    embedding_endpoint = getattr(args, 'experience_embedding_endpoint', None) or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
                    
                    # 兼容只填 base url 的情况，统一补成 /v1。
                    if embedding_endpoint and not embedding_endpoint.endswith("/v1"):
                        embedding_endpoint = embedding_endpoint.rstrip("/") + "/v1"
                    
                    # 任务分解和经验改写都需要额外的 LLM client。
                    llm_client = None
                    use_decomposition = getattr(args, 'experience_retrieval_decomposition', False)
                    use_rewrite = getattr(args, 'experience_retrieval_rewrite', False)
                    if use_decomposition or use_rewrite:
                        features = []
                        if use_decomposition:
                            features.append("task decomposition")
                        if use_rewrite:
                            features.append("experience rewrite")
                        print(f"Initializing LLM client for {', '.join(features)}...")
                        llm_client = ExperienceLLM()
                        print(f"LLM client initialized.")
                    
                    print(f"\nInitializing Experience Retriever (this may take a moment for first-time embedding generation)...")
                    experience_retriever = ExperienceRetriever(
                        experiences=exps,
                        embedding_model=embedding_model,
                        embedding_api_key=embedding_api_key,
                        embedding_endpoint=embedding_endpoint,
                        cache_dir=cache_dir,
                        enable_cache=enable_cache,
                        llm_client=llm_client,
                        experience_library_path=args.experience_library,
                    )
                    stats = experience_retriever.get_embedding_stats()
                    print(f"Experience Retriever initialized:")
                    print(f"  - Total experiences: {stats['total_experiences']}")
                    print(f"  - Embedded count: {stats['embedded_count']}")
                    print(f"  - Embedding model: {stats['embedding_model']}")
                    print(f"  - Cache enabled: {stats['cache_enabled']}")
                    if stats['cache_path']:
                        print(f"  - Cache path: {stats['cache_path']}")
                    if getattr(args, 'experience_retrieval_decomposition', False):
                        print(f"  - Task decomposition: ENABLED")
                    if getattr(args, 'experience_retrieval_rewrite', False):
                        print(f"  - Experience rewrite: ENABLED")
                    print(f"\n{'='*80}\n")
                except Exception as e:
                    print(f"Warning: Failed to initialize Experience Retriever: {e}")
                    # 检索器失败时不中断评测，退回到全量经验注入。
                    print(f"  Falling back to full experience injection mode.")
                    experience_retriever = None
                    use_retrieval = False
            
            if not use_retrieval:
                # 全量注入模式：直接把经验库文本拼进 system prompt
                injection_text = format_for_prompt(exps, max_items=getattr(args, 'experience_max_items', 256))
                if injection_text:
                    system_prompt = injection_text  # Will be combined with base prompt later
                    print(f"Loaded {len(exps)} experiences for injection (full injection mode).")
        except Exception as e:
            print(f"Experience injection skipped: {e}")
    
    return experience_retriever, system_prompt


def retrieve_experiences_for_sample(sample, args, experience_retriever, base_system_prompt, 
                                     question_id, sample_dir, update_global_prompts=True):
    """
    Retrieve experiences and/or inject skills for a single sample.
    Supports: Experience only, Skill only, or both combined.
    获取单个样本的相关经验或注入相关技能。
    支持的选项：仅经验、仅技能，或两者结合。
    """
    # 对应 Phase II: 任务分解、经验检索、经验重写、技能适配与注入。
    # 分别判断经验检索和技能注入(skill_inference)是否启用。skill_library是skill地址
    exp_enabled = experience_retriever is not None
    skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
    
    if not (exp_enabled or skill_enabled):
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)
    
    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
    
    try:
        query_text = sample.get('problem', sample.get('question', ''))
        # 图像会参与多模态任务分解、经验改写和技能适配。
        images = _load_images_for_retrieval(sample, args)
        
        # --- 1. Experience Retrieval (optional) ---
        retrieved_exps, retrieval_info, original_retrieved_exps = {}, {}, None
        use_rewrite = getattr(args, 'experience_retrieval_rewrite', False)
        llm_client = None
        
        if exp_enabled:
            use_decomposition = getattr(args, 'experience_retrieval_decomposition', False)
            if use_decomposition:
                # 分解检索：先拆成子任务，再分别召回经验并去重。
                retrieved_exps, retrieval_info = experience_retriever.retrieve_with_decomposition(
                    task_description=query_text,
                    top_k=getattr(args, 'experience_retrieval_top_k', 5),
                    min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
                    subtask_top_k=getattr(args, 'experience_retrieval_subtask_top_k', None),
                    images=images,
                )
            else:
                # 普通检索：直接用整道题作为 query 做向量召回。
                retrieved_exps, retrieval_info = experience_retriever.retrieve(
                    query=query_text,
                    top_k=getattr(args, 'experience_retrieval_top_k', 5),
                    min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
                )
            
            if not retrieved_exps:
                print(f"  [Retrieval] No relevant experiences found for {question_id}")
            
            if retrieved_exps and use_rewrite:
                original_retrieved_exps = copy.deepcopy(retrieved_exps)
                llm_client = experience_retriever.llm_client or ExperienceLLM()
                print(f"  [Retrieval] Rewriting {len(retrieved_exps)} experiences...")
                # 经验重写：把历史经验改成更贴合当前题目的提示。
                retrieved_exps = rewrite_experiences_for_task(retrieved_exps, query_text, llm_client, images)
        
        # --- 2. Skill Processing (optional) ---
        skill_injection = ""
        if skill_enabled and os.path.exists(args.skill_library):
            # 技能处理：读取 SKILL.md，并按当前题目做可选适配。
            skill_injection = _process_skill(args, query_text, retrieved_exps, images, llm_client, sample_dir, question_id)
        
        # --- 3. Build Final Injection ---
        exp_injection = format_for_prompt(retrieved_exps, max_items=len(retrieved_exps)) if retrieved_exps else ""
        
        # Build final injection: combine skill and experience when appropriate
        use_adaptation = getattr(args, 'skill_adaptation', True)
        
        if skill_injection and exp_injection:
            # 适配后的 skill 已吸收经验信息；raw skill 模式下才同时注入经验。
            if skill_enabled and not use_adaptation:
                full_injection = f"{skill_injection}\n\n{exp_injection}"
                print(f"  [Injection] Combined raw skill + experiences for {question_id}")
            else:
                full_injection = skill_injection
        elif skill_injection:
            full_injection = skill_injection
        elif exp_injection:
            full_injection = exp_injection
        else:
            full_injection = ""
        
        dynamic_system_prompt = (
            f"{full_injection}\n\n{base_system_prompt}" if full_injection and base_system_prompt
            else (full_injection or base_system_prompt)
        )
        
        # 保存样本级检索/重写细节，便于复现实验和后续经验更新。
        if exp_enabled and sample_dir:
            save_retrieval_info(experience_retriever, sample_dir, retrieved_exps or None, 
                              original_retrieved_exps, use_rewrite, retrieval_info=retrieval_info)
        
        return _finalize_prompt(args, dynamic_system_prompt, update_global_prompts)
        
    except Exception as e:
        logger.warning(f"Failed to process for {question_id}: {e}", exc_info=True)
        print(f"  [Pipeline] Warning: {e}")
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)


def _process_skill(args, query_text, retrieved_exps, images, llm_client, sample_dir, question_id):
    """Process skill: adapt or use raw based on skill_adaptation flag.
        流程技能：根据“技能适应”标志来调整或使用原始数据。
    """
    # 对应论文中的 Skill Adaptation。
    try:
        with open(args.skill_library, 'r', encoding='utf-8') as f:
            base_skill = f.read()
        
        use_adaptation = getattr(args, 'skill_adaptation', True)  # Default: adapt
        
        # 适配模式需要 LLM 把通用 skill 改写成样本专用 skill。
        if use_adaptation and llm_client is None:
            llm_client = ExperienceLLM()
        
        exp_text = "\n\n".join(f"--- {k} ---\n{v}" for k, v in retrieved_exps.items()) if retrieved_exps else ""
        
        if use_adaptation:
            # === Standard Adaptation Mode ===
            adapted_skill = adapt_skill_for_task(base_skill, exp_text, query_text, llm_client, images)
            skill_content = adapted_skill
            print(f"  [Skill] Adapted for {question_id}")
            
            if sample_dir:
                with open(os.path.join(sample_dir, "tt_skill_adapted.md"), 'w', encoding='utf-8') as f:
                    f.write(skill_content)
        
        else:
            # === Raw Mode ===
            # 关闭适配时直接注入原始技能库。
            skill_content = base_skill
            print(f"  [Skill] Using raw skill for {question_id}")
            
            if sample_dir:
                with open(os.path.join(sample_dir, "tt_skill_original.md"), 'w', encoding='utf-8') as f:
                    f.write(skill_content)
        
        return SKILL_INJECTION_HEADER.format(skill_content=skill_content)
    except Exception as e:
        print(f"  [Skill] Warning: {e}")
        return ""


def _finalize_prompt(args, prompt, update_global_prompts):
    """Helper to finalize prompt: update globals or set on args."""
    # 串行路径改全局 prompt；并行路径把 prompt 放到样本 args，避免线程互相覆盖。
    if update_global_prompts:
        set_processors_prompts(prompt)
    else:
        args._sample_system_prompt = prompt
    return prompt


def _load_images_for_retrieval(sample, args):
    """
    Load images from original data source for experience retrieval.
    从原始数据源加载图像以进行经验检索

    Args:
        sample: Data sample
        args: Command line arguments
        
    Returns:
        list or None: List of processed images, or None if no images found
        列表或 None：已处理图像的列表，若未找到图像则为 None
    """
    if not (hasattr(args, 'image_folder') and args.image_folder):
        return None
    
    image_paths = sample.get('images', [])
    if not image_paths:
        return None
    
    images = []
    max_pixels = getattr(args, 'max_pixels', 2000000)
    min_pixels = getattr(args, 'min_pixels', 40000)
    
    for img_path in image_paths:
        try:
            # 数据中的图片路径是相对 image_folder 的相对路径。
            full_path = os.path.join(args.image_folder, img_path)
            if not os.path.exists(full_path):
                print(f"  [Retrieval] Warning: Image file not found: {full_path}")
                continue
            
            # 与正式推理保持一致的图片缩放/像素限制，对图片大小进行调整
            original_image = fetch_image({'image': full_path, 'max_pixels': max_pixels})
            processed_image = process_image(original_image, max_pixels, min_pixels)
            images.append(processed_image)
        except Exception as e:
            logger.warning(f"Failed to load image from {img_path}: {e}", exc_info=True)
            print(f"  [Retrieval] Warning: Failed to load image from {img_path}: {e}")
    
    if images:
        # print(f"  [Retrieval] Successfully loaded {len(images)} image(s) for task decomposition")
        pass
    else:
        print(f"  [Retrieval] Warning: Failed to load any images from original data source")
    
    return images if images else None


def _parse_exp_blocks(section: str) -> Dict[str, str]:
    """Parse [id]\\ncontent blocks from a section string. Returns dict id -> content."""
    # 从检索日志中恢复 [经验ID] -> 经验文本。
    result = {}
    exp_blocks = re.split(r'\n\[([^\]]+)\]\n', section)
    for i in range(1, len(exp_blocks), 2):
        exp_id = exp_blocks[i]
        exp_content = exp_blocks[i + 1].strip().strip('-').strip() if i + 1 < len(exp_blocks) else ""
        if exp_content:
            result[exp_id] = exp_content
    return result


def _parse_exp_ids(section: str) -> set:
    """Parse experience IDs from [id] markers in a section string."""
    # 只提取经验 ID，用来判断哪些经验最终被使用。
    return set(re.findall(r'\[([^\]]+)\]', section))


def get_used_original_experiences(sample_dir: str) -> Dict[str, str]:
    """
    Get original experiences that were actually used (not skipped during rewrite).
    
    Returns the original content of experiences that survived the rewrite process.
    If rewrite was not used, returns the retrieved experiences as-is.
    
    Args:
        sample_dir: Sample directory path
        
    Returns:
        Dict mapping experience ID to original content, or empty dict if not available
    """
    # 对应 H_usage：把本次实际使用的经验反馈给后续积累。
    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    if not os.path.exists(retrieval_info_file):
        return {}
    
    try:
        with open(retrieval_info_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析检索日志，区分“原始召回经验”和“重写后保留经验”。
        original_exps = {}
        rewritten_ids = set()
        
        # 提取重写前的原始经验。
        if "ORIGINAL RETRIEVED EXPERIENCES" in content:
            orig_section = content.split("ORIGINAL RETRIEVED EXPERIENCES")[1]
            if "REWRITTEN EXPERIENCES" in orig_section:
                orig_section = orig_section.split("REWRITTEN EXPERIENCES")[0]
            elif "RETRIEVED EXPERIENCE CONTENT" in orig_section:
                orig_section = orig_section.split("RETRIEVED EXPERIENCE CONTENT")[0]
            original_exps = _parse_exp_blocks(orig_section)
        
        # 提取重写后仍存在的经验 ID；这些才算本轮真正使用。
        if "REWRITTEN EXPERIENCES" in content:
            rewrite_section = content.split("REWRITTEN EXPERIENCES")[1]
            rewritten_ids = _parse_exp_ids(rewrite_section)
        elif "RETRIEVED EXPERIENCE CONTENT" in content:
            # No rewrite used, all retrieved experiences are "used"
            rewrite_section = content.split("RETRIEVED EXPERIENCE CONTENT")[1]
            rewritten_ids = _parse_exp_ids(rewrite_section)
            if not original_exps:
                original_exps = _parse_exp_blocks(rewrite_section)
                rewritten_ids = set(original_exps.keys())
        
        # 返回“被使用经验”的原始文本，供经验库更新阶段追踪 H_usage。
        return {k: v for k, v in original_exps.items() if k in rewritten_ids}
    
    except Exception as e:
        logger.warning(f"Failed to parse retrieval info: {e}")
        return {}


def save_retrieval_info(experience_retriever, sample_dir, retrieved_exps=None, original_retrieved_exps=None, rewrite_used=False, retrieval_info=None):
    """
    Save retrieval information to file, including rewrite information if applicable.
    将检索信息保存至文件中，包括（如适用）重写后的信息。

    Args:
        experience_retriever: ExperienceRetriever instance
        sample_dir: Sample directory path
        retrieved_exps: Final experiences dict (after rewrite if rewrite was used) or None
        original_retrieved_exps: Original retrieved experiences dict (before rewrite) or None
        rewrite_used: Whether rewrite was applied
        retrieval_info: Optional retrieval info dict (if provided, used instead of getting from retriever)
    """
    if retrieval_info is None:
        retrieval_info = experience_retriever.get_last_retrieval_info() if experience_retriever else None
    if not retrieval_info:
        return
    
    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 写入人可读日志：query、子任务、召回 ID、重写前后内容。
    with open(retrieval_info_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIENCE RETRIEVAL INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Original Query:\n{retrieval_info.get('original_query', 'N/A')}\n\n")
        f.write(f"Decomposition Used: {retrieval_info.get('decomposition_used', False)}\n")
        f.write(f"Rewrite Used: {rewrite_used}\n\n")
        
        if retrieval_info.get('decomposition_used', False):
            f.write(f"Subtasks ({len(retrieval_info.get('subtasks', []))}):\n")
            for i, subtask in enumerate(retrieval_info.get('subtasks', []), 1):
                f.write(f"\n  {i}. Type: {subtask.get('type', 'unknown')}\n")
                f.write(f"     Query: {subtask.get('query', '')}\n")
            
            f.write(f"\n\nRetrieval Details:\n")
            for i, detail in enumerate(retrieval_info.get('retrieval_details', []), 1):
                f.write(f"\n  {i}. Subtask: {detail.get('subtask_type', 'unknown')}\n")
                f.write(f"     Query: {detail.get('query', '')}\n")
                f.write(f"     Retrieved {detail.get('count', 0)} experiences:\n")
                for exp_id in detail.get('retrieved_experience_ids', []):
                    f.write(f"       - {exp_id}\n")
        
        f.write(f"\n\nFinal Retrieved Experiences ({retrieval_info.get('total_unique_experiences', 0)} total):\n")
        for exp_id in retrieval_info.get('retrieved_experiences', []):
            f.write(f"  - {exp_id}\n")
        
        # 如启用 rewrite，保留重写前文本，方便对照。
        if rewrite_used and original_retrieved_exps:
            f.write(f"\n\n{'=' * 80}\n")
            f.write("ORIGINAL RETRIEVED EXPERIENCES (Before Rewrite)\n")
            f.write("=" * 80 + "\n")
            for exp_id, exp_text in original_retrieved_exps.items():
                f.write(f"\n[{exp_id}]\n")
                f.write("-" * 80 + "\n")
                f.write(exp_text)
                f.write("\n\n")
        
        # 写入最终注入给模型的经验文本。
        if retrieved_exps:
            section_title = "REWRITTEN EXPERIENCES (After Rewrite)" if rewrite_used else "RETRIEVED EXPERIENCE CONTENT"
            f.write(f"\n\n{'=' * 80}\n")
            f.write(f"{section_title}\n")
            f.write("=" * 80 + "\n")
            for exp_id, exp_text in retrieved_exps.items():
                f.write(f"\n[{exp_id}]\n")
                f.write("-" * 80 + "\n")
                f.write(exp_text)
                f.write("\n\n")
        else:
            f.write("\nNo relevant experiences were retrieved for this query.\n")


def compute_and_save_sample_summary(question_id, sample_rollout_results, rollouts_per_sample, sample_dir):
    """
    Compute and save sample-level pass@k and average@k metrics.
    计算并保存样本级别的“通过率@k”和“平均值@k”指标
    
    Args:
        question_id: Question ID
        sample_rollout_results: List of rollout result dictionaries
        rollouts_per_sample: Number of rollouts per sample
        sample_dir: Sample directory path
        
    Returns:
        dict: Sample summary dictionary
    """
    if not sample_rollout_results:
        return None
    
    # 并行 rollout 完成顺序不稳定，先按 rollout_idx 还原顺序。
    if sample_rollout_results and '_rollout_idx' in sample_rollout_results[0]:
        sorted_results = sorted(sample_rollout_results, key=lambda r: r.get('_rollout_idx', 0))
    else:
        sorted_results = sample_rollout_results

    accuracies = [r["accuracy_score"] for r in sorted_results]
    sample_summary = {
        "question_id": question_id,
        "num_rollouts": len(sorted_results),
        "accuracies": accuracies,
    }

    # 单题指标：任一 rollout 正确即 pass@k，同时计算前 k 次平均分。
    for k in range(1, len(accuracies) + 1):
        acc_at_k = accuracies[:k]
        sample_summary[f"pass@{k}"] = 1.0 if any(acc == 1.0 for acc in acc_at_k) else 0.0
        sample_summary[f"average@{k}"] = sum(acc_at_k) / k

    # 每道题单独落盘，数据集汇总时再读取。
    with open(os.path.join(sample_dir, SAMPLE_SUMMARY_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(sample_summary, f, indent=4, ensure_ascii=False)
    
    print(f"  \nSample {question_id}: pass@{rollouts_per_sample}={sample_summary[f'pass@{rollouts_per_sample}']:.2f}, "
          f"average@{rollouts_per_sample}={sample_summary[f'average@{rollouts_per_sample}']:.4f}")
    
    return sample_summary


def check_sample_completed(sample_dir, args):
    """
    Check if a sample is already completed (has traj.jsonl with turn_idx and metrics.json).
    检查样本是否已完整（是否包含 traj.jsonl 文件以及带有 turn_idx 和 metrics.json 的内容）

    Args:
        sample_dir: Sample directory path
        args: Command line arguments
        
    Returns:
        dict or None: Sample info dict if completed, None otherwise
    """
    if not getattr(args, 'skip_completed', False):
        return None
    
    traj_file = os.path.join(sample_dir, 'traj.jsonl')
    metrics_file = os.path.join(sample_dir, 'metrics.json')
    
    # 断点续跑：同时存在轨迹和指标文件才认为该样本已完成。
    is_complete = False
    if os.path.exists(traj_file) and os.path.exists(metrics_file):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip()]
                # Check if there are any turn_idx entries
                for line in lines:
                    try:
                        traj_data = json.loads(line)
                        if 'turn_idx' in traj_data and traj_data['turn_idx'] is not None:
                            is_complete = True
                            break
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Error parsing trajectory line: {e}")
                        pass
        except Exception as e:
            logger.debug(f"Error reading trajectory file: {e}")
            pass
    
    if not is_complete:
        return None
    
    # Try to load existing result
    sample_rollout_results = []
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # 从轨迹中恢复 turn 数，保证汇总统计不失真。
        num_turns = 0
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                max_turn_idx = -1
                for line in f:
                    if line.strip():
                        try:
                            traj_data = json.loads(line)
                            turn_idx = traj_data.get('turn_idx')
                            if turn_idx is not None and turn_idx > max_turn_idx:
                                max_turn_idx = turn_idx
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug(f"Error parsing trajectory line: {e}")
                            pass
            num_turns = max_turn_idx + 1 if max_turn_idx >= 0 else 0
        except Exception as e:
            logger.debug(f"Error reading trajectory file for turn count: {e}")
            pass
        
        # 构造最小 conversation_history，只用于后续统计 turn 数。
        conversation_history = [{"role": "assistant", "content": ""} for _ in range(num_turns)] if num_turns > 0 else []
        
        # 返回与新跑样本一致的最小结果结构。
        sample_rollout_results.append({
            'question_id': os.path.basename(sample_dir),
            'accuracy_score': metrics.get('accuracy_score', 0.0),
            'conversation_history': conversation_history,
        })
    except Exception as e:
        logger.debug(f"Error loading existing metrics: {e}")
        pass
    
    return {
        'sample_rollout_results': sample_rollout_results
    }


def prepare_sample_args(args):
    """
    Create a thread-safe copy of args for parallel execution.
    为并行执行创建一个线程安全的 args 复制版本
    
    Args:
        args: Original command line arguments
        参数：原始命令行参数
        
    Returns:
        A shallow copy of args with deep-copied tool_configs
        一个浅层复制的 args，以及深度复制的 tool_configs
    """
    sample_args = copy.copy(args)
    
    # tool_configs 里可能包含 work_dir 等可变字段，并行时必须隔离。
    if hasattr(args, 'tool_configs') and args.tool_configs:
        sample_args.tool_configs = copy.deepcopy(args.tool_configs)
    
    return sample_args


def get_sample_metadata(sample, sample_idx, output_dir):
    """
    Extract question_id and sample_dir from a sample.
    从样本中提取question_id和sample_dir。
    
    Args:
        sample: Data sample dictionary
        sample_idx: Sample index (for fallback question_id)
        output_dir: Output directory path
        
    Returns:
        tuple: (question_id, sample_dir)
    """
    question_id = sample.get('doc_id', sample.get('question_id', f"sample_{sample_idx}"))
    sample_dir = os.path.join(output_dir, question_id)
    return question_id, sample_dir


def compute_dataset_summary(data, args):
    """
    Compute and save dataset-level pass@k and average@k metrics.
    
    Args:
        data: List of data samples
        args: Command line arguments
        
    Returns:
        dict: Dataset summary dictionary or None if no summaries found
    """
    # 读取每道题已保存的 metrics_sample.json。
    sample_summaries = []
    for sample in data:
        question_id = sample.get('doc_id', sample.get('question_id', f"sample_{data.index(sample)}"))
        summary_path = os.path.join(args.output_dir, question_id, SAMPLE_SUMMARY_FILENAME)
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                sample_summaries.append(json.load(f))
    
    if not sample_summaries:
        return None
    
    # 汇总为数据集级 pass@k / average@k。
    dataset_summary = {
        "num_samples": len(sample_summaries),
        "num_rollouts": args.rollouts_per_sample,
        "k_metrics": {}
    }
    
    for k in range(1, args.rollouts_per_sample + 1):
        pass_values = [s[f"pass@{k}"] for s in sample_summaries if f"pass@{k}" in s]
        avg_values = [s[f"average@{k}"] for s in sample_summaries if f"average@{k}" in s]
        
        dataset_summary["k_metrics"][k] = {
            "pass_at_k": round(sum(pass_values) / len(pass_values), 4) if pass_values else 0.0,
            "average_at_k": round(sum(avg_values) / len(avg_values), 4) if avg_values else 0.0,
        }
    
    # 数据集指标统一写到输出目录根部。
    with open(os.path.join(args.output_dir, DATASET_SUMMARY_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print(f"Total samples: {dataset_summary['num_samples']}")
    print(f"Rollouts per sample: {dataset_summary['num_rollouts']}\n")
    for k in range(1, args.rollouts_per_sample + 1):
        metrics = dataset_summary["k_metrics"][k]
        print(f"k={k}:")
        print(f"  Pass@{k}:    {metrics['pass_at_k']:.4f}")
        print(f"  Average@{k}: {metrics['average_at_k']:.4f}")
    print("="*80)
    
    return dataset_summary


def reload_experiences(args, base_system_prompt, experience_retriever=None):
    """
    Reload experiences from library and update system prompt and/or retriever.
    
    Args:
        args: Command line arguments
        base_system_prompt: Base system prompt (without experiences)
        experience_retriever: Optional ExperienceRetriever instance to update (for retrieval mode)
        
    Returns:
        str: Updated system prompt with experiences injected
    """
    if not (getattr(args, 'experience_enable', False) and args.experience_library):
        return base_system_prompt
    
    try:
        exps = load_experiences(args.experience_library)
        use_retrieval = getattr(args, 'experience_retrieval', False)

        if use_retrieval and experience_retriever is not None:
            # 在线生成新经验后，检索模式只需增量刷新向量索引。
            print(f"  Updating experience retriever with {len(exps)} experiences...")
            experience_retriever.update_experiences(exps, incremental=True)
            print(f"  Experience retriever updated successfully.")
            return base_system_prompt

        # 非检索模式则重新拼接全量经验 prompt。
        injection_text = format_for_prompt(exps, max_items=getattr(args, 'experience_max_items', 256))
        if injection_text:
            system_prompt = (injection_text + "\n\n" + base_system_prompt) if base_system_prompt else injection_text
            set_processors_prompts(system_prompt)
            print(f"  Reloaded {len(exps)} experiences for next batch.")
            return system_prompt
    except Exception as e:
        print(f"  Warning: Failed to reload experiences: {e}")
        import traceback
        traceback.print_exc()

    return base_system_prompt


def execute_pipeline_parallel_processing(
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
    """
    Executes a pipeline parallel processing of samples to maximize hardware utilization:
    
    Pipeline stages:
    1. Retrieval Stage: Submits experience retrieval tasks for samples. To prevent 
       "worker starvation" for the reasoning stage, retrieval tasks are submitted 
       gradually as slots open up.
    2. Reasoning (Rollout) Stage: As soon as a sample's retrieval (and optional rewrite)
       completes, all its rollouts (pass@k) are immediately submitted to the pool.
    3. Management Stage: A unified monitoring loop uses wait() to react to any completed 
       future (retrieve or rollout), ensuring smooth transition between stages.
    
    Args:
        samples: List of sample dictionaries from the dataset.
        sample_indices: List of actual sample indices (mapping to original dataset).
        sample_results_dict: Shared dictionary to store results for each sample.
        args: Command line arguments containing configuration (num_workers, etc).
        experience_retriever: ExperienceRetriever instance. If None, retrieval stage 
            may still run if skill_inference is enabled (skill-only mode).
        base_system_prompt: The initial system prompt used if retrieval is disabled or fails.
        sampling_params: Parameters for the LLM call (temperature, top_p, etc).
        run_single_rollout_func: The core function to execute one rollout (one pass).
        progress_desc: String description for the tqdm progress bar.
        all_results_list: Optional list to collect every single rollout result across all samples.
        
    Returns:
        None (results are mutated in sample_results_dict and all_results_list).
    """
    # 工程实现：把“检索/适配”和“多 rollout 执行”做成并行流水线。
    from tqdm import tqdm  # type: ignore

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # active_futures 统一管理检索任务和 rollout 任务。
        active_futures = {}  # future -> info dict
        pending_retrieval_samples = []
        
        # 预先整理每个样本的独立 args，避免线程共享状态。
        for local_idx, sample in enumerate(samples):
            actual_sample_idx = sample_indices[local_idx]
            question_id = sample_results_dict[actual_sample_idx]['question_id']
            sample_dir = sample_results_dict[actual_sample_idx]['sample_dir']
            sample_summary_path = os.path.join(sample_dir, SAMPLE_SUMMARY_FILENAME)
            if getattr(args, 'skip_completed', False) and os.path.exists(sample_summary_path):
                print(f"  [Pipeline] Sample {question_id}: already has {SAMPLE_SUMMARY_FILENAME}, skipping...")
                continue
            sample_args = prepare_sample_args(args)
            pending_retrieval_samples.append({
                'sample_idx': actual_sample_idx,
                'sample': sample,
                'sample_args': sample_args,
                'question_id': question_id,
                'sample_dir': sample_dir
            })

        # 进度条按 rollout 数计数，而不是按样本数计数。
        total_rollouts = len(samples) * args.rollouts_per_sample
        skipped_rollouts = (len(samples) - len(pending_retrieval_samples)) * args.rollouts_per_sample
        if skipped_rollouts:
            print(f"  [Pipeline] Skipped {skipped_rollouts} completed rollout(s).")
        total_rollouts = len(pending_retrieval_samples) * args.rollouts_per_sample
        if total_rollouts == 0:
            print("  [Pipeline] All samples in this batch are already completed.")
            return
        pbar = tqdm(total=total_rollouts, desc=progress_desc or "Processing rollouts")
        
        rollout_counts = {}  # track completed rollouts per sample
        completed_samples = set()  # track samples with all rollouts finished
        next_sample_idx = 0
        
        # 只启用 skill 时也需要先做样本级 prompt 处理。
        skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
        needs_retrieval_stage = experience_retriever is not None or skill_enabled
        
        # 检索任务渐进提交，避免把线程池全塞满导致 rollout 饥饿。
        max_concurrent_retrievals = args.num_workers if needs_retrieval_stage else 0
        active_retrieval_count = 0

        # 初始提交：无检索时直接提交所有 rollout；有检索时先提交检索。
        if not needs_retrieval_stage:
            # 无经验/技能注入：每个 rollout 都使用同一个基础 system prompt。
            for item in pending_retrieval_samples:
                actual_sample_idx = item['sample_idx']
                sample = item['sample']
                sample_args = item['sample_args']
                sample_args._sample_system_prompt = base_system_prompt
                for rollout_idx in range(args.rollouts_per_sample):
                    f = executor.submit(run_single_rollout_func, sample, sample_args, sampling_params, rollout_idx)
                    active_futures[f] = {'type': 'rollout', 'sample_idx': actual_sample_idx, 'rollout_idx': rollout_idx}
            next_sample_idx = len(pending_retrieval_samples)
        else:
            # 有经验/技能注入：先为前几个样本生成样本级 prompt。
            while next_sample_idx < len(pending_retrieval_samples) and active_retrieval_count < max_concurrent_retrievals:
                item = pending_retrieval_samples[next_sample_idx]
                f = executor.submit(
                    retrieve_experiences_for_sample,
                    item['sample'], item['sample_args'], experience_retriever, base_system_prompt,
                    item['question_id'], item['sample_dir'], update_global_prompts=False
                )
                active_futures[f] = {
                    'type': 'retrieve', 
                    'sample_idx': item['sample_idx'], 
                    'sample': item['sample'], 
                    'sample_args': item['sample_args'],
                    'question_id': item['question_id'],
                    'sample_dir': item['sample_dir']
                }
                active_retrieval_count += 1
                next_sample_idx += 1

        # 主循环：任意 future 完成就立即处理，保持线程池持续有活干。
        while active_futures:
            # wait(FIRST_COMPLETED) 同时监听检索和 rollout 两类任务。
            done_futures, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in done_futures:
                if future not in active_futures:
                    continue
                    
                info = active_futures.pop(future)
                
                if info['type'] == 'retrieve':
                    # 检索/技能处理完成后，立即为该样本提交所有 rollout。
                    active_retrieval_count -= 1
                    actual_sample_idx = info['sample_idx']
                    sample = info['sample']
                    sample_args = info['sample_args']
                    question_id = info['question_id']
                    
                    try:
                        future.result()  # Ensure retrieval finished successfully
                        print(f"  [Pipeline] Sample {question_id}: Retrieve completed, submitting {args.rollouts_per_sample} rollouts...")
                    except Exception as e:
                        print(f"  [Pipeline] Error retrieving for sample {question_id}: {e}. Falling back to base prompt.")
                        sample_args._sample_system_prompt = base_system_prompt
                    
                    # 同一样本的多个 rollout 对应 pass@k 的多次尝试。
                    for rollout_idx in range(args.rollouts_per_sample):
                        f_rollout = executor.submit(run_single_rollout_func, sample, sample_args, sampling_params, rollout_idx)
                        active_futures[f_rollout] = {'type': 'rollout', 'sample_idx': actual_sample_idx, 'rollout_idx': rollout_idx}
                    
                    # 补一个新的检索任务进来，形成检索与推理的流水线。
                    if next_sample_idx < len(pending_retrieval_samples):
                        item = pending_retrieval_samples[next_sample_idx]
                        f_new = executor.submit(
                            retrieve_experiences_for_sample,
                            item['sample'], item['sample_args'], experience_retriever, base_system_prompt,
                            item['question_id'], item['sample_dir'], update_global_prompts=False
                        )
                        active_futures[f_new] = {
                            'type': 'retrieve', 
                            'sample_idx': item['sample_idx'], 
                            'sample': item['sample'], 
                            'sample_args': item['sample_args'],
                            'question_id': item['question_id'],
                            'sample_dir': item['sample_dir']
                        }
                        active_retrieval_count += 1
                        next_sample_idx += 1
                
                elif info['type'] == 'rollout':
                    # 单次 rollout 完成后收集结果，并更新该样本的完成计数。
                    actual_sample_idx = info['sample_idx']
                    rollout_idx = info['rollout_idx']
                    
                    try:
                        result = future.result()
                        if result:
                            # 记录 rollout_idx，后面计算 pass@k 时按原顺序排序。
                            result['_rollout_idx'] = rollout_idx
                            sample_results_dict[actual_sample_idx]['results'].append(result)
                            if all_results_list is not None:
                                all_results_list.append(result)
                    except Exception as e:
                        print(f"  [Pipeline] Error in rollout {rollout_idx} for sample {actual_sample_idx}: {e}")
                    
                    # 一个 rollout 完成，对进度条推进一次。
                    pbar.update(1)
                    
                    # 统计当前样本已经完成了几次 rollout。
                    if actual_sample_idx not in rollout_counts:
                        rollout_counts[actual_sample_idx] = 0
                    rollout_counts[actual_sample_idx] += 1
                    
                    # 该样本所有 rollout 完成后，立即写单题指标。
                    if rollout_counts[actual_sample_idx] == args.rollouts_per_sample and actual_sample_idx not in completed_samples:
                        completed_samples.add(actual_sample_idx)
                        sample_info = sample_results_dict[actual_sample_idx]
                        if sample_info['results']:
                            compute_and_save_sample_summary(
                                sample_info['question_id'], sample_info['results'],
                                args.rollouts_per_sample, sample_info['sample_dir']
                            )

        pbar.close()

