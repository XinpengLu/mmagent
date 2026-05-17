import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image

from prompts.experience_prompts import (
    INTRA_SAMPLE_CRITIQUE,
)

from .llm_client import ExperienceLLM
from .experience_schema import format_experience_for_prompt


# --------- Constants ---------

# 经验生成 LLM 调用失败时的重试次数。
MAX_RETRIES = 3

# cross-rollout critique 的输出 token 上限。
MAX_TOKENS = 12288


# --------- Intra-sample critique ---------

def intra_sample_experiences(question: str, groundtruth: str, summaries: Dict[str, str], llm: ExperienceLLM, max_ops: int = 2, debug_dir: Optional[str] = None, system_prompt: Optional[str] = None, used_experiences: Optional[Dict[str, str]] = None) -> List[dict]:
    """Generate experience critiques from trajectory summaries.
    
    Args:
        question: The question/task being solved
        groundtruth: Ground truth answer (if available)
        summaries: Dictionary mapping node_id to summary text
        llm: ExperienceLLM instance for generating critiques
        max_ops: Maximum number of operations to generate
        debug_dir: Optional directory to save debug information
        system_prompt: Optional system prompt used in the task
        used_experiences: Optional dict mapping experience ID to original content 
                         (experiences that were retrieved and used for this sample)
    
    Returns:
        List of experience operation dictionaries
    """
    # 对应 Phase I / Cross-Rollout Critique：比较成功和失败轨迹，抽取经验操作。
    if not summaries:
        return []
    # 把每个 rollout 的摘要整理成统一文本块。
    formatted_summaries = []
    for i, (nid, summ) in enumerate(summaries.items()):
        formatted_summaries.append(f"Trajectory {i+1} (node {nid}):\n{summ}")
    
    # 题目和 system prompt 一起给 critique，帮助判断经验是否可迁移。
    question_section_parts = []
    if isinstance(question, str) and question.strip():
        question_section_parts.append(question.strip())
    if isinstance(system_prompt, str) and system_prompt.strip():
        question_section_parts.append(f"\n==== System prompt:\n{system_prompt.strip()}")
    
    formatted_question = "\n".join(question_section_parts) if question_section_parts else ""
    
    # 注入本题实际用过的经验，便于 LLM 生成 modify 操作而不是重复 add。
    formatted_experiences = ""
    if used_experiences:
        exp_lines = []
        for exp_id, exp_content in used_experiences.items():
            exp_lines.append(f"[{exp_id}] {format_experience_for_prompt(exp_id, exp_content)}")
        formatted_experiences = "\n\n".join(exp_lines)
    
    # prompt 要求输出 add/modify 等经验库操作。
    prompt = INTRA_SAMPLE_CRITIQUE.format(
        max_ops=max_ops,
        question=formatted_question,
        summaries="\n\n".join(formatted_summaries),
        experiences=formatted_experiences or "(No experiences were retrieved for this sample)",
        groundtruth=groundtruth or "[REDACTED]",
    )
    
    # 如果样本目录里有原始图片，就用多模态 critique。
    question_images = []
    if debug_dir:
        sample_path = Path(debug_dir)
        if sample_path.exists() and sample_path.is_dir():
            for img_file in sample_path.glob("original_image*"):
                if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    question_images.append(img_file)
    
    # LLM 调用带重试，网络/上游偶发错误不直接中断。
    resp = None
    llm_start = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            # 有原题图片时，让 critique 同时参考视觉内容。
            if question_images:
                pil_images = [Image.open(img_path).convert('RGB') for img_path in question_images]
                resp = llm.chat_with_image(
                    prompt=prompt,
                    image=pil_images if len(pil_images) > 1 else pil_images[0],
                    max_tokens=MAX_TOKENS,
                    return_placeholder_on_error=False
                )
            else:
                # 没有图片时退回纯文本 critique。
                resp = llm.chat(prompt, max_tokens=MAX_TOKENS)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  Warning: Experience critique failed (attempt {attempt+1}/{MAX_RETRIES}): {e}, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
    llm_time = time.time() - llm_start
    print(f"  [Timing] cross rollout critique generation: {llm_time:.1f}s")
    if resp is None:
        raise RuntimeError("Failed to generate experiences after all retries")
    if debug_dir:
        try:
            # 保存 prompt 和响应，便于分析经验生成质量。
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "exp_intra_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(os.path.join(debug_dir, "exp_intra_resp.txt"), "w", encoding="utf-8") as f:
                f.write(resp)
        except Exception:
            pass
    try:
        # 只接受 ```json fenced block 中的 list 操作。
        payload = resp.split("```json")[-1].split("```")[0]
        ops = json.loads(payload)
        if isinstance(ops, list):
            return ops
    except Exception:
        pass
    return []
