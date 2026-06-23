from __future__ import annotations

import time
from typing import Dict, List, Optional

from PIL import Image

from prompts.experience_generation_templates import INTRA_SAMPLE_CRITIQUE, SUPERVISED_CORRECTION_PROMPT

from utils.json_utils import parse_llm_json_list, parse_llm_json_object
from .experience_llm import ExperienceLLM
from .schema import format_experience_for_prompt
from .config import ExperienceGenerationSettings
from .utils import find_original_images, write_debug_texts


def settings() -> ExperienceGenerationSettings:
    return ExperienceGenerationSettings.from_runtime()


def format_question(question: str, system_prompt: str) -> str:
    return f"{question.strip()}\n\n==== System prompt:\n{system_prompt.strip()} ===="


def format_summaries(summaries: Dict[str, str]) -> str:
    return "\n\n".join(
        f"Trajectory {i + 1}:\n{summary}"
        for i, summary in enumerate(summaries.values())
    )


def format_used_experiences(used_experiences: Optional[Dict[str, str]]) -> str:
    if not used_experiences:
        return "(No experiences were retrieved for this sample)"
    return "\n\n".join(
        f"[{exp_id}] {format_experience_for_prompt(exp_id, exp_content)}"
        for exp_id, exp_content in used_experiences.items()
    )


def chat_with_sample_images(
    llm: ExperienceLLM,
    prompt: str,
    *,
    debug_dir: Optional[str],
    max_tokens: int,
    temperature: Optional[float] = None,
) -> str:
    question_images = find_original_images(debug_dir)
    if question_images:
        pil_images = [Image.open(path).convert("RGB") for path in question_images]
        return llm.chat_with_image(
            prompt=prompt,
            image=pil_images,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return llm.chat(prompt, max_tokens=max_tokens, temperature=temperature)


def call_with_retries(
    label: str,
    llm: ExperienceLLM,
    prompt: str,
    *,
    debug_dir: Optional[str],
    max_tokens: int,
    temperature: Optional[float] = None,
    raise_on_final_failure: bool = True,
) -> Optional[str]:
    max_retries = settings().critique_retries
    for attempt in range(max_retries):
        try:
            return chat_with_sample_images(
                llm,
                prompt,
                debug_dir=debug_dir,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            if attempt == max_retries - 1:
                if raise_on_final_failure:
                    raise
                print(f"Warning: {label} failed: {exc}")
                return None
            print(f"Warning: {label} failed (attempt {attempt + 1}/{max_retries}): {exc}, retrying...")
            time.sleep(2 ** attempt)
    return None

# Pipeline 1: Generate experience operations from rollout summaries.
def experiences_from_rollouts(
    question: str,
    groundtruth: str,
    summaries: Dict[str, str],
    llm: ExperienceLLM,
    max_ops: int = 2,
    debug_dir: Optional[str] = None,
    system_prompt: str = "",
    used_experiences: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Generate add/modify operations by comparing rollout summaries."""
    if not summaries:
        return []

    prompt = INTRA_SAMPLE_CRITIQUE.format(
        max_ops=max_ops,
        question=format_question(question, system_prompt),
        summaries=format_summaries(summaries),
        experiences=format_used_experiences(used_experiences),
        groundtruth=groundtruth or "[REDACTED]",
    )

    start = time.time()
    response = call_with_retries(
        "Experience critique",
        llm,
        prompt,
        debug_dir=debug_dir,
        max_tokens=settings().critique_max_tokens,
    )
    print(f"  [Timing] cross rollout critique generation: {time.time() - start:.1f}s")
    if response is None:
        raise RuntimeError("Failed to generate experiences after all retries")

    write_debug_texts(
        debug_dir,
        {
            "exp_intra_prompt.txt": prompt,
            "exp_intra_resp.txt": response,
        },
    )

    try:
        return [item for item in parse_llm_json_list(response) if isinstance(item, dict)]
    except Exception:
        return []


def supervised_correction_prompt(
    question: str,
    groundtruth: str,
    summaries: Dict[str, str],
    system_prompt: str,
    used_experiences: Optional[Dict[str, str]],
) -> str:
    return SUPERVISED_CORRECTION_PROMPT.format(
        question=format_question(question, system_prompt),
        summaries=format_summaries(summaries),
        groundtruth=groundtruth,
        experiences=format_used_experiences(used_experiences),
    )


def parse_supervised_proposal(response: Optional[str]) -> Optional[dict]:
    try:
        proposal = parse_llm_json_object(response or "")
    except Exception as exc:
        print(f"  Warning: failed to parse supervised correction response: {exc}")
        return None
    return proposal if proposal.get("recoverable") is True else None


def supervised_proposal_to_operation(proposal: dict) -> dict:
    return {
        "option": "add",
        "kind": "success",
        "content": proposal.get("content"),
        "task_tags": proposal.get("task_tags", []),
        "tool_tags": proposal.get("tool_tags", []),
        "tool_sequence": proposal.get("tool_sequence", []),
        "validation_checks": proposal.get("validation_checks", []),
        "fallback_plan": proposal.get("fallback_plan", []),
        "anti_pattern": proposal.get("anti_pattern", ""),
        "code_template": proposal.get("code_template", ""),
        "recovery_status": "candidate_unverified",
        "stats": {
            "support_count": 1,
            "used_count": 0,
            "success_after_used": 0,
            "failure_after_used": 0,
            "recovered_from_failure": True,
            "supervised_correction": True,
        },
    }

# Pipeline 2: Supervised failure recovery experience generation.
def recovery_experience(
    question: str,
    groundtruth: str,
    summaries: Dict[str, str],
    llm: ExperienceLLM,
    *,
    debug_dir: Optional[str] = None,
    system_prompt: str = "",
    used_experiences: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Use training labels to turn all-failed rollouts into one recovery candidate."""
    if not summaries or not str(groundtruth or "").strip():
        return []

    prompt = supervised_correction_prompt(
        question, groundtruth, summaries, system_prompt, used_experiences
    )
    start = time.time()
    response = call_with_retries(
        "supervised correction",
        llm,
        prompt,
        debug_dir=debug_dir,
        max_tokens=4096,
        temperature=0.2,
        raise_on_final_failure=False,
    )
    print(f"[Timing] supervised correction generation: {time.time() - start:.1f}s")

    write_debug_texts(
        debug_dir,
        {
            "exp_supervised_correction_prompt.txt": prompt,
            "exp_supervised_correction_resp.txt": response or "",
        },
    )

    proposal = parse_supervised_proposal(response)
    if proposal is None:
        return []
    return [supervised_proposal_to_operation(proposal)]
