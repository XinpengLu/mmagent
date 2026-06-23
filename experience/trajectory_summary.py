"""Summarize rollout trajectories into text used for experience generation."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from prompts.experience_generation_templates import SINGLE_ROLLOUT_SUMMARY

from .experience_llm import ExperienceLLM
from .image_captioning import generate_image_captions
from .config import ExperienceGenerationSettings
from .utils import write_debug_texts


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def settings() -> ExperienceGenerationSettings:
    return ExperienceGenerationSettings.from_runtime()


def scan_all_images(sample_dir: str) -> Dict[str, Dict[str, Any]]:
    """Return rollout image files keyed by caption lookup name."""
    sample_path = Path(sample_dir)
    if not sample_path.exists() or not sample_path.is_dir():
        return {}

    image_map: Dict[str, Dict[str, Any]] = {}
    for item in sample_path.iterdir():
        if not item.is_dir() or not item.name.startswith("rollout_"):
            continue
        for image_path in item.glob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                key = f"{item.name}/{image_path.name}"
                image_map[key] = {
                    "file_path": image_path,
                    "rollout_dir": item.name,
                    "is_original": image_path.name.startswith("original_image"),
                }
    return image_map


def image_sort_key(key: str) -> tuple:
    base = os.path.basename(key)
    rollout = key.split("/", 1)[0]
    match = re.search(r"(\d+)", base)
    number = int(match.group(1)) if match else -1
    return rollout, base.startswith("tool_image"), number, base


def select_images_for_summary(all_images: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Keep original images plus a compact first/last sample of tool images."""
    config = settings()
    image_budget = config.trajectory_image_budget
    if len(all_images) <= image_budget:
        return all_images

    selected: Dict[str, Dict[str, Any]] = {}
    originals = [(key, info) for key, info in all_images.items() if info["is_original"]]
    for key, info in sorted(originals, key=lambda item: image_sort_key(item[0])):
        if len(selected) < image_budget:
            selected[key] = info

    by_rollout: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for key, info in all_images.items():
        if key in selected or info["is_original"]:
            continue
        by_rollout.setdefault(info["rollout_dir"], []).append((key, info))

    per_rollout = max(1, config.trajectory_tool_images_per_rollout)
    for rollout in sorted(by_rollout):
        items = sorted(by_rollout[rollout], key=lambda item: image_sort_key(item[0]))
        chosen = items[:per_rollout]
        if len(items) > per_rollout and per_rollout > 1:
            chosen = items[:1] + items[-(per_rollout - 1):]
        for key, info in chosen:
            if len(selected) < image_budget:
                selected[key] = info

    if len(selected) < image_budget:
        for key, info in sorted(all_images.items(), key=lambda item: image_sort_key(item[0])):
            if key not in selected:
                selected[key] = info
            if len(selected) >= image_budget:
                break

    dropped = len(all_images) - len(selected)
    if dropped > 0:
        print(
            f"  [Image Budget] Captioning {len(selected)}/{len(all_images)} images "
            f"(dropped {dropped}; budget={image_budget}, per_rollout={per_rollout})"
        )
    return selected


def get_caption_for_image(name_or_path: str, image_captions: Dict[str, str]) -> Optional[str]:
    image_stem = Path(name_or_path).stem
    for key, caption in image_captions.items():
        if Path(key).stem == image_stem:
            return caption
    return None


def truncate_for_summary(text: Any, max_chars: int, label: str) -> Any:
    """Truncate oversized text fields only in the LLM summary prompt."""
    if not isinstance(text, str):
        return text
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n...[truncated {omitted} chars from {label}; original length={len(text)}]...\n"
        + text[-tail:]
    )


def compact_record_for_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    config = settings()
    if record.get("initial_prompt"):
        record["initial_prompt"] = truncate_for_summary(
            record["initial_prompt"], config.trajectory_max_initial_prompt_chars, "initial_prompt"
        )
    if record.get("text_output"):
        record["text_output"] = truncate_for_summary(
            record["text_output"], config.trajectory_max_text_output_chars, "text_output"
        )

    tool_call = record.get("tool_call")
    if tool_call:
        for key, value in tool_call["parameters"].items():
            limit = config.trajectory_max_tool_code_chars if key == "code" else config.trajectory_max_other_field_chars
            tool_call["parameters"][key] = truncate_for_summary(value, limit, f"tool_call.parameters.{key}")
        tool_call["result"] = truncate_for_summary(
            tool_call["result"], config.trajectory_max_tool_result_chars, "tool_call.result"
        )

    for key, value in record.items():
        if key in {"initial_prompt", "text_output", "tool_call"}:
            continue
        record[key] = truncate_for_summary(value, config.trajectory_max_other_field_chars, key)
    return record


def replace_image_refs_in_jsonl(jsonl_content: str, image_captions: Dict[str, str]) -> str:
    """Replace markdown image refs and tool_image entries with caption text."""
    image_pattern = r"!\[([^\]]+\.(?:jpg|jpeg|png))\]\(([^\)]+\.(?:jpg|jpeg|png))\)"
    lines = []

    for line in jsonl_content.split("\n"):
        if not line.strip():
            lines.append(line)
            continue
        try:
            record = json.loads(line)
            if record.get("tool_call"):
                tool_call = record["tool_call"]

                def replace_with_caption(match: re.Match) -> str:
                    image_name = match.group(1) or match.group(2)
                    caption = get_caption_for_image(image_name, image_captions)
                    return f"[Image: {image_name}]\nCaption: {caption}" if caption else match.group(0)

                tool_call["result"] = re.sub(image_pattern, replace_with_caption, tool_call["result"])

            if record.get("tool_image"):
                tool_image = record["tool_image"]
                caption = get_caption_for_image(tool_image["image_name"], image_captions)
                if caption:
                    tool_image["caption"] = caption

            lines.append(json.dumps(compact_record_for_summary(record), ensure_ascii=False))
        except Exception:
            lines.append(truncate_for_summary(line, settings().trajectory_max_other_field_chars, "non_json_line"))

    return "\n".join(lines)


def replace_image_tags_in_question(question: str, image_captions: Dict[str, str]) -> str:
    """Replace each <image> placeholder with the matching original-image caption."""
    captions = iter(
        image_captions[key]
        for key in sorted(image_captions)
        if os.path.basename(key).startswith("original_image") and image_captions[key]
    )
    return re.sub(r"<image>", lambda _: f"[Image: {next(captions)}]", question)


def load_sample(traj_paths: List[str]) -> dict:
    with open(traj_paths[0], "r", encoding="utf-8") as f:
        first = json.loads(next(f))
    question = first["initial_prompt"]
    ground_truth = first["ground_truth"]
    prompt_path = Path(traj_paths[0]).parent / "injected_system_prompt.txt"
    system_prompt_text = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""

    return {
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt": system_prompt_text,
    }


def build_caption_contexts(
    all_images: Dict[str, Dict[str, Any]],
    question: Optional[str],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str], List[str]]:
    base_task_context: Dict[str, Any] = {}
    if question:
        base_task_context["question"] = question

    image_contexts: Dict[str, Dict[str, Any]] = {}
    image_filenames = []

    for caption_key, image_info in all_images.items():
        rollout_dir = image_info["rollout_dir"]

        image_contexts[caption_key] = {
            "task_context": base_task_context.copy(),
        }
        image_filenames.append(caption_key)

    return image_contexts, image_filenames


def caption_images(
    all_images: Dict[str, Dict[str, Any]],
    resolved_sample_dir: str,
    question: Optional[str],
    llm: ExperienceLLM,
) -> Dict[str, str]:
    print(f"  Found {len(all_images)} images in directory")
    image_contexts, image_filenames = build_caption_contexts(all_images, question)

    print(f"  Generating analysis for {len(image_filenames)} images...")
    analysis_start = time.time()
    image_captions = generate_image_captions(
        image_filenames,
        resolved_sample_dir,
        llm=llm,
        cache_dir=os.path.join(resolved_sample_dir, ".analysis_cache"),
        image_contexts=image_contexts,
    )
    print(f"  [Timing] Image analysis generation: {time.time() - analysis_start:.1f}s")
    return image_captions


def split_captions_by_rollout(image_captions: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    top_level_captions: Dict[str, str] = {}
    rollout_captions: Dict[str, Dict[str, str]] = {}
    for caption_key, caption_text in image_captions.items():
        if "/" in caption_key:
            rollout_dir, filename = caption_key.split("/", 1)
            rollout_captions.setdefault(rollout_dir, {})[caption_key] = caption_text
            rollout_captions[rollout_dir][filename] = caption_text
        else:
            top_level_captions[caption_key] = caption_text
    return top_level_captions, rollout_captions


def format_rollout_blocks(
    paths: List[str],
    image_captions: Dict[str, str],
) -> List[str]:
    top_level_captions, rollout_captions_map = split_captions_by_rollout(image_captions)
    formatted_blocks = []

    for rollout_idx, path in enumerate(paths):
        rollout_dir = Path(path).parent.name
        rollout_captions = {**top_level_captions, **rollout_captions_map.get(rollout_dir, {})}
        with open(path, "r", encoding="utf-8") as f:
            raw_jsonl_content = f.read()

        processed_content = replace_image_refs_in_jsonl(raw_jsonl_content, rollout_captions)
        formatted_blocks.append(f"==== Rollout {rollout_idx} ({rollout_dir}) ====\n{processed_content}")

    return formatted_blocks


def prepend_sample_header(
    process_text: str,
    question: Any,
    ground_truth: Any,
    system_prompt_text: Any,
    image_captions: Dict[str, str],
) -> str:
    formatted_question = replace_image_tags_in_question(question.strip(), image_captions)
    header_parts = [f"==== Question: ====\n{formatted_question}"]
    if system_prompt_text.strip():
        header_parts.append(f"==== System prompt: ====\n{system_prompt_text}")
    header_parts.append(f"==== Ground truth (for reference): ====\n{ground_truth.strip()}")

    return "\n\n".join(header_parts) + "\n\n" + process_text


def load_question_images(all_images: Dict[str, Dict[str, Any]]) -> List[Path]:
    originals: Dict[str, Path] = {}
    for image_info in all_images.values():
        path = image_info["file_path"]
        if image_info["is_original"] and path.name not in originals:
            originals[path.name] = path
    return list(originals.values())


def generate_summary_with_retries(prompt: str, question_images: List[Path], llm: ExperienceLLM) -> str:
    llm_start = time.time()
    response_text = None
    config = settings()

    for attempt in range(config.trajectory_retries):
        try:
            if question_images:
                pil_images = [Image.open(image_path).convert("RGB") for image_path in question_images]
                response_text = llm.chat_with_image(
                    prompt=prompt,
                    image=pil_images,
                    max_tokens=config.trajectory_max_tokens,
                )
            else:
                response_text = llm.chat(prompt, max_tokens=config.trajectory_max_tokens)
            break
        except Exception as exc:
            if attempt == config.trajectory_retries - 1:
                raise
            print(f"  Warning: Summary generation failed (attempt {attempt + 1}/{config.trajectory_retries}): {exc}, retrying...")
            time.sleep(2 ** attempt)

    if response_text is None:
        raise RuntimeError("Failed to generate summary after all retries")

    print(f"  [Timing] exp_summary_resp generation: {time.time() - llm_start:.1f}s")
    return response_text


def summarize_rollouts(
    traj_paths: List[str],
    llm: ExperienceLLM,
    sample_dir: str,
) -> dict:
    """Summarize rollout trajectory for experience extraction."""
    uniq_paths = list(dict.fromkeys(traj_paths))

    sample = load_sample(uniq_paths)
    question = sample["question"]
    ground_truth = sample["ground_truth"]
    system_prompt_text = sample["system_prompt"]

    all_images = select_images_for_summary(scan_all_images(sample_dir))
    image_captions = caption_images(all_images, sample_dir, question, llm)

    formatted_blocks = format_rollout_blocks(uniq_paths, image_captions)
    process_text = "\n\n".join(formatted_blocks)
    process_text = prepend_sample_header(
        process_text,
        question,
        ground_truth,
        system_prompt_text,
        image_captions,
    )

    prompt = SINGLE_ROLLOUT_SUMMARY.format(trajectory=process_text)
    response_text = generate_summary_with_retries(
        prompt,
        load_question_images(all_images),
        llm,
    )

    write_debug_texts(
        sample_dir,
        {
            "exp_summary_prompt.txt": prompt,
            "exp_summary_resp.txt": response_text,
        },
    )

    return {
        "process": response_text,
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt": system_prompt_text,
    }
