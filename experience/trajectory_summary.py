"""Summarize rollout trajectories into text used for experience generation."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from prompts.experience_generation_templates import SINGLE_ROLLOUT_SUMMARY

from .llm_client import ExperienceLLM
from .image_captioning import generate_image_captions
from .config import ExperienceGenerationSettings
from .utils import _write_debug_texts


SETTINGS = ExperienceGenerationSettings.from_env()
MAX_RETRIES = SETTINGS.trajectory_retries
MAX_TOKENS = SETTINGS.trajectory_max_tokens
MAX_INITIAL_PROMPT_CHARS = SETTINGS.trajectory_max_initial_prompt_chars
MAX_TEXT_OUTPUT_CHARS = SETTINGS.trajectory_max_text_output_chars
MAX_TOOL_CODE_CHARS = SETTINGS.trajectory_max_tool_code_chars
MAX_TOOL_RESULT_CHARS = SETTINGS.trajectory_max_tool_result_chars
MAX_OTHER_FIELD_CHARS = SETTINGS.trajectory_max_other_field_chars
IMAGE_BUDGET = SETTINGS.trajectory_image_budget
TOOL_IMAGES_PER_ROLLOUT = SETTINGS.trajectory_tool_images_per_rollout
CAPTION_INCLUDE_SYSTEM_PROMPT = SETTINGS.caption_include_system_prompt
CAPTION_TOOL_WITH_ORIGINAL = SETTINGS.caption_tool_with_original
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _normalize_traj_paths(traj_paths: Union[str, List[str]]) -> List[str]:
    if isinstance(traj_paths, str):
        return [traj_paths]
    return list(traj_paths) if traj_paths else []


def _scan_all_images(sample_dir: str) -> Dict[str, Dict[str, Any]]:
    """Return top-level and rollout image files keyed by caption lookup name."""
    sample_path = Path(sample_dir)
    if not sample_path.exists() or not sample_path.is_dir():
        return {}

    image_map: Dict[str, Dict[str, Any]] = {}
    for image_path in sample_path.glob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            image_map[image_path.name] = {
                "file_path": image_path,
                "rollout_dir": "",
                "is_original": image_path.name.startswith("original_image"),
            }

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


def _image_sort_key(key: str) -> tuple:
    base = os.path.basename(key)
    rollout = key.split("/", 1)[0] if "/" in key else ""
    match = re.search(r"(\d+)", base)
    number = int(match.group(1)) if match else -1
    return rollout, base.startswith("tool_image"), number, base


def _select_images_for_summary(all_images: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Keep original images plus a compact first/last sample of tool images."""
    if not all_images or IMAGE_BUDGET <= 0 or len(all_images) <= IMAGE_BUDGET:
        return all_images

    selected: Dict[str, Dict[str, Any]] = {}
    originals = [(key, info) for key, info in all_images.items() if info.get("is_original")]
    for key, info in sorted(originals, key=lambda item: _image_sort_key(item[0])):
        if len(selected) < IMAGE_BUDGET:
            selected[key] = info

    by_rollout: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for key, info in all_images.items():
        if key in selected or info.get("is_original"):
            continue
        by_rollout.setdefault(str(info.get("rollout_dir") or ""), []).append((key, info))

    per_rollout = max(1, TOOL_IMAGES_PER_ROLLOUT)
    for rollout in sorted(by_rollout):
        items = sorted(by_rollout[rollout], key=lambda item: _image_sort_key(item[0]))
        chosen = items if len(items) <= per_rollout else items[:1] + items[-max(0, per_rollout - 1):]
        for key, info in chosen:
            if len(selected) < IMAGE_BUDGET:
                selected[key] = info

    if len(selected) < IMAGE_BUDGET:
        for key, info in sorted(all_images.items(), key=lambda item: _image_sort_key(item[0])):
            if key not in selected:
                selected[key] = info
            if len(selected) >= IMAGE_BUDGET:
                break

    dropped = len(all_images) - len(selected)
    if dropped > 0:
        print(
            f"  [Image Budget] Captioning {len(selected)}/{len(all_images)} images "
            f"(dropped {dropped}; budget={IMAGE_BUDGET}, per_rollout={per_rollout})"
        )
    return selected


def _find_original_images_for_tool_image(
    tool_image_rollout: str,
    all_images: Dict[str, Dict[str, Any]],
) -> List[str]:
    same_rollout = []
    top_level = []
    for key, info in all_images.items():
        if not info["is_original"]:
            continue
        if info["rollout_dir"] == tool_image_rollout:
            same_rollout.append(key)
        elif info["rollout_dir"] == "":
            top_level.append(key)
    return same_rollout + [key for key in top_level if key not in same_rollout]


def _get_caption_for_image(name_or_path: str, image_captions: Dict[str, str]) -> Optional[str]:
    """Find a caption by exact key, basename, no-extension key, or partial path."""
    if not name_or_path or not image_captions:
        return None
    if name_or_path in image_captions:
        return image_captions[name_or_path]

    basename = os.path.basename(name_or_path)
    if basename in image_captions:
        return image_captions[basename]

    name_no_ext = os.path.splitext(name_or_path)[0]
    for key, caption in image_captions.items():
        if os.path.splitext(key)[0] == name_no_ext:
            return caption
    for key, caption in image_captions.items():
        if name_or_path in key or os.path.basename(key) == name_or_path:
            return caption
    return None


def _truncate_for_summary(text: Any, max_chars: int, label: str) -> Any:
    """Truncate oversized text fields only in the LLM summary prompt."""
    if not isinstance(text, str) or max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n...[truncated {omitted} chars from {label}; original length={len(text)}]...\n"
        + text[-tail:]
    )


def _compact_record_for_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return record

    if isinstance(record.get("initial_prompt"), str):
        record["initial_prompt"] = _truncate_for_summary(
            record["initial_prompt"], MAX_INITIAL_PROMPT_CHARS, "initial_prompt"
        )
    if isinstance(record.get("text_output"), str):
        record["text_output"] = _truncate_for_summary(
            record["text_output"], MAX_TEXT_OUTPUT_CHARS, "text_output"
        )

    tool_call = record.get("tool_call")
    if isinstance(tool_call, dict):
        params = tool_call.get("parameters")
        if isinstance(params, dict):
            for key, value in list(params.items()):
                if isinstance(value, str):
                    limit = MAX_TOOL_CODE_CHARS if key == "code" else MAX_OTHER_FIELD_CHARS
                    params[key] = _truncate_for_summary(value, limit, f"tool_call.parameters.{key}")
        if isinstance(tool_call.get("result"), str):
            tool_call["result"] = _truncate_for_summary(
                tool_call["result"], MAX_TOOL_RESULT_CHARS, "tool_call.result"
            )

    for key, value in list(record.items()):
        if key in {"initial_prompt", "text_output", "tool_call"}:
            continue
        if isinstance(value, str):
            record[key] = _truncate_for_summary(value, MAX_OTHER_FIELD_CHARS, key)
    return record


def _replace_image_refs_in_jsonl(jsonl_content: str, image_captions: Dict[str, str]) -> str:
    """Replace markdown image refs and tool_image entries with caption text."""
    if not jsonl_content:
        return jsonl_content

    image_captions = image_captions or {}
    image_pattern = r"!\[([^\]]+\.(?:jpg|jpeg|png))\]\(([^\)]+\.(?:jpg|jpeg|png))\)"
    lines = []

    for line in jsonl_content.split("\n"):
        if not line.strip():
            lines.append(line)
            continue

        try:
            record = json.loads(line)

            tool_call = record.get("tool_call")
            if isinstance(tool_call, dict) and tool_call.get("result"):
                result = str(tool_call["result"])

                def replace_with_caption(match: re.Match) -> str:
                    image_name = match.group(1) or match.group(2)
                    caption = _get_caption_for_image(image_name, image_captions)
                    return f"[Image: {image_name}]\nCaption: {caption}" if caption else match.group(0)

                tool_call["result"] = re.sub(image_pattern, replace_with_caption, result)

            tool_image = record.get("tool_image")
            if isinstance(tool_image, dict):
                image_name = tool_image.get("image_name", "")
                file_path = tool_image.get("file_path", "")
                caption = _get_caption_for_image(image_name, image_captions)
                if not caption and file_path:
                    caption = _get_caption_for_image(file_path, image_captions)
                if caption:
                    tool_image["caption"] = caption

            lines.append(json.dumps(_compact_record_for_summary(record), ensure_ascii=False))
        except Exception:
            lines.append(_truncate_for_summary(line, MAX_OTHER_FIELD_CHARS, "non_json_line"))

    return "\n".join(lines)


def _replace_image_tags_in_question(question: str, image_captions: Dict[str, str]) -> str:
    """Replace <image> placeholders with original-image captions when available."""
    if not question or not image_captions:
        return question

    matches = list(re.finditer(r"<image>", question))
    if not matches:
        return question

    original_captions = [
        image_captions[key]
        for key in sorted(image_captions)
        if os.path.basename(key).startswith("original_image") and image_captions[key]
    ]
    if not original_captions:
        return question

    result_parts = []
    last_end = 0
    for idx, match in enumerate(matches):
        result_parts.append(question[last_end:match.start()])
        caption = original_captions[idx] if idx < len(original_captions) else original_captions[-1]
        result_parts.append(f"[Image: {caption}]")
        last_end = match.end()
    result_parts.append(question[last_end:])
    return "".join(result_parts)


def _resolve_sample_dir(traj_paths: Union[str, List[str]], provided_sample_dir: Optional[str] = None) -> str:
    if provided_sample_dir and os.path.exists(provided_sample_dir):
        return provided_sample_dir
    paths = _normalize_traj_paths(traj_paths)
    if not paths:
        raise ValueError("No trajectory paths provided")

    first_dir = os.path.dirname(paths[0])
    return os.path.dirname(first_dir) if os.path.basename(first_dir).startswith("rollout_") else first_dir


def _load_metadata(traj_paths: Union[str, List[str]], sample_dir: str) -> dict:
    """Load question, ground truth, and injected system prompt for the sample."""
    paths = _normalize_traj_paths(traj_paths)
    question = None
    ground_truth = None

    if paths:
        try:
            with open(paths[0], "r", encoding="utf-8") as f:
                first = json.loads(next(f))
            question = first.get("initial_prompt")
            ground_truth = first.get("ground_truth")
        except Exception:
            pass

    system_prompt_text = None
    try:
        prompt_path = os.path.join(sample_dir, "injected_system_prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt_text = f.read().strip()
        else:
            for item in os.listdir(sample_dir):
                rollout_dir = os.path.join(sample_dir, item)
                prompt_path = os.path.join(rollout_dir, "injected_system_prompt.txt")
                if os.path.isdir(rollout_dir) and item.startswith("rollout_") and os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        system_prompt_text = f.read().strip()
                    break
    except Exception:
        pass

    return {
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt": system_prompt_text,
    }


def _build_caption_contexts(
    all_images: Dict[str, Dict[str, Any]],
    question: Optional[str],
    system_prompt_text: Optional[str],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str], List[str]]:
    base_task_context: Dict[str, Any] = {}
    if question:
        base_task_context["question"] = question
    if system_prompt_text and CAPTION_INCLUDE_SYSTEM_PROMPT:
        base_task_context["system_prompt"] = system_prompt_text

    image_contexts: Dict[str, Dict[str, Any]] = {}
    image_to_rollout: Dict[str, str] = {}
    image_filenames = []

    for caption_key, image_info in all_images.items():
        rollout_dir = image_info["rollout_dir"]
        task_context = base_task_context.copy()

        if CAPTION_TOOL_WITH_ORIGINAL and not image_info["is_original"]:
            original_keys = _find_original_images_for_tool_image(rollout_dir, all_images)
            original_paths = [
                all_images[original_key]["file_path"]
                for original_key in original_keys
                if original_key in all_images
            ]
            if original_paths:
                task_context["original_image_paths"] = original_paths

        image_contexts[caption_key] = {
            "task_context": task_context,
            "generation_context": None,
        }
        image_to_rollout[caption_key] = rollout_dir
        image_filenames.append(caption_key)

    return image_contexts, image_to_rollout, image_filenames


def _caption_images(
    all_images: Dict[str, Dict[str, Any]],
    resolved_sample_dir: str,
    question: Optional[str],
    system_prompt_text: Optional[str],
    llm: ExperienceLLM,
) -> Dict[str, str]:
    if not all_images:
        print(f"  Warning: No images found in {resolved_sample_dir}")
        return {}

    print(f"  Found {len(all_images)} images in directory")
    image_contexts, image_to_rollout, image_filenames = _build_caption_contexts(
        all_images,
        question,
        system_prompt_text,
    )

    print(f"  Generating analysis for {len(image_filenames)} images...")
    analysis_start = time.time()
    image_captions = generate_image_captions(
        image_filenames,
        resolved_sample_dir,
        llm=llm,
        cache_dir=os.path.join(resolved_sample_dir, ".analysis_cache"),
        image_contexts=image_contexts,
        image_to_rollout=image_to_rollout,
    )
    print(f"  [Timing] Image analysis generation: {time.time() - analysis_start:.1f}s")
    return image_captions


def _split_captions_by_rollout(image_captions: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
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


def _format_rollout_blocks(
    paths: List[str],
    image_captions: Dict[str, str],
    is_single: bool,
) -> List[str]:
    top_level_captions, rollout_captions_map = _split_captions_by_rollout(image_captions)
    formatted_blocks = []

    for rollout_idx, path in enumerate(paths):
        rollout_dir = Path(path).parent.name
        rollout_captions = {**top_level_captions, **rollout_captions_map.get(rollout_dir, {})}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_jsonl_content = f.read()
        except Exception as exc:
            print(f"  Warning: Failed to read trajectory file {path}: {exc}")
            continue

        if not raw_jsonl_content.strip():
            continue

        processed_content = _replace_image_refs_in_jsonl(raw_jsonl_content, rollout_captions)
        if is_single:
            formatted_blocks.append(processed_content)
        else:
            formatted_blocks.append(f"==== Rollout {rollout_idx} ({rollout_dir}) ====\n{processed_content}")

    return formatted_blocks


def _prepend_metadata_header(
    process_text: str,
    question: Any,
    ground_truth: Any,
    system_prompt_text: Any,
    image_captions: Dict[str, str],
    is_single: bool,
) -> str:
    header_prefix = "==== " if not is_single else ""
    header_parts = []
    if isinstance(question, str) and question.strip():
        formatted_question = _replace_image_tags_in_question(question.strip(), image_captions)
        header_parts.append(f"{header_prefix}Question:\n{formatted_question}")
    if isinstance(system_prompt_text, str) and system_prompt_text.strip():
        header_parts.append(f"{header_prefix}System prompt:\n{system_prompt_text}")
    if isinstance(ground_truth, str) and ground_truth.strip():
        header_parts.append(f"{header_prefix}Ground truth (for reference):\n{ground_truth.strip()}")

    return "\n\n".join(header_parts) + "\n\n" + process_text if header_parts else process_text


def _load_question_images(all_images: Dict[str, Dict[str, Any]]) -> List[Path]:
    return [
        image_info["file_path"]
        for image_info in all_images.values()
        if image_info["is_original"] and image_info["rollout_dir"] == ""
    ]


def _generate_summary_with_retries(
    prompt: str,
    question_images: List[Path],
    llm: ExperienceLLM,
) -> str:
    llm_start = time.time()
    response_text = None

    for attempt in range(MAX_RETRIES):
        try:
            if question_images:
                pil_images = [Image.open(image_path).convert("RGB") for image_path in question_images]
                response_text = llm.chat_with_image(
                    prompt=prompt,
                    image=pil_images,
                    max_tokens=MAX_TOKENS,
                    return_placeholder_on_error=False,
                )
            else:
                response_text = llm.chat(prompt, max_tokens=MAX_TOKENS)
            break
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  Warning: Summary generation failed (attempt {attempt + 1}/{MAX_RETRIES}): {exc}, retrying...")
            time.sleep(2 ** attempt)

    if response_text is None:
        raise RuntimeError("Failed to generate summary after all retries")

    print(f"  [Timing] exp_summary_resp generation: {time.time() - llm_start:.1f}s")
    return response_text


def summarize_rollouts(
    traj_paths: Union[str, List[str]],
    llm: ExperienceLLM,
    sample_dir: Optional[str] = None,
) -> dict:
    """Summarize one or more rollout trajectory files for experience extraction."""
    paths = _normalize_traj_paths(traj_paths)
    if not paths:
        return {}

    uniq_paths = list(dict.fromkeys(paths))
    is_single = len(uniq_paths) == 1

    try:
        resolved_sample_dir = _resolve_sample_dir(uniq_paths, sample_dir)
    except Exception as exc:
        print(f"  Warning: Failed to resolve sample_dir: {exc}")
        resolved_sample_dir = os.path.dirname(uniq_paths[0])

    metadata = _load_metadata(uniq_paths, resolved_sample_dir)
    question = metadata.get("question")
    ground_truth = metadata.get("ground_truth")
    system_prompt_text = metadata.get("system_prompt")

    all_images = _select_images_for_summary(_scan_all_images(resolved_sample_dir))
    image_captions = _caption_images(
        all_images,
        resolved_sample_dir,
        question,
        system_prompt_text,
        llm,
    )

    formatted_blocks = _format_rollout_blocks(uniq_paths, image_captions, is_single)
    if not formatted_blocks:
        return {}

    process_text = "\n\n".join(formatted_blocks)
    process_text = _prepend_metadata_header(
        process_text,
        question,
        ground_truth,
        system_prompt_text,
        image_captions,
        is_single,
    )

    prompt = SINGLE_ROLLOUT_SUMMARY.format(trajectory=process_text)
    response_text = _generate_summary_with_retries(
        prompt,
        _load_question_images(all_images),
        llm,
    )

    _write_debug_texts(
        resolved_sample_dir,
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
