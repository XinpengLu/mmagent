import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from prompts.experience_retrieval_templates import IMAGE_DESCRIPTION_PROMPT
from utils.env import env_int
from .llm_client import ExperienceLLM


# Constants
MAX_TOKENS = env_int("EXPERIENCE_IMAGE_CAPTION_MAX_TOKENS", 512)

MAX_WORKERS = env_int("EXPERIENCE_IMAGE_CAPTION_WORKERS", 4)


CaptionTarget = Dict[str, Any]


def _caption_cache_path(
    sample_path: Path, rollout_dir: str, filename: str, cache_dir: Optional[str]
) -> Optional[Path]:
    if rollout_dir:
        return sample_path / rollout_dir / ".analysis_cache" / f"{Path(filename).stem}_analysis.txt"
    if cache_dir:
        return Path(cache_dir) / f"{Path(filename).stem}_analysis.txt"
    return None


def _caption_candidates(raw_filename: str) -> List[str]:
    clean_filename = os.path.basename(raw_filename)
    if clean_filename.endswith(('.jpg', '.jpeg', '.png')):
        return [clean_filename]
    return [f"{clean_filename}.jpg", clean_filename]


def _rollout_search_dirs(key: str, rollout_hint: str, image_to_rollout: Optional[Dict[str, str]]) -> List[str]:
    rollout_dirs = []
    if rollout_hint:
        rollout_dirs.append(rollout_hint)
    if image_to_rollout and key in image_to_rollout:
        rollout_dirs.append(image_to_rollout[key])
    return rollout_dirs


def _find_image_in_rollout_dirs(
    sample_path: Path,
    candidate: str,
    rollout_dirs: List[str],
) -> Optional[CaptionTarget]:
    for rollout_dir in rollout_dirs:
        if not rollout_dir:
            continue
        image_path = sample_path / rollout_dir / candidate
        if image_path.exists() and image_path.is_file():
            return {
                "caption_key": f"{rollout_dir}/{candidate}",
                "rollout_dir": rollout_dir,
                "filename": candidate,
                "image_path": image_path,
            }
    return None


def _find_image_in_sample(sample_path: Path, candidate: str) -> Optional[CaptionTarget]:
    top_path = sample_path / candidate
    if top_path.exists() and top_path.is_file():
        return {
            "caption_key": candidate,
            "rollout_dir": "",
            "filename": candidate,
            "image_path": top_path,
        }

    for item in sample_path.iterdir():
        if item.is_dir() and item.name.startswith("rollout_"):
            rollout_image_path = item / candidate
            if rollout_image_path.exists() and rollout_image_path.is_file():
                return {
                    "caption_key": f"{item.name}/{candidate}",
                    "rollout_dir": item.name,
                    "filename": candidate,
                    "image_path": rollout_image_path,
                }
    return None


def _find_caption_target(
    key: str,
    sample_path: Path,
    image_to_rollout: Optional[Dict[str, str]],
) -> Optional[CaptionTarget]:
    if "/" in key:
        rollout_hint, raw_filename = key.split("/", 1)
    else:
        rollout_hint, raw_filename = "", key

    rollout_dirs = _rollout_search_dirs(key, rollout_hint, image_to_rollout)
    for candidate in _caption_candidates(raw_filename):
        target = _find_image_in_rollout_dirs(sample_path, candidate, rollout_dirs)
        if target:
            return target
        target = _find_image_in_sample(sample_path, candidate)
        if target:
            return target
    return None


def _existing_caption_targets(
    image_filenames: List[str],
    sample_path: Path,
    image_to_rollout: Optional[Dict[str, str]],
) -> List[CaptionTarget]:
    targets = []
    for key in image_filenames:
        target = _find_caption_target(key, sample_path, image_to_rollout)
        if target:
            targets.append(target)
        else:
            clean_filename = os.path.basename(key.split("/", 1)[-1])
            print(f"    Warning: Image file not found: {clean_filename} in {sample_path} (checked top-level and rollout_* subdirectories)")
    return targets


def _caption_context(caption_key: str, image_contexts: Optional[Dict[str, Dict[str, Any]]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if image_contexts and caption_key in image_contexts:
        context = image_contexts[caption_key]
        return context.get("task_context"), context.get("generation_context")
    return None, None


def _cached_caption(
    sample_path: Path,
    target: CaptionTarget,
    cache_dir: Optional[str],
) -> Optional[str]:
    cache_path = _caption_cache_path(sample_path, target["rollout_dir"], target["filename"], cache_dir)
    if not cache_path or not cache_path.exists():
        return None
    try:
        cached_caption = cache_path.read_text(encoding="utf-8").strip()
        return cached_caption or None
    except Exception:
        return None


def _pending_caption_jobs(
    targets: List[CaptionTarget],
    sample_path: Path,
    cache_dir: Optional[str],
    image_contexts: Optional[Dict[str, Dict[str, Any]]],
) -> Tuple[Dict[str, str], List[CaptionTarget]]:
    captions = {}
    to_generate = []
    for target in targets:
        caption_key = target["caption_key"]
        image_path = target["image_path"]
        filename = target["filename"]
        if not image_path.exists() or not image_path.is_file():
            print(f"  Warning: Image file disappeared: {filename}")
            captions[caption_key] = f"[Image file not found: {filename}]"
            continue

        cached_caption = _cached_caption(sample_path, target, cache_dir)
        if cached_caption:
            captions[caption_key] = cached_caption
            continue

        task_context, generation_context = _caption_context(caption_key, image_contexts)
        job = dict(target)
        job["task_context"] = task_context
        job["generation_context"] = generation_context
        to_generate.append(job)
    return captions, to_generate


def _save_caption_cache(
    sample_path: Path,
    target: CaptionTarget,
    caption: str,
    cache_dir: Optional[str],
) -> None:
    if not caption or caption.startswith("[Error") or caption.startswith("[Failed"):
        return
    try:
        cache_path = _caption_cache_path(sample_path, target["rollout_dir"], target["filename"], cache_dir)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(caption, encoding="utf-8")
    except Exception as e:
        print(f"  Warning: Failed to save caption cache for {target['caption_key']}: {e}")


def _generate_caption_job(
    target: CaptionTarget,
    llm: ExperienceLLM,
    sample_path: Path,
    cache_dir: Optional[str],
) -> Tuple[str, str]:
    task_context = target.get("task_context")
    original_image_paths = task_context.get("original_image_paths") if task_context else None
    caption = _generate_single_image_caption(
        target["image_path"],
        llm,
        task_context=task_context,
        generation_context=target.get("generation_context"),
        original_image_paths=original_image_paths,
    )
    _save_caption_cache(sample_path, target, caption, cache_dir)
    return target["caption_key"], caption


def _generate_missing_captions(
    to_generate: List[CaptionTarget],
    llm: ExperienceLLM,
    sample_path: Path,
    cache_dir: Optional[str],
) -> Dict[str, str]:
    if not to_generate:
        return {}

    captions = {}
    max_workers = min(len(to_generate), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(_generate_caption_job, target, llm, sample_path, cache_dir): target["caption_key"]
            for target in to_generate
        }

        for future in as_completed(future_to_key):
            try:
                caption_key, caption = future.result()
                captions[caption_key] = caption
            except Exception as e:
                caption_key = future_to_key[future]
                print(f"  Warning: Failed to generate caption for {caption_key}: {e}")
                captions[caption_key] = f"[Error generating caption: {str(e)}]"
    return captions


def _generate_single_image_caption(
    image_path: Path, 
    llm: ExperienceLLM,
    prompt: Optional[str] = None,
    task_context: Optional[Dict[str, Any]] = None,
    generation_context: Optional[str] = None,
    original_image_paths: Optional[List[Path]] = None
) -> str:
    """Generate one contextual image caption for trajectory summarization."""
    if not image_path.exists():
        return f"[Image file not found: {image_path.name}]"
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        if prompt is None:
            prompt_parts = []
            
            if task_context:
                context_info = []
                if task_context.get("system_prompt"):
                    context_info.append(f"Task: {task_context['system_prompt']}")
                if task_context.get("question"):
                    context_info.append(f"Question: {task_context['question']}")
                
                if context_info:
                    prompt_parts.append("Background context (to help you understand the image's purpose and origin): " + " | ".join(context_info))
            
            if generation_context:
                prompt_parts.append(f"Image generation context (how this image was created): {generation_context}")
            
            if original_image_paths:
                prompt_parts.append("Note: The original question image(s) are provided below for reference. Please describe the current image (the tool-generated image) in relation to the question context.")
            
            prompt_parts.append(IMAGE_DESCRIPTION_PROMPT)
            
            prompt = "\n".join(prompt_parts)
        
        images_to_send = [image]
        if original_image_paths:
            for orig_path in original_image_paths:
                if orig_path.exists():
                    try:
                        orig_img = Image.open(orig_path).convert('RGB')
                        images_to_send.append(orig_img)
                    except Exception as e:
                        print(f"  Warning: Failed to load original image {orig_path}: {e}")
        
        return llm.chat_with_image(
            prompt=prompt,
            image=images_to_send,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            return_placeholder_on_error=True
        )
    except Exception as e:
        return f"[Error generating caption: {str(e)}]"


def generate_image_captions(
    image_filenames: List[str],
    sample_dir: str,
    llm: Optional[ExperienceLLM] = None,
    cache_dir: Optional[str] = None,
    image_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    image_to_rollout: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Generate captions for a list of image files using ExperienceLLM with context.
    Only processes files that actually exist in the sample directory.
   
    Returns:
        Dictionary mapping "{rollout_dir}/{filename}" to caption text for multi-rollout images,
        or just "{filename}" for single rollout or top-level images
    """
    if not image_filenames:
        return {}

    if llm is None:
        llm = ExperienceLLM()

    sample_path = Path(sample_dir)
    targets = _existing_caption_targets(image_filenames, sample_path, image_to_rollout)
    if not targets:
        return {}

    captions, to_generate = _pending_caption_jobs(targets, sample_path, cache_dir, image_contexts)
    captions.update(_generate_missing_captions(to_generate, llm, sample_path, cache_dir))
    return captions
