import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from prompts.experience_retrieval_templates import IMAGE_DESCRIPTION_PROMPT
from .config import ExperienceGenerationSettings
from .experience_llm import ExperienceLLM


def settings() -> ExperienceGenerationSettings:
    return ExperienceGenerationSettings.from_runtime()


CaptionTarget = Dict[str, Any]


def caption_cache_path(
    sample_path: Path, rollout_dir: str, filename: str, cache_dir: Optional[str]
) -> Optional[Path]:
    if rollout_dir:
        return sample_path / rollout_dir / ".analysis_cache" / f"{Path(filename).stem}_analysis.txt"
    if cache_dir:
        return Path(cache_dir) / f"{Path(filename).stem}_analysis.txt"
    return None


def find_caption_target(key: str, sample_path: Path) -> Optional[CaptionTarget]:
    rollout_dir, filename = key.split("/", 1)
    image_path = sample_path / rollout_dir / filename
    if image_path.exists() and image_path.is_file():
        return {
            "caption_key": key,
            "rollout_dir": rollout_dir,
            "filename": filename,
            "image_path": image_path,
        }
    return None


def existing_caption_targets(image_filenames: List[str], sample_path: Path) -> List[CaptionTarget]:
    targets = []
    for key in image_filenames:
        target = find_caption_target(key, sample_path)
        if target:
            targets.append(target)
        else:
            clean_filename = os.path.basename(key.split("/", 1)[-1])
            print(f"    Warning: Image file not found: {clean_filename} in {sample_path} (checked top-level and rollout_* subdirectories)")
    return targets


def load_cached_caption(
    sample_path: Path,
    target: CaptionTarget,
    cache_dir: Optional[str],
) -> Optional[str]:
    cache_path = caption_cache_path(sample_path, target["rollout_dir"], target["filename"], cache_dir)
    if not cache_path or not cache_path.exists():
        return None
    caption = cache_path.read_text(encoding="utf-8").strip()
    return caption or None


def pending_caption_jobs(
    targets: List[CaptionTarget],
    sample_path: Path,
    cache_dir: Optional[str],
    image_contexts: Dict[str, Dict[str, Any]],
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

        cached = load_cached_caption(sample_path, target, cache_dir)
        if cached:
            captions[caption_key] = cached
            continue

        context = image_contexts[caption_key]
        job = dict(target)
        job["task_context"] = context["task_context"]
        to_generate.append(job)
    return captions, to_generate


def save_caption_cache(
    sample_path: Path,
    target: CaptionTarget,
    caption: str,
    cache_dir: Optional[str],
) -> None:
    if not caption or caption.startswith("[Error"):
        return
    try:
        cache_path = caption_cache_path(sample_path, target["rollout_dir"], target["filename"], cache_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(caption, encoding="utf-8")
    except Exception as e:
        print(f"  Warning: Failed to save caption cache for {target['caption_key']}: {e}")


def generate_single_image_caption(
    image_path: Path, 
    llm: ExperienceLLM,
    task_context: Dict[str, Any],
) -> str:
    """Generate one contextual image caption for trajectory summarization."""
    try:
        image = Image.open(image_path).convert('RGB')
        prompt = (
            f"==== Question ====\n{task_context['question']}\n\n"
            f"==== Image Captioning Instruction ====\n{IMAGE_DESCRIPTION_PROMPT}"
        )
        return llm.chat_with_image(
            prompt=prompt,
            image=[image],
            max_tokens=settings().caption_max_tokens,
            temperature=settings().caption_temperature,
        )
    except Exception as e:
        return f"[Error generating caption: {str(e)}]"


def generate_caption_job(
    target: CaptionTarget,
    llm: ExperienceLLM,
    sample_path: Path,
    cache_dir: Optional[str],
) -> Tuple[str, str]:
    caption = generate_single_image_caption(
        target["image_path"],
        llm,
        task_context=target["task_context"],
    )
    save_caption_cache(sample_path, target, caption, cache_dir)
    return target["caption_key"], caption


def generate_missing_captions(
    to_generate: List[CaptionTarget],
    llm: ExperienceLLM,
    sample_path: Path,
    cache_dir: Optional[str],
) -> Dict[str, str]:
    if not to_generate:
        return {}

    captions = {}
    max_workers = min(len(to_generate), settings().caption_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(generate_caption_job, target, llm, sample_path, cache_dir): target["caption_key"]
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


def generate_image_captions(
    image_filenames: List[str],
    sample_dir: str,
    *,
    image_contexts: Dict[str, Dict[str, Any]],
    llm: ExperienceLLM,
    cache_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate captions for rollout images using ExperienceLLM with task context. Returns a mapping from "{rollout_dir}/{filename}" to caption text.
    """
    sample_path = Path(sample_dir)
    targets = existing_caption_targets(image_filenames, sample_path)
    captions, to_generate = pending_caption_jobs(targets, sample_path, cache_dir, image_contexts)
    captions.update(generate_missing_captions(to_generate, llm, sample_path, cache_dir))
    return captions
