import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from .llm_client import ExperienceLLM


# --------- Constants ---------

# 单张图片描述的输出 token 上限。
MAX_TOKENS = 2048

# 图片 caption 并发生成的最大线程数。
MAX_WORKERS = 8


# --------- Image Caption Generation ---------

# 图像描述的全局提示
IMAGE_DESCRIPTION_PROMPT = "Please provide a detailed visual description and analysis of this image. The background context above is provided to help you understand the image's purpose and origin, which should guide you to focus on relevant visual details. Describe key objects, their spatial relationships, notable visual details, and any observable actions or events. Highlight how this image contributes to the overall task. Format your response as a single paragraph (no headings, no bullet points, no problem-solving steps), written in clear English, with no more than 200 words."


def _caption_cache_path(
    sample_path: Path, rollout_dir: str, filename: str, cache_dir: Optional[str]
) -> Optional[Path]:
    """标题缓存文件的返回路径，若此情况未启用缓存则返回值为 None"""
    # rollout 内图片的缓存放在各自 rollout/.analysis_cache 下。
    if rollout_dir:
        return sample_path / rollout_dir / ".analysis_cache" / f"{Path(filename).stem}_analysis.txt"
    if cache_dir:
        # 顶层图片或显式 cache_dir 使用公共缓存目录。
        return Path(cache_dir) / f"{Path(filename).stem}_analysis.txt"
    return None


def _generate_single_image_caption(
    image_path: Path, 
    llm: ExperienceLLM,
    prompt: Optional[str] = None,
    task_context: Optional[Dict[str, Any]] = None,
    generation_context: Optional[str] = None,
    original_image_paths: Optional[List[Path]] = None
) -> str:
    """
    Generate a detailed caption for a single image using ExperienceLLM with context.
    使用 ExperienceLLM 并结合相关背景信息为单张图片生成详细的说明文字
    
    Args:
        image_path: Path to the image file
        llm: ExperienceLLM instance for API calls
        prompt: Optional custom prompt (default: detailed image description prompt)
        task_context: Optional dict with 'question' and 'system_prompt' keys
        generation_context: Optional text describing how the image was generated (previous reasoning/tool calls)
        original_image_paths: Optional list of Path objects to original question images (for tool-generated images)
        
    Returns:
        Caption text string (or error placeholder on failure)
    """
    if not image_path.exists():
        return f"[Image file not found: {image_path.name}]"
    
    try:
        # 统一转 RGB，避免透明通道或调色板图影响 API。
        image = Image.open(image_path).convert('RGB')
        
        # 根据题目、system prompt、图片生成上下文构造描述 prompt。
        if prompt is None:
            prompt_parts = []
            
            # 加入题目背景，让 caption 关注和解题有关的视觉细节。
            if task_context:
                context_info = []
                if task_context.get("system_prompt"):
                    context_info.append(f"Task: {task_context['system_prompt']}")
                if task_context.get("question"):
                    context_info.append(f"Question: {task_context['question']}")
                
                if context_info:
                    prompt_parts.append("Background context (to help you understand the image's purpose and origin): " + " | ".join(context_info))
            
            # 工具生成图像时，补充它是如何产生的。
            if generation_context:
                prompt_parts.append(f"Image generation context (how this image was created): {generation_context}")
            
            # 对工具图像附带原图，帮助模型比较变化。
            if original_image_paths:
                prompt_parts.append("Note: The original question image(s) are provided below for reference. Please describe the current image (the tool-generated image) in relation to the question context.")
            
            # 最后追加统一的图片描述格式要求。
            prompt_parts.append(IMAGE_DESCRIPTION_PROMPT)
            
            prompt = "\n".join(prompt_parts)
        
        # 多模态输入顺序：当前图片在前，原始题图作为参考在后。
        images_to_send = [image]
        
        if original_image_paths:
            # 加载原始题图作为上下文图片。
            for orig_path in original_image_paths:
                if orig_path.exists():
                    try:
                        orig_img = Image.open(orig_path).convert('RGB')
                        images_to_send.append(orig_img)
                    except Exception as e:
                        print(f"  Warning: Failed to load original image {orig_path}: {e}")
        
        # 调用 ExperienceLLM 的多模态接口生成 caption。
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
    
    Args:
        image_filenames: List of image filenames (e.g., ["tool_image_1.jpg", "search_image_2.jpg"])
        sample_dir: Directory containing the images
        llm: Optional ExperienceLLM instance (creates new one if not provided)
        cache_dir: Optional base directory for caching (actual cache will be in rollout subdirs)
        image_contexts: Optional dict mapping image filename to context dict with:
            - "task_context": dict with "question" and "system_prompt" keys
            - "generation_context": str describing how image was generated (previous reasoning/tool calls)
        image_to_rollout: Optional dict mapping image filename to rollout directory (e.g., "rollout_0")
        
    Returns:
        Dictionary mapping "{rollout_dir}/{filename}" to caption text for multi-rollout images,
        or just "{filename}" for single rollout or top-level images
    """
    if not image_filenames:
        return {}
    
    # 没有传入 llm 时创建默认 ExperienceLLM。
    if llm is None:
        llm = ExperienceLLM()
    
    sample_path = Path(sample_dir)
    # 解析图片路径：支持顶层图片和 rollout_X/filename 两种形式。
    existing_files = []  # List of tuples: (caption_key, rollout_dir, filename)
    for key in image_filenames:
        # key 可能是 "rollout_X/filename"，也可能只有文件名。
        if "/" in key:
            rollout_hint, raw_filename = key.split("/", 1)
        else:
            rollout_hint, raw_filename = "", key
        
        clean_filename = os.path.basename(raw_filename)
        
        # 兼容带扩展名和不带扩展名的引用。
        candidates = []
        if clean_filename.endswith(('.jpg', '.jpeg', '.png')):
            candidates.append(clean_filename)
        else:
            candidates.append(f"{clean_filename}.jpg")
            candidates.append(clean_filename)
        
        # 按 hint/mapping、顶层目录、rollout_* 子目录依次查找。
        found = False
        for candidate in candidates:
            # First check rollout hint or mapping
            rollout_dirs = []
            if rollout_hint:
                rollout_dirs.append(rollout_hint)
            if image_to_rollout and key in image_to_rollout:
                rollout_dirs.append(image_to_rollout[key])
            
            for rollout_dir in rollout_dirs:
                if rollout_dir:
                    # 优先查指定 rollout 目录。
                    rollout_path = sample_path / rollout_dir / candidate
                    if rollout_path.exists() and rollout_path.is_file():
                        caption_key = f"{rollout_dir}/{candidate}"
                        existing_files.append((caption_key, rollout_dir, candidate, rollout_path))
                        found = True
                        break
            if found:
                break

            # 再查样本顶层目录。
            top_path = sample_path / candidate
            if top_path.exists() and top_path.is_file():
                existing_files.append((candidate, "", candidate, top_path))
                found = True
                break

            # 最后扫所有 rollout_X 子目录。
            if not found:
                for item in sample_path.iterdir():
                    if item.is_dir() and item.name.startswith("rollout_"):
                        rollout_image_path = item / candidate
                        if rollout_image_path.exists() and rollout_image_path.is_file():
                            caption_key = f"{item.name}/{candidate}"
                            existing_files.append((caption_key, item.name, candidate, rollout_image_path))
                            found = True
                            break
                if found:
                    break
        
        if not found:
            # 缺失图片只警告，不中断后续 caption。
            print(f"    Warning: Image file not found: {clean_filename} in {sample_dir} (checked top-level and rollout_* subdirectories)")
    
    if not existing_files:
        return {}
    
    # 分离已缓存和需要新生成 caption 的图片。
    captions = {}
    to_generate = []

    for caption_key, rollout_dir, filename, image_path in existing_files:
        if not image_path.exists() or not image_path.is_file():
            print(f"  Warning: Image file disappeared: {filename}")
            captions[caption_key] = f"[Image file not found: {filename}]"
            continue

        cache_path = _caption_cache_path(sample_path, rollout_dir, filename, cache_dir)
        if cache_path and cache_path.exists():
            try:
                # 命中缓存时直接复用，减少多模态 API 调用。
                cached_caption = cache_path.read_text(encoding="utf-8").strip()
                if cached_caption:
                    captions[caption_key] = cached_caption
                    continue
            except Exception:
                pass

        task_context = None
        generation_context = None
        if image_contexts and caption_key in image_contexts:
            # 每张图片可以带不同的题目上下文和生成上下文。
            context = image_contexts[caption_key]
            task_context = context.get("task_context")
            generation_context = context.get("generation_context")

        to_generate.append((caption_key, image_path, task_context, generation_context, rollout_dir, filename))
    
    # 对未缓存图片并发生成 caption。
    if to_generate:
        def _generate_with_cache(args):
            """Helper function to generate caption and save to cache."""
            caption_key, image_path, task_context, generation_context, rollout_dir, filename = args
            original_image_paths = None
            if task_context and "original_image_paths" in task_context:
                original_image_paths = task_context["original_image_paths"]
            # 生成 caption，并在成功后写入缓存。
            caption = _generate_single_image_caption(
                image_path,
                llm,
                task_context=task_context,
                generation_context=generation_context,
                original_image_paths=original_image_paths
            )
            if caption and not caption.startswith("[Error") and not caption.startswith("[Failed"):
                try:
                    cache_path = _caption_cache_path(sample_path, rollout_dir, filename, cache_dir)
                    if cache_path:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(caption, encoding="utf-8")
                except Exception as e:
                    print(f"  Warning: Failed to save caption cache for {caption_key}: {e}")
            return caption_key, caption
        
        # 使用线程池并发调用图片理解 API。
        max_workers = min(len(to_generate), MAX_WORKERS)  # Limit concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有图片 caption 任务。
            future_to_key = {
                executor.submit(_generate_with_cache, args): args[0] 
                for args in to_generate
            }
            
            # 哪个任务先完成就先写入 captions。
            for future in as_completed(future_to_key):
                try:
                    caption_key, caption = future.result()
                    captions[caption_key] = caption
                except Exception as e:
                    caption_key = future_to_key[future]
                    print(f"  Warning: Failed to generate caption for {caption_key}: {e}")
                    captions[caption_key] = f"[Error generating caption: {str(e)}]"
    
    return captions


