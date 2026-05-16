import os
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
from PIL import Image

from prompts.experience_prompts import (
    SINGLE_ROLLOUT_SUMMARY,
)

from .llm_client import ExperienceLLM
from .multimodal_analysis import generate_image_captions


# --------- Constants ---------

# 摘要生成 LLM 调用失败时的重试次数。
MAX_RETRIES = 3

# 轨迹摘要的输出 token 上限。
MAX_TOKENS = 12288  # Max tokens for trajectory summarization


# --------- Summarization ---------
def _normalize_traj_paths(traj_paths: Union[str, List[str]]) -> List[str]:
    """将轨迹路径标准化为列表形式 (single path -> [path], list unchanged)."""
    # 后续统一按 list 处理单 rollout 和多 rollout。
    if isinstance(traj_paths, str):
        return [traj_paths]
    return list(traj_paths) if traj_paths else []


def _scan_all_images(
    sample_dir: str
) -> Dict[str, Dict[str, Any]]:
    """Scan all image files in the sample directory and identify original_image files.
    扫描样本目录中的所有图像文件，并识别出‘original_image’的文件

    Args:
        sample_dir: Path to the sample directory
        样本目录：样本目录的路径
        
    Returns:
        Dictionary mapping caption_key (e.g., "rollout_0/tool_image_1.jpg" or "original_image.jpg") 
        to dict with:
            - "file_path": Path object to the image file
            - "rollout_dir": Rollout directory name (e.g., "rollout_0") or empty string for top-level
            - "is_original": True if this is an original_image* file, False otherwise
    
        返回一个 image_map，记录每张图片：
        文件路径 file_path
        属于哪个 rollout 目录 rollout_dir
        是否是原始题图 is_original
    """
    sample_path = Path(sample_dir)
    if not sample_path.exists() or not sample_path.is_dir():
        return {}
    
    image_map = {}
    
    # 扫描样本顶层图片，通常包含 original_image*。
    for img_file in sample_path.glob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            filename = img_file.name
            is_original = filename.startswith('original_image')
            image_map[filename] = {
                "file_path": img_file,
                "rollout_dir": "",
                "is_original": is_original
            }
    
    # 扫描各 rollout_X 子目录中的工具图像和原图副本。
    for item in sample_path.iterdir():
        if item.is_dir() and item.name.startswith("rollout_"):
            rollout_dir = item.name
            for img_file in item.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    filename = img_file.name
                    is_original = filename.startswith('original_image')
                    caption_key = f"{rollout_dir}/{filename}"
                    image_map[caption_key] = {
                        "file_path": img_file,
                        "rollout_dir": rollout_dir,
                        "is_original": is_original
                    }
    
    return image_map


def _find_original_images_for_tool_image(
    tool_image_key: str,
    tool_image_rollout: str,
    all_images: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Find original_image files that should be included in task_context for a tool-generated image.
    
    For a tool-generated image in rollout_X, we look for original_image* files in:
    1. The same rollout directory (rollout_X/original_image*.jpg)
    2. The top-level directory (original_image*.jpg)

    找出应包含在工具生成图像的任务上下文中之中的原始图像文件。
    对于在“rollout_X”文件夹中生成的工具生成的图像，我们会在以下位置查找“original_image*”文件：
    1. 相同的部署目录（rollout_X/original_image*.jpg）
    2. 顶级目录（original_image*.jpg）
    
    Args:
        tool_image_key: Caption key for the tool image (e.g., "rollout_0/tool_image_1.jpg")
        tool_image_rollout: Rollout directory for the tool image (e.g., "rollout_0")
        all_images: Dictionary from _scan_all_images
        
    Returns:
        List of caption keys for original_image files to include
    """
    original_image_keys = []
    
    # 工具图像优先关联同一个 rollout 下的 original_image。
    if tool_image_rollout:
        for key, info in all_images.items():
            if info["is_original"] and info["rollout_dir"] == tool_image_rollout:
                original_image_keys.append(key)
    
    # 顶层 original_image 作为通用参考图。
    for key, info in all_images.items():
        if info["is_original"] and info["rollout_dir"] == "":
            if key not in original_image_keys:
                original_image_keys.append(key)
    
    same_rollout = []
    top_level = []
    for key, info in all_images.items():
        if not info["is_original"]:
            continue
        if info["rollout_dir"] == tool_image_rollout:
            same_rollout.append(key)
        elif info["rollout_dir"] == "" and key not in same_rollout:
            top_level.append(key)
    return same_rollout + top_level


def _get_caption_for_image(name_or_path: str, image_captions: Dict[str, str]) -> Optional[str]:
    """Find caption for an image by name or path (exact, basename, no-ext, partial match)."""
    # 轨迹里引用图片的方式不统一，这里做多种匹配。
    # 根据图片名或路径，在 image_captions 中找到对应 caption
    if not name_or_path or not image_captions:
        return None
    if name_or_path in image_captions:
        return image_captions[name_or_path]
    basename = os.path.basename(name_or_path)
    if basename in image_captions:
        return image_captions[basename]
    name_no_ext = os.path.splitext(name_or_path)[0]
    for key in image_captions:
        if os.path.splitext(key)[0] == name_no_ext:
            return image_captions[key]
    for key in image_captions:
        if name_or_path in key or os.path.basename(key) == name_or_path:
            return image_captions[key]
    return None


def _replace_image_refs_in_jsonl(
    jsonl_content: str, 
    image_captions: Dict[str, str]
) -> str:
    """Replace image references in raw JSONL content with captions.
    将原始 JSONL 内容中的图像引用替换为相应的描述文字
    This function processes the raw JSONL file and:
    1. Replaces markdown image references in tool_call.result with captions
    2. Adds caption field to tool_image entries
    此功能会处理原始的 JSONL 文件：
    1. 将工具调用结果中的 Markdown 图像引用替换为相应的说明文字
    2. 为“tool_image”条目添加“标题”字段
    
    Args:
        jsonl_content: Raw JSONL file content as string
        image_captions: Dictionary mapping image filename to caption text
        
    Returns:
        Processed JSONL content with image references replaced
    """
    if not jsonl_content or not image_captions:
        return jsonl_content
    
    lines = []
    for line in jsonl_content.split('\n'):
        if not line.strip():
            lines.append(line)
            continue
        
        try:
            rec = json.loads(line)
            modified = False
            
            # 将工具结果中的 markdown 图片链接替换成文字 caption。
            if 'tool_call' in rec:
                tool_call_data = rec['tool_call']
                if 'result' in tool_call_data and tool_call_data['result']:
                    result = str(tool_call_data['result'])
                    # Pattern to match markdown image references: ![image_name.jpg](image_name.jpg)
                    image_pattern = r'!\[([^\]]+\.(?:jpg|jpeg|png))\]\(([^\)]+\.(?:jpg|jpeg|png))\)'
                    
                    def replace_with_caption(match):
                        # 找到对应 caption 时，用 caption 替换图片引用。
                        img_name = match.group(1) or match.group(2)
                        caption = _get_caption_for_image(img_name, image_captions)
                        if caption:
                            return f"[Image: {img_name}]\nCaption: {caption}"
                        return match.group(0)
                    
                    new_result = re.sub(image_pattern, replace_with_caption, result)
                    if new_result != result:
                        tool_call_data['result'] = new_result
                        modified = True
            
            # 对单独的 tool_image 记录补充 caption 字段。
            if 'tool_image' in rec:
                tool_image_data = rec['tool_image']
                image_name = tool_image_data.get('image_name', '')
                file_path = tool_image_data.get('file_path', '')
                caption = _get_caption_for_image(image_name, image_captions) or (_get_caption_for_image(file_path, image_captions) if file_path else None)
                if caption:
                    tool_image_data['caption'] = caption
                    modified = True
            
            # 只有修改过的 JSONL 行才重新序列化。
            if modified:
                lines.append(json.dumps(rec, ensure_ascii=False))
            else:
                lines.append(line)
        except (json.JSONDecodeError, Exception):
            # 非 JSON 行保持原样，避免破坏轨迹文件。
            lines.append(line)
    
    return '\n'.join(lines)


def _replace_image_tags_in_question(question: str, image_captions: Dict[str, str]) -> str:
    """Replace <image> tags in question with corresponding captions.
    把题面中的 <image> 标签替换成原始图片 caption
    这样最终的轨迹摘要 prompt 里，题图也有文字描述。

    Args:
        question: Question text that may contain <image> tags
        image_captions: Dictionary mapping image filename to caption text
    
    Returns:
        Question text with <image> tags replaced by captions
    """
    if not question or not image_captions:
        return question
    
    # 找到题面中的所有 <image> 占位符。
    image_tag_pattern = r'<image>'
    matches = list(re.finditer(image_tag_pattern, question))
    
    if not matches:
        return question
    
    # 按文件名排序收集 original_image 的 caption，保证替换顺序稳定
    original_captions = []
    for caption_key in sorted(image_captions.keys()):
        # Extract basename to handle rollout-aware keys (e.g., "rollout_0/original_image.jpg")
        basename = os.path.basename(caption_key)
        if basename.startswith('original_image'):
            caption = image_captions[caption_key]
            if caption:
                original_captions.append(caption)
    
    if not original_captions:
        return question
    
    # 逐个替换 <image> 标签，图片多于标签或标签多于图片时做兜底。
    result_parts = []
    last_end = 0
    
    for idx, match in enumerate(matches):
        # 保留当前 <image> 前面的原始文本。
        result_parts.append(question[last_end:match.start()])
        
        # 用对应图片 caption 替换 <image>。
        if idx < len(original_captions):
            result_parts.append(f"[Image: {original_captions[idx]}]")
        else:
            result_parts.append(f"[Image: {original_captions[-1]}]")
        
        last_end = match.end()
    
    # 拼回最后一段文本。
    result_parts.append(question[last_end:])
    
    return ''.join(result_parts)


def _resolve_sample_dir(traj_paths: Union[str, List[str]], provided_sample_dir: Optional[str] = None) -> str:
    """Resolve the top-level sample directory from trajectory paths.
    根据轨迹路径解析出顶级样本目录

    Priority:
    1. If provided_sample_dir is given, use it
    2. If single path: check if in rollout_X/, go up to top level
    3. If multiple paths: infer from first path
    优先级：
    1. 如果提供了 sample_dir 参数，则使用该参数。
    2. 如果是单路径：检查是否在 rollout_X/ 目录下，然后升至顶层。
    3. 如果有多种路径：则依据第一条路径进行推断
    
    Args:
        traj_paths: Single path (str) or list of paths 单条路径（字符串）或路径列表
        provided_sample_dir: Explicitly provided sample directory 明确指定的样本目录
    
    Returns:
        Top-level sample directory path 顶级样本目录路径
    """
    if provided_sample_dir and os.path.exists(provided_sample_dir):
        return provided_sample_dir
    paths = _normalize_traj_paths(traj_paths)
    if not paths:
        raise ValueError("No trajectory paths provided")
    
    first_path = paths[0]
    first_dir = os.path.dirname(first_path)
    
    # traj.jsonl 可能在 rollout_X/ 下，需要回到样本顶层。
    dir_name = os.path.basename(first_dir)
    if dir_name.startswith("rollout_"):
        # Go up one level to get sample_dir
        sample_dir = os.path.dirname(first_dir)
    else:
        # Already at top level
        sample_dir = first_dir
    
    return sample_dir


def _load_metadata(traj_paths: Union[str, List[str]], sample_dir: str) -> dict:
    """Load metadata (question, ground_truth, system_prompt) from trajectory and files.
    从轨迹和文件中加载元数据（问题、真实答案、系统提示）

    Args:
        traj_paths: Single path (str) or list of paths
        sample_dir: Top-level sample directory
        轨迹路径：单条路径（字符串）或路径列表
        样本目录：顶级样本目录
    
    Returns:
        {
            "question": str,
            "ground_truth": str,
            "system_prompt": str
        }
            “问题”：字符串类型
            “真实答案”：字符串类型
            “系统提示”：字符串类型
    """
    paths = _normalize_traj_paths(traj_paths)
    question = None
    ground_truth = None
    
    # 从第一条轨迹首行读取原始题目和 ground truth。
    if paths:
        try:
            with open(paths[0], "r", encoding="utf-8") as f:
                first = json.loads(next(f))
                question = first.get("initial_prompt")
                ground_truth = first.get("ground_truth")
        except Exception:
            pass
    
    # injected_system_prompt 可能在样本顶层，也可能在 rollout 子目录。
    system_prompt_text = None
    try:
        # First try sample_dir itself
        sp_path = os.path.join(sample_dir, "injected_system_prompt.txt")
        if os.path.exists(sp_path):
            with open(sp_path, "r", encoding="utf-8") as f:
                system_prompt_text = f.read().strip()
        else:
            # Try in rollout subdirectories
            for item in os.listdir(sample_dir):
                rollout_dir = os.path.join(sample_dir, item)
                if os.path.isdir(rollout_dir) and item.startswith("rollout_"):
                    sp_path = os.path.join(rollout_dir, "injected_system_prompt.txt")
                    if os.path.exists(sp_path):
                        with open(sp_path, "r", encoding="utf-8") as f:
                            system_prompt_text = f.read().strip()
                        break
    except Exception:
        pass
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt": system_prompt_text
    }


def summarize_rollouts(traj_paths: Union[str, List[str]], llm: ExperienceLLM, sample_dir: Optional[str] = None) -> dict:
    """Unified function to summarize single or multiple trajectory files.
    统一的函数用于对单个或多个轨迹文件进行汇总

    This function handles both single and multiple trajectory files.
    It loads trajectories with tool calls and results, formats them consistently,
    and generates summaries using LLM.
    此功能可处理单个及多个轨迹文件。
    它会加载包含工具调用及结果的轨迹数据，并对其进行统一格式化，
    然后使用 LLM 生成汇总报告。
    
    Args:
        traj_paths: Single trajectory path (str) or list of paths (List[str])
        llm: ExperienceLLM instance for generating summary
        sample_dir: Top-level sample directory (optional, will be inferred if not provided)
    
    Returns:
        {
            "process": summary_text,
            "question": question,
            "ground_truth": groundtruth,
            "system_prompt": system_prompt_text
        }
    """
    # 对应 Phase I / Rollout Summary：把多条轨迹和中间图像压缩成可学习摘要。
    paths = _normalize_traj_paths(traj_paths)
    if not paths:
        return {}
    uniq_paths = list(dict.fromkeys(paths))
    is_single = len(uniq_paths) == 1
    
    # 解析样本顶层目录，后续用于找图和写调试文件
    try:
        resolved_sample_dir = _resolve_sample_dir(uniq_paths, sample_dir)
    except Exception as e:
        print(f"  Warning: Failed to resolve sample_dir: {e}")
        # Fallback: use first path's directory
        resolved_sample_dir = os.path.dirname(uniq_paths[0])
    
    # 读取 question、ground truth、system prompt
    metadata = _load_metadata(uniq_paths, resolved_sample_dir)
    question = metadata.get("question")
    ground_truth = metadata.get("ground_truth")
    system_prompt_text = metadata.get("system_prompt")
    
    # 扫描样本目录中的所有图片，用于生成 caption
    all_images = _scan_all_images(resolved_sample_dir)
    
    if not all_images:
        print(f"  Warning: No images found in {resolved_sample_dir}")
        image_captions = {}
    else:
        print(f"  Found {len(all_images)} images in directory")
        
        # 图片 caption 会带上题目和 system prompt 作为背景
        base_task_context = {}
        if question:
            base_task_context["question"] = question
        if system_prompt_text:
            base_task_context["system_prompt"] = system_prompt_text
        
        # 为每张图片准备上下文；工具图像额外关联原始题图
        image_contexts: Dict[str, Dict[str, Any]] = {}
        image_to_rollout: Dict[str, str] = {}
        existing_image_filenames = []
        
        for caption_key, image_info in all_images.items():
            file_path = image_info["file_path"]
            rollout_dir = image_info["rollout_dir"]
            is_original = image_info["is_original"]
            
            # 每张图片都继承基础任务上下文
            task_context = base_task_context.copy()
            
            # 工具生成图像附带原图，帮助 caption 模型理解变化
            if not is_original:
                original_image_keys = _find_original_images_for_tool_image(
                    caption_key, rollout_dir, all_images
                )
                if original_image_keys:
                    # Get the actual Path objects for original images
                    original_image_paths = []
                    for orig_key in original_image_keys:
                        orig_info = all_images.get(orig_key)
                        if orig_info:
                            original_image_paths.append(orig_info["file_path"])
                    if original_image_paths:
                        task_context["original_image_paths"] = original_image_paths
            
            image_contexts[caption_key] = {
                "task_context": task_context,
                "generation_context": None  # Simplified: no generation context
            }
            image_to_rollout[caption_key] = rollout_dir
            existing_image_filenames.append(caption_key)
        
        # 先把所有图片变成文字描述，再写入轨迹摘要 prompt
        print(f"  Generating analysis for {len(existing_image_filenames)} images...")
        analysis_start = time.time()
        image_captions = generate_image_captions(
            existing_image_filenames,
            resolved_sample_dir,
            llm=llm,
            cache_dir=os.path.join(resolved_sample_dir, ".analysis_cache"),
            image_contexts=image_contexts,
            image_to_rollout=image_to_rollout
        )
        analysis_time = time.time() - analysis_start
        print(f"  [Timing] Image analysis generation: {analysis_time:.1f}s")
    
    # 预先拆成顶层 caption 和每个 rollout 的 caption，后续查找更快
    top_level_captions: Dict[str, str] = {}
    rollout_captions_map: Dict[str, Dict[str, str]] = {}
    for caption_key, caption_text in image_captions.items():
        if '/' in caption_key:
            key_rollout, filename = caption_key.split('/', 1)
            if key_rollout not in rollout_captions_map:
                rollout_captions_map[key_rollout] = {}
            rollout_captions_map[key_rollout][caption_key] = caption_text
            rollout_captions_map[key_rollout][filename] = caption_text
        else:
            top_level_captions[caption_key] = caption_text

    # 拼接多个 rollout 的轨迹内容
    formatted_blocks = []
    for rollout_idx, path in enumerate(uniq_paths):
        rollout_dir = Path(path).parent.name
        rollout_captions = {**top_level_captions, **rollout_captions_map.get(rollout_dir, {})}
        # 读取当前 rollout 的原始轨迹 JSONL
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_jsonl_content = f.read()
        except Exception as e:
            print(f"  Warning: Failed to read trajectory file {path}: {e}")
            continue
        
        if not raw_jsonl_content.strip():
            continue
        
        # 将轨迹中的图片引用替换成 caption。
        processed_content = _replace_image_refs_in_jsonl(raw_jsonl_content, rollout_captions)
        
        # 多 rollout 时加标题，方便 LLM 区分不同尝试。
        if not is_single:
            formatted_blocks.append(f"==== Rollout {rollout_idx} ({rollout_dir}) ====\n{processed_content}")
        else:
            formatted_blocks.append(processed_content)
    
    if not formatted_blocks:
        return {}
    
    process_text = "\n\n".join(formatted_blocks)
    header_prefix = "==== " if not is_single else ""
    header_parts = []
    if isinstance(question, str) and question.strip():
        # 题面中的 <image> 也替换成原图 caption。
        formatted_question = _replace_image_tags_in_question(question.strip(), image_captions)
        header_parts.append(f"{header_prefix}Question:\n{formatted_question}")
    if isinstance(system_prompt_text, str) and system_prompt_text.strip():
        header_parts.append(f"{header_prefix}System prompt:\n{system_prompt_text}")
    if isinstance(ground_truth, str) and ground_truth.strip():
        header_parts.append(f"{header_prefix}Ground truth (for reference):\n{ground_truth.strip()}")
    
    if header_parts:
        process_text = "\n\n".join(header_parts) + "\n\n" + process_text
    
    # 生成最终轨迹摘要 prompt
    prompt = SINGLE_ROLLOUT_SUMMARY.format(trajectory=process_text)
    
    # 原始题图会作为多模态输入再次提供给摘要模型 (在构造轨迹时会先用LLM生成图片的摘要)
    question_images = []
    if all_images:
        for caption_key, image_info in all_images.items():
            if image_info["is_original"] and image_info["rollout_dir"] == "":
                question_images.append(image_info["file_path"])
    
    # 调用 LLM 生成摘要；有原图时走多模态接口。
    response_text = None
    llm_start = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            if question_images:
                # 多图题把所有原图一起传入。
                pil_images = [Image.open(img_path).convert('RGB') for img_path in question_images]
                response_text = llm.chat_with_image(
                    prompt=prompt,
                    image=pil_images,
                    max_tokens=MAX_TOKENS,
                    return_placeholder_on_error=False
                )
            else:
                # 没有原图时退回纯文本摘要。
                response_text = llm.chat(prompt, max_tokens=MAX_TOKENS)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  Warning: Summary generation failed (attempt {attempt+1}/{MAX_RETRIES}): {e}, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    if response_text is None:
        raise RuntimeError("Failed to generate summary after all retries")
    
    llm_time = time.time() - llm_start
    print(f"  [Timing] exp_summary_resp generation: {llm_time:.1f}s")
    
    # 保存摘要 prompt 和响应，后续 skill/experience 生成会读取。
    try:
        # 无论单 rollout 还是多 rollout，都统一使用 exp_summary_* 命名。
        prompt_file = os.path.join(resolved_sample_dir, "exp_summary_prompt.txt")
        resp_file = os.path.join(resolved_sample_dir, "exp_summary_resp.txt")
        
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(resp_file, "w", encoding="utf-8") as f:
            f.write(response_text)
    except Exception:
        pass
    
    # 返回摘要和元信息，供 critique 阶段继续使用。
    return {
        "process": response_text,
        "question": question,
        "ground_truth": ground_truth,
        "system_prompt": system_prompt_text
    }
