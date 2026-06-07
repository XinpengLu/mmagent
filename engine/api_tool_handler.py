"""Tool execution and observation wiring for API-based inference."""

import os
import re
import json
import shutil
from PIL import Image
from utils.image_utils import estimate_tokens, pil_to_base64_data_uri, process_image
from utils.artifacts import save_trajectory
from qwen_vl_utils import fetch_image # type: ignore
from tools import get_tool

class APIToolHandler:
    """Execute tools and append their observations to the active trajectory."""

    def __init__(self, args, save_dir, tool_instances=None):
        self.args = args
        self.save_dir = save_dir
        self.tool_instances = tool_instances or {}
        self.tool_configs = getattr(args, 'tool_configs', {})

        self.tool_call_counts = {}
        image_search_max_calls = getattr(args, 'image_search_max_calls', None)
        web_search_max_calls = getattr(args, 'web_search_max_calls', None)
        code_interpreter_max_calls = getattr(args, 'code_interpreter_max_calls', None)

        if image_search_max_calls is None:
            image_search_max_calls = 3
        if web_search_max_calls is None:
            web_search_max_calls = 5

        self.tool_call_limits = {
            'image_search': image_search_max_calls,
            'web_search': web_search_max_calls,
        }
        if code_interpreter_max_calls is not None and int(code_interpreter_max_calls) > 0:
            self.tool_call_limits['code_interpreter'] = int(code_interpreter_max_calls)

    def close(self):
        """Release cached tool instances at the end of one rollout."""
        for tool in list(self.tool_instances.values()):
            try:
                close = getattr(tool, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        self.tool_instances.clear()

    def _save_event(self, node, event):
        if not self.save_dir:
            return
        try:
            if hasattr(node, "node_id") and node.node_id:
                event["node_id"] = node.node_id
            save_trajectory(self.save_dir, event)
        except Exception:
            pass

    def _determine_image_name(self, tool_name, image_ref, alt_text, node):
        name_without_ext = os.path.splitext(alt_text)[0]

        if re.match(r'^(tool_image_\d+|observation_\d+)$', name_without_ext):
            return name_without_ext

        print(f"[Warning] Tool {tool_name} did not generate final image name '{alt_text}', using fallback")
        tool_image_count = len([k for k in node.image_map.keys() if k.startswith('tool_image_')])
        tool_image_count += 1
        return f'tool_image_{tool_image_count}'

    def _skip_image_extraction(self, tool_name):
        if tool_name in ['image_search', 'visit']:
            if tool_name == 'image_search':
                print(f"[Image Processing] Skipping image extraction for {tool_name} (returns URLs only, no markdown images)")
            else:  # visit
                print(f"[Image Processing] Skipping image extraction for {tool_name} (returns text summary, images should not be extracted)")
            return True
        return False

    def _resolve_image_ref(self, image_ref):
        if image_ref.startswith('http://') or image_ref.startswith('https://'):
            return image_ref, fetch_image({'image': image_ref, 'max_pixels': self.args.max_pixels})

        resolved_ref = image_ref if os.path.isabs(image_ref) else os.path.join(self.save_dir, image_ref)
        if not os.path.exists(resolved_ref):
            print(f"Warning: Image file not found: {resolved_ref}")
            return resolved_ref, None
        return resolved_ref, Image.open(resolved_ref)

    def _process_tool_image(self, pil_image):
        original_size = pil_image.width * pil_image.height
        processed_image = process_image(pil_image, self.args.max_pixels, self.args.min_pixels)
        processed_size = processed_image.width * processed_image.height
        if original_size != processed_size:
            print(f"[Image Processing] Resized: {pil_image.width}x{pil_image.height} ({original_size:,}px) -> {processed_image.width}x{processed_image.height} ({processed_size:,}px)")
        else:
            print(f"[Image Processing] No resize needed: {processed_image.width}x{processed_image.height} ({processed_size:,}px)")
        return processed_image

    def _original_filename(self, image_ref):
        if os.path.isfile(image_ref) or '/' in image_ref or '\\' in image_ref:
            return os.path.basename(image_ref)
        return None

    def _create_image_alias(self, original_filename, image_name):
        if not original_filename:
            return
        original_path = os.path.join(self.save_dir, original_filename)
        standard_path = os.path.join(self.save_dir, f"{image_name}.jpg")
        if os.path.exists(original_path) or not os.path.exists(standard_path):
            return
        try:
            shutil.copy(standard_path, original_path)
            print(f"[Image Alias] Created alias: {original_filename} -> {image_name}.jpg")
        except Exception as e:
            print(f"[Image Alias] Failed to create alias: {e}")

    def _image_replacement_text(self, image_name, original_filename):
        if original_filename:
            return f'[Image: {image_name}, file: {original_filename}]'
        return f'[Image: {image_name}]'

    def _load_markdown_image(self, tool_name, alt_text, image_ref, node):
        resolved_ref, pil_image = self._resolve_image_ref(image_ref)
        if pil_image is None:
            return None
        processed_image = self._process_tool_image(pil_image)
        image_name = self._determine_image_name(tool_name, resolved_ref, alt_text, node)
        node.image_map[image_name] = processed_image
        original_filename = self._original_filename(resolved_ref)
        self._create_image_alias(original_filename, image_name)
        print(f"[Image Processing] Loaded {image_name} from tool output")
        return image_name, processed_image, resolved_ref, original_filename

    def process_tool_output_for_images(self, tool_result, tool_name, node):
        """Load Markdown image references from a tool result into node.image_map."""
        if self._skip_image_extraction(tool_name):
            return [], tool_result

        new_images = []
        processed_text = tool_result
        for alt_text, image_ref in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', tool_result):
            try:
                loaded = self._load_markdown_image(tool_name, alt_text, image_ref, node)
                if not loaded:
                    continue
                image_name, processed_image, resolved_ref, original_filename = loaded
                new_images.append((image_name, processed_image))
                processed_text = processed_text.replace(
                    f'![{alt_text}]({resolved_ref})',
                    self._image_replacement_text(image_name, original_filename),
                )
            except Exception as e:
                print(f"Error loading image from {image_ref}: {e}")
        return new_images, processed_text

    def handle_image_search_reference(self, parameters, node):
        """Resolve image_search image_url values before executing the tool."""
        if 'image_url' not in parameters:
            return

        image_ref = parameters['image_url']

        if image_ref.startswith('http://') or image_ref.startswith('https://'):
            print(f"[Image Reference] Using URL directly: {image_ref[:80]}...")
            parameters['image_url'] = image_ref
            return

        image_key = os.path.splitext(image_ref)[0]

        found_image = False

        if image_key in node.image_map:
            img = node.image_map[image_key]
            file_path = os.path.join(self.save_dir, f"{image_key}.jpg")
            if not os.path.exists(file_path):
                img.save(file_path, 'JPEG', quality=95)
                print(f"[Image Saved] {image_key} -> {file_path}")
            parameters['image_url'] = file_path
            print(f"[Image Reference] Using {image_key} from node.image_map: {file_path}")
            found_image = True

        if not found_image and os.path.exists(os.path.join(self.save_dir, f"{image_key}.jpg")):
            file_path = os.path.join(self.save_dir, f"{image_key}.jpg")
            parameters['image_url'] = file_path
            print(f"[Image Reference] Using existing file: {file_path}")
            found_image = True

        if not found_image and os.path.exists(os.path.join(self.save_dir, image_ref)):
            file_path = os.path.join(self.save_dir, image_ref)
            parameters['image_url'] = file_path
            print(f"[Image Reference] Using original filename alias: {file_path}")
            found_image = True

        if not found_image:
            error_msg = f"Image reference '{image_ref}' not found in node.image_map or save_dir. Available images: 'tool_image_k', 'observation_k' or original filenames. For URLs from image_search results, use the URL directly."
            print(f"[Error] {error_msg}")
            raise ValueError(error_msg)

    def get_or_create_tool(self, tool_name):
        if tool_name not in self.tool_instances:
            tool_cls = get_tool(tool_name)

            tool_config = {}
            if tool_name in self.tool_configs:
                tool_config = self.tool_configs[tool_name]

            tool_config['args'] = self.args
            tool_config['save_dir'] = self.save_dir

            self.tool_instances[tool_name] = tool_cls(config=tool_config)
            print(f"[Tool Cache] Created new instance of {tool_name}")
        else:
            print(f"[Tool Cache] Reusing existing instance of {tool_name}")

        return self.tool_instances[tool_name]

    def prepare_tool_kwargs(self, tool_name, node):
        tool_kwargs = {}
        if tool_name in ('code_interpreter', 'zoom'):
            tool_kwargs['image_map'] = node.image_map.copy()
        if tool_name == 'image_search':
            tool_kwargs['save_dir'] = self.save_dir

        return tool_kwargs

    def _tool_feedback(self, feedback_msg, tool_call_id):
        return [{
            "role": "tool",
            "content": json.dumps({"error": feedback_msg}),
            "tool_call_id": tool_call_id or "call_0",
        }]

    def _error_result(self, feedback_msg, tool_call_id):
        return {
            'tool_result': '',
            'processed_result': '',
            'new_images': [],
            'feedback_messages': self._tool_feedback(feedback_msg, tool_call_id),
            'error': True,
        }

    def _limit_exceeded_result(self, tool_name, node, tool_call_id):
        tool_limit = self.tool_call_limits.get(tool_name, None)
        if tool_limit is None:
            return None

        current_count = self.tool_call_counts.get(tool_name, 0)
        if current_count < tool_limit:
            return None

        error_msg = f"Maximum number of '{tool_name}' calls ({tool_limit}) exceeded. Current count: {current_count}"
        print(f"[Tool Limit] {error_msg}")
        feedback_msg = (
            f"{error_msg}\n\nDo not call {tool_name} again in this rollout. "
            "Use the existing observations and tool outputs to provide the best supported final answer now."
        )
        node.conversation_history.append({"role": "user", "content": feedback_msg})
        node.current_token_count += estimate_tokens(feedback_msg)
        return self._error_result(feedback_msg, tool_call_id)

    def _call_tool(self, tool_name, parameters, node):
        tool = self.get_or_create_tool(tool_name)
        if tool_name == 'image_search' and 'image_url' in parameters:
            self.handle_image_search_reference(parameters, node)
        return tool.call(parameters, **self.prepare_tool_kwargs(tool_name, node))

    def _save_tool_call_event(self, node, turn_idx, tool_name, parameters, tool_result):
        self._save_event(node, {
            "turn_idx": turn_idx,
            "tool_call": {
                "tool_name": tool_name,
                "parameters": parameters,
                "result": tool_result,
            },
        })

    def _save_tool_images(self, node, turn_idx, tool_name, new_images):
        for image_name, pil_image in new_images:
            image_path = os.path.join(self.save_dir, f"{image_name}.jpg")
            if not os.path.exists(image_path):
                pil_image.save(image_path)
                print(f"[Image Saved] {image_name} -> {image_path}")
            self._save_event(node, {
                "turn_idx": turn_idx,
                "tool_image": {
                    "image_name": image_name,
                    "source_tool": tool_name,
                    "file_path": f"{image_name}.jpg",
                },
            })

    def _tool_usage_text(self, tool_name):
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1
        if tool_name not in self.tool_call_limits:
            return ""

        remaining_calls_info = []
        for tname, limit in self.tool_call_limits.items():
            if limit is not None:
                current = self.tool_call_counts.get(tname, 0)
                remaining_calls_info.append(f"{tname} {current}/{limit}")
        return f"\n\n[Tool Usage: {', '.join(remaining_calls_info)}]" if remaining_calls_info else ""

    def _image_feedback_message(self, new_images, processed_result, usage_text):
        message_content = []
        total_base64_size = 0
        for image_name, pil_image in new_images:
            base64_data = pil_to_base64_data_uri(pil_image)
            base64_size = len(base64_data)
            total_base64_size += base64_size
            print(f"[Image Encoding] {image_name}: {pil_image.width}x{pil_image.height} -> base64 size: {base64_size:,} bytes ({base64_size/1024:.1f} KB)")
            message_content.append({
                "type": "image_url",
                "image_url": {"url": base64_data},
            })
        print(f"[Image Encoding] Total base64 size for {len(new_images)} images: {total_base64_size:,} bytes ({total_base64_size/1024:.1f} KB, {total_base64_size/1024/1024:.2f} MB)")
        message_content.append({"type": "text", "text": f"Images from tool output: {processed_result}{usage_text}"})
        return {"role": "user", "content": message_content}

    def _build_success_feedback(self, tool_name, tool_result, processed_result, new_images, tool_call_id):
        feedback_messages = [{
            "role": "tool",
            "content": json.dumps({"result": processed_result if new_images else tool_result}),
            "tool_call_id": tool_call_id or "call_0",
        }]
        usage_text = self._tool_usage_text(tool_name)
        if new_images:
            feedback_messages.append(self._image_feedback_message(new_images, processed_result, usage_text))
            feedback_msg = f"Tool '{tool_name}' returned with images: {processed_result}{usage_text}"
        else:
            feedback_msg = f"Tool '{tool_name}' returned: {tool_result}{usage_text}"
        return feedback_messages, feedback_msg

    def _record_tool_feedback(self, node, feedback_msg):
        node.conversation_history.append({"role": "user", "content": feedback_msg})
        node.current_token_count += estimate_tokens(feedback_msg)

    def _tool_exception_result(self, tool_name, node, turn_idx, tool_call_id, exc):
        error_msg = f"Error executing tool '{tool_name}': {str(exc)}"
        print(f"[Tool Error] {error_msg}")
        self._save_event(node, {
            "turn_idx": turn_idx,
            "tool_error": {
                "tool_name": tool_name,
                "error": str(exc),
            },
        })
        feedback_msg = f"{error_msg}\n\nPlease try a different approach or provide your answer based on available information."
        self._record_tool_feedback(node, feedback_msg)
        return self._error_result(feedback_msg, tool_call_id)

    def execute_tool_call(self, tool_name, parameters, node, turn_idx, tool_call_id=None):
        """Execute one tool call and return messages to append to the model history."""
        print(f"[Tool Call] {tool_name} with params: {parameters}")

        limit_result = self._limit_exceeded_result(tool_name, node, tool_call_id)
        if limit_result:
            return limit_result

        try:
            tool_result = self._call_tool(tool_name, parameters, node)
            self._save_tool_call_event(node, turn_idx, tool_name, parameters, tool_result)
            new_images, processed_result = self.process_tool_output_for_images(
                tool_result, tool_name, node
            )
            if new_images:
                self._save_tool_images(node, turn_idx, tool_name, new_images)
            feedback_messages, feedback_msg = self._build_success_feedback(
                tool_name, tool_result, processed_result, new_images, tool_call_id
            )
            self._record_tool_feedback(node, feedback_msg)

            return {
                'tool_result': tool_result,
                'processed_result': processed_result,
                'new_images': new_images,
                'feedback_messages': feedback_messages,
                'error': False,
                'skip_processing': False
            }

        except Exception as e:
            return self._tool_exception_result(tool_name, node, turn_idx, tool_call_id, e)
