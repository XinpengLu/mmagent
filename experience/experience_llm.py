import os
import time
import requests
from typing import Optional, Dict, List, Union, Any
from PIL import Image

from .config import ExperienceLLMSettings
from .utils import image_to_base64
from utils.api_usage_logger import log_api_usage
from utils.key_pool import openai_should_rotate, rotate_current_key


def is_openai_reasoning_model(model_name: str) -> bool:
    """Return True for OpenAI o-series / GPT-5 reasoning models."""
    model_id = (model_name or "").lower().split("/")[-1]
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


def default_system_prompt() -> str:
    return "You are an expert at summarizing visual reasoning trajectories and extracting generalizable experiences."


def build_chat_payload(
    model_name: str,
    user_content: Union[str, List[Dict[str, Any]]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str],
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt or default_system_prompt()},
            {"role": "user", "content": user_content},
        ],
    }

    # Reasoning model
    if is_openai_reasoning_model(model_name):
        payload["max_completion_tokens"] = max_tokens
    # Other model
    else:
        payload.update({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens})
    return payload


class ExperienceLLM:
    def __init__(self, model_name: Optional[str] = None):
        self.settings = ExperienceLLMSettings.from_env(model_name)
        self.model_name = self.settings.model_name

    def chat(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: float = 1.0) -> str:
        max_tokens = self.settings.default_max_tokens
        t = temperature if temperature is not None else 0.6
        return self._call_api(
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=t,
            top_p=top_p,
            api_name="Experience API",
        )


    def _call_api(
        self,
        user_content: Union[str, List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        system_prompt: Optional[str] = None,
        timeout: Optional[int] = None,
        api_name: str = "Experience API",
    ) -> str:
        """Call the configured experience API with bounded retries."""
        max_retries = self.settings.max_retries
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        endpoint = f"{base_url}/chat/completions" if base_url else ""
        if not api_key or not endpoint:
            raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL must be set")

        return self._try_single_experience_api(
            api_key,
            endpoint,
            user_content,
            max_tokens,
            temperature,
            top_p,
            max_retries,
            api_name=api_name,
            system_prompt=system_prompt,
            timeout=timeout,
        )

    def _try_single_experience_api(
        self, 
        api_key: str, 
        end_point: str, 
        user_content: Union[str, List[Dict[str, Any]]], 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        max_retries: int, 
        api_name: str = "API",
        system_prompt: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Call one configured endpoint with bounded retries."""
        timeout = self.settings.api_timeout if timeout is None else timeout
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = build_chat_payload(self.model_name, user_content, max_tokens, temperature, top_p, system_prompt)
        last_error = ""

        for attempt_idx in range(max_retries):
            attempt = attempt_idx + 1
            try:
                resp = requests.post(end_point, headers=headers, json=payload, timeout=timeout)
                try:
                    response_data = resp.json()
                except Exception:
                    response_data = None
                log_api_usage(
                    source=api_name,
                    model=self.model_name,
                    endpoint=end_point,
                    payload=payload,
                    response_data=response_data,
                    status_code=resp.status_code,
                    elapsed_seconds=resp.elapsed.total_seconds(),
                    success=resp.status_code == 200,
                    attempt=attempt,
                )
                content = self.content_from_response(resp)
                if content:
                    return content
                if resp.status_code == 200:
                    last_error = f"[{api_name}] API returned empty content on attempt {attempt}/{max_retries}"
                    print(last_error)
                else:
                    last_error = self.response_error_message(api_name, resp, attempt, max_retries)
                    print(last_error)
                    if openai_should_rotate(resp.status_code, last_error):
                        api_key = rotate_current_key("openai", api_key)
                        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                        continue
                    if resp.status_code in {400, 401, 403, 404}:
                        raise RuntimeError(last_error)
                    if resp.status_code >= 500 and attempt < max_retries:
                        print(f"[{api_name}] Transient server error; waiting {self._retry_wait(attempt_idx):.2f}s before retrying.")
            except requests.exceptions.RequestException as e:
                reason = "timeout" if isinstance(e, requests.exceptions.Timeout) else str(e)
                log_api_usage(
                    source=api_name,
                    model=self.model_name,
                    endpoint=end_point,
                    payload=payload,
                    success=False,
                    error=reason,
                    attempt=attempt,
                )
                last_error = f"[{api_name}] API call failed on attempt {attempt}/{max_retries}: {reason}"
                print(last_error)
            except RuntimeError:
                raise
            except Exception as e:
                last_error = f"[{api_name}] Unexpected error on attempt {attempt}/{max_retries}: {e}"
                print(last_error)

            if attempt < max_retries:
                time.sleep(min(self.settings.base_wait * (2 ** attempt_idx), self.settings.max_wait))

        raise RuntimeError(last_error or f"[{api_name}] All retry attempts exhausted")

    def content_from_response(self, resp: requests.Response) -> str:
        if resp.status_code != 200:
            return ""
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def response_error_message(self, api_name: str, resp: requests.Response, attempt: int, max_retries: int) -> str:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        return f"[{api_name}] Experience API error {resp.status_code} on attempt {attempt}/{max_retries}: {detail}"

    def chat_with_image(
        self,
        prompt: str,
        image: Union[Image.Image, List[Image.Image]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
    ) -> str:
        """Call the multimodal experience API with one or more PIL images."""
        max_tokens = self.settings.default_max_tokens if max_tokens is None else max_tokens
        t = temperature if temperature is not None else 0.6
        
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image
        
        user_content = [{"type": "text", "text": prompt}]
        
        for img in images:
            base64_image = image_to_base64(img)
            user_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        return self._call_api(
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=t,
            top_p=top_p,
            system_prompt="You are an expert at analyzing images and providing detailed visual descriptions.",
            timeout=self.settings.image_api_timeout,
            api_name="Multimodal Experience API",
        )
