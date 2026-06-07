import os
import time
import requests
from typing import Optional, Dict, List, Union, Any
from PIL import Image

from .config import ExperienceLLMSettings
from .utils import _image_to_base64


_DEFAULT_SETTINGS = ExperienceLLMSettings.from_env()
MAX_RETRIES = _DEFAULT_SETTINGS.max_retries

API_TIMEOUT = _DEFAULT_SETTINGS.api_timeout
API_TIMEOUT_IMAGE = _DEFAULT_SETTINGS.image_api_timeout

MAX_TOKENS_DEFAULT = _DEFAULT_SETTINGS.default_max_tokens


def _is_openai_reasoning_model(model_name: str) -> bool:
    """Return True for OpenAI o-series / GPT-5 reasoning models."""
    model_id = (model_name or "").lower().split("/")[-1]
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


def _default_system_prompt() -> str:
    return "You are an expert at summarizing visual reasoning trajectories and extracting generalizable experiences."


def _build_chat_payload(
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
            {"role": "system", "content": system_prompt or _default_system_prompt()},
            {"role": "user", "content": user_content},
        ],
    }
    if _is_openai_reasoning_model(model_name):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload.update({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens})
    return payload


def _placeholder(enabled: bool, reason: str) -> Optional[str]:
    return f"[Failed to generate: {reason}]" if enabled else None


def _api_configured(api_key: Optional[str], endpoint: Optional[str]) -> bool:
    return bool(api_key and endpoint)


class ExperienceLLM:
    def __init__(self, model_name: Optional[str] = None):
        self.settings = ExperienceLLMSettings.from_env(model_name)
        self.model_name = self.settings.model_name

    def chat(self, prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT, temperature: Optional[float] = None, top_p: float = 1.0) -> str:
        t = temperature if temperature is not None else 0.6
        return self._call_with_fallback(
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=t,
            top_p=top_p,
            primary_api_name="Primary Experience API",
            fallback_api_name="Fallback Experience API"
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
        timeout: int = API_TIMEOUT,
        return_placeholder_on_error: bool = False
    ) -> str:
        """Call one configured endpoint with bounded retries."""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = _build_chat_payload(self.model_name, user_content, max_tokens, temperature, top_p, system_prompt)
        last_error = ""

        for attempt_idx in range(max_retries):
            attempt = attempt_idx + 1
            try:
                resp = requests.post(end_point, headers=headers, json=payload, timeout=timeout)
                content = self._content_from_response(resp)
                if content:
                    return content
                if resp.status_code == 200:
                    last_error = f"[{api_name}] API returned empty content on attempt {attempt}/{max_retries}"
                    print(last_error)
                else:
                    last_error = self._response_error_message(api_name, resp, attempt, max_retries)
                    print(last_error)
                    if resp.status_code in {400, 401, 403, 404}:
                        return self._finish_api_failure(last_error, return_placeholder_on_error, str(resp.status_code))
                    if resp.status_code >= 500 and attempt < max_retries:
                        print(f"[{api_name}] Transient server error; waiting {self._retry_wait(attempt_idx):.2f}s before retrying.")
            except requests.exceptions.RequestException as e:
                reason = "timeout" if isinstance(e, requests.exceptions.Timeout) else str(e)
                last_error = f"[{api_name}] API call failed on attempt {attempt}/{max_retries}: {reason}"
                print(last_error)
            except RuntimeError:
                raise
            except Exception as e:
                last_error = f"[{api_name}] Unexpected error on attempt {attempt}/{max_retries}: {e}"
                print(last_error)

            if attempt < max_retries:
                time.sleep(self._retry_wait(attempt_idx))

        return self._finish_api_failure(
            last_error or f"[{api_name}] All retry attempts exhausted",
            return_placeholder_on_error,
            "empty response" if "empty content" in last_error else "API error",
        )

    def _retry_wait(self, attempt_idx: int) -> float:
        return min(self.settings.base_wait * (2 ** attempt_idx), self.settings.max_wait)

    def _content_from_response(self, resp: requests.Response) -> str:
        if resp.status_code != 200:
            return ""
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def _response_error_message(self, api_name: str, resp: requests.Response, attempt: int, max_retries: int) -> str:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        return f"[{api_name}] Experience API error {resp.status_code} on attempt {attempt}/{max_retries}: {detail}"

    def _finish_api_failure(self, message: str, return_placeholder_on_error: bool, placeholder_reason: str) -> str:
        placeholder = _placeholder(return_placeholder_on_error, placeholder_reason)
        if placeholder is not None:
            return placeholder
        raise RuntimeError(message)

    def _get_api_config(self, primary_key: str, primary_endpoint: str, 
                       fallback_key: str, fallback_endpoint: str) -> tuple:
        openai_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        openai_chat_endpoint = os.environ.get("OPENAI_CHAT_COMPLETIONS_URL")
        if not openai_chat_endpoint and openai_base:
            openai_chat_endpoint = openai_base.rstrip("/") + "/chat/completions"

        api_key_1 = (
            os.environ.get(primary_key)
            or os.environ.get("REASONING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        end_point_1 = (
            os.environ.get(primary_endpoint)
            or os.environ.get("REASONING_END_POINT")
            or openai_chat_endpoint
        )
        
        api_key_2 = os.environ.get(fallback_key) or os.environ.get("REASONING_API_KEY_2")
        end_point_2 = os.environ.get(fallback_endpoint) or os.environ.get("REASONING_END_POINT_2")
        
        return api_key_1, end_point_1, api_key_2, end_point_2

    def _normalize_endpoint(self, endpoint: str, require_chat_completions: bool = False) -> str:
        if require_chat_completions and not endpoint.endswith("/chat/completions"):
            if endpoint.endswith("/"):
                endpoint = endpoint + "chat/completions"
            else:
                endpoint = endpoint + "/chat/completions"
        return endpoint

    def _primary_experience_config(self, require_chat_completions: bool) -> tuple:
        api_key, endpoint, _, _ = self._get_api_config(
            "EXPERIENCE_API_KEY", "EXPERIENCE_END_POINT",
            "EXPERIENCE_API_KEY_2", "EXPERIENCE_END_POINT_2"
        )
        endpoint = self._normalize_endpoint(endpoint, require_chat_completions) if endpoint else endpoint
        return api_key, endpoint

    def _fallback_experience_config(self, require_chat_completions: bool) -> tuple:
        _, _, api_key, endpoint = self._get_api_config(
            "EXPERIENCE_API_KEY", "EXPERIENCE_END_POINT",
            "EXPERIENCE_API_KEY_2", "EXPERIENCE_END_POINT_2"
        )
        endpoint = self._normalize_endpoint(endpoint, require_chat_completions) if endpoint else endpoint
        return api_key, endpoint

    def _call_configured_api(
        self,
        api_key: str,
        endpoint: str,
        api_name: str,
        user_content: Union[str, List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        max_retries: int,
        system_prompt: Optional[str],
        timeout: int,
        return_placeholder_on_error: bool,
    ) -> str:
        return self._try_single_experience_api(
            api_key, endpoint, user_content, max_tokens,
            temperature, top_p, max_retries,
            api_name=api_name,
            system_prompt=system_prompt,
            timeout=timeout,
            return_placeholder_on_error=return_placeholder_on_error
        )

    def _call_fallback_api(
        self,
        primary_error: RuntimeError,
        user_content: Union[str, List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        max_retries: int,
        system_prompt: Optional[str],
        timeout: int,
        return_placeholder_on_error: bool,
        primary_api_name: str,
        fallback_api_name: str,
        require_chat_completions: bool,
    ) -> str:
        fallback_key, fallback_endpoint = self._fallback_experience_config(require_chat_completions)
        if not _api_configured(fallback_key, fallback_endpoint):
            if return_placeholder_on_error:
                return f"[Failed: {str(primary_error)}]"
            raise RuntimeError(f"{primary_api_name} failed and no fallback configured: {primary_error}")

        print(f"[API Fallback] {primary_api_name} failed, switching to {fallback_api_name}...")
        try:
            return self._call_configured_api(
                fallback_key, fallback_endpoint, fallback_api_name,
                user_content, max_tokens, temperature, top_p, max_retries,
                system_prompt, timeout, return_placeholder_on_error
            )
        except RuntimeError as fallback_error:
            if return_placeholder_on_error:
                return "[Failed: both APIs failed]"
            raise RuntimeError(
                f"Both {primary_api_name} and {fallback_api_name} failed. "
                f"Primary: {primary_error}. Fallback: {fallback_error}"
            )

    def _call_with_fallback(
        self,
        user_content: Union[str, List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        system_prompt: Optional[str] = None,
        timeout: int = API_TIMEOUT,
        return_placeholder_on_error: bool = False,
        primary_api_name: str = "Primary API",
        fallback_api_name: str = "Fallback API",
        require_chat_completions: bool = False
    ) -> str:
        """Call the primary experience API, then fallback if configured."""
        max_retries = MAX_RETRIES
        api_key, endpoint = self._primary_experience_config(require_chat_completions)
        if not _api_configured(api_key, endpoint):
            if return_placeholder_on_error:
                return "[API not configured]"
            raise ValueError("EXPERIENCE_API_KEY/END_POINT or REASONING_API_KEY/END_POINT must be set")

        try:
            return self._call_configured_api(
                api_key, endpoint, primary_api_name,
                user_content, max_tokens, temperature, top_p, max_retries,
                system_prompt, timeout, return_placeholder_on_error
            )
        except RuntimeError as primary_error:
            return self._call_fallback_api(
                primary_error, user_content, max_tokens, temperature, top_p, max_retries,
                system_prompt, timeout, return_placeholder_on_error,
                primary_api_name, fallback_api_name, require_chat_completions
            )

    def chat_with_image(
        self, 
        prompt: str, 
        image: Union[Image.Image, List[Image.Image]],
        max_tokens: int = MAX_TOKENS_DEFAULT, 
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        return_placeholder_on_error: bool = True
    ) -> str:
        """Call the multimodal experience API with one or more PIL images."""
        t = temperature if temperature is not None else 0.6
        if system_prompt is None:
            system_prompt = "You are an expert at analyzing images and providing detailed visual descriptions."
        
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image
        
        user_content = [{"type": "text", "text": prompt}]
        
        for img in images:
            base64_image = _image_to_base64(img)
            user_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        return self._call_with_fallback(
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=t,
            top_p=top_p,
            system_prompt=system_prompt,
            timeout=API_TIMEOUT_IMAGE,
            return_placeholder_on_error=return_placeholder_on_error,
            primary_api_name="Primary Multimodal API",
            fallback_api_name="Fallback Multimodal API",
            require_chat_completions=True
        )
