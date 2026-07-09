"""
LLM-as-Judge evaluator for assessing answer correctness.

This module provides an LLM-based evaluation system that compares predicted
answers against ground truth using configurable prompts.
"""

import re
import os
import time
import openai
from typing import Tuple, List, Optional, Dict

from prompts.judge_templates import SYSTEM_PROMPT, QUERY_PROMPT
from utils.api_usage_logger import log_api_usage
from utils.key_pool import openai_should_rotate, rotate_current_key


# Global client instance (lazily initialized)
_client = None


def _get_client():
    """Get or create the global LLM judge client instance."""
    global _client
    if _client is None:
        _client = LLMJudgeClient()
    return _client


def _is_openai_reasoning_model(model_name: str) -> bool:
    """Return True for OpenAI o-series / GPT-5 reasoning models."""
    model_id = (model_name or "").lower().split("/")[-1]
    return model_id.startswith(("o1", "o3", "o4", "gpt-5"))


def _judge_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict]:
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    })
    return messages


def _score_from_response_text(response_text: str) -> str:
    if 'score:' not in response_text.lower():
        raise ValueError("No 'score:' found in response")
    score_str = (
        response_text.lower()
        .split("score:")[-1]
        .strip()
        .split("\n")[0]
        .strip()
        .split(" ")[0]
    )
    if "1" not in score_str and '0' not in score_str:
        raise ValueError(f"No valid score ('0' or '1') found: {score_str}")
    return score_str


def _response_message_content(response) -> str:
    if not response or not hasattr(response, 'choices') or not response.choices:
        raise ValueError("Invalid response: missing choices field")
    if len(response.choices) == 0:
        raise ValueError("Invalid response: empty choices array")
    first_choice = response.choices[0]
    if not first_choice or not hasattr(first_choice, 'message'):
        raise ValueError("Invalid response: missing message in choices[0]")
    message = first_choice.message
    if not message or not hasattr(message, 'content'):
        raise ValueError("Invalid response: missing content in message")
    if not message.content:
        raise ValueError("Invalid response: empty content")
    return message.content


def _analysis_from_response(response_text: str) -> str:
    if response_text and "explanation:" in response_text.lower():
        explanation_part = response_text.lower().split("explanation:")[-1].strip()
        if explanation_part:
            return explanation_part
    return "No explanation provided in response" if not response_text else ""


def _usage_dict(response) -> Dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else {}


class LLMJudgeClient:
    """Client for LLM-based answer evaluation using OpenAI-compatible APIs."""

    def __init__(self):
        """Initialize the LLM judge client with environment variables."""
        self.api_key = os.environ.get("VERIFIER_API_KEY")
        self.endpoint = os.environ.get("VERIFIER_END_POINT")
        self.model_name = os.environ.get("VERIFIER_MODEL_NAME")
        
        if not self.api_key or not self.endpoint:
            raise ValueError(
                "VERIFIER_API_KEY and VERIFIER_END_POINT must be set in environment variables"
            )
        
        self._build_client()

    def _build_client(self) -> None:
        self.client = openai.OpenAI(base_url=self.endpoint, api_key=self.api_key)

    def _rotate_api_key(self) -> None:
        self.api_key = rotate_current_key("verifier", self.api_key)
        self._build_client()

    def evaluate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        max_retries: int = 5
    ) -> Tuple[str, str]:
        """
        Evaluate an answer using the LLM judge.
        
        Args:
            prompt: The evaluation prompt with question, ground truth, and prediction
            system_prompt: Optional system prompt (defaults to None)
            max_retries: Maximum number of retry attempts (default: 5)
            
        Returns:
            Tuple of (score_str, response_text) where:
            - score_str: Extracted score ("1" or "0") 
            - response_text: Full response text with explanation
        """
        messages = _judge_messages(prompt, system_prompt)

        for attempt in range(max_retries):
            try:
                start = time.time()
                kwargs = self._request_kwargs(messages, attempt)
                response = self.client.chat.completions.create(
                    **kwargs
                )
                log_api_usage(
                    source="Verifier API",
                    model=self.model_name,
                    endpoint=os.environ.get("VERIFIER_END_POINT", ""),
                    payload={"messages": messages},
                    response_data={"usage": _usage_dict(response)},
                    elapsed_seconds=time.time() - start,
                    success=True,
                    attempt=attempt + 1,
                )
                response_text = _response_message_content(response)
                return _score_from_response_text(response_text), response_text
                
            except openai.RateLimitError as e:
                error_text = str(e)
                log_api_usage(
                    source="Verifier API",
                    model=self.model_name,
                    endpoint=os.environ.get("VERIFIER_END_POINT", ""),
                    payload={"messages": messages},
                    success=False,
                    error=error_text,
                    attempt=attempt + 1,
                )
                print(f"[RateLimitError] Attempt {attempt + 1}/{max_retries}: {error_text}")
                if openai_should_rotate(getattr(e, "status_code", None), error_text):
                    self._rotate_api_key()
                    continue
                time.sleep(min(3 * (attempt + 1), 10))
                continue
                
            except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as e:
                log_api_usage(
                    source="Verifier API",
                    model=self.model_name,
                    endpoint=os.environ.get("VERIFIER_END_POINT", ""),
                    payload={"messages": messages},
                    success=False,
                    error=str(e),
                    attempt=attempt + 1,
                )
                error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                print(f"[APIError] Attempt {attempt + 1}/{max_retries}: {str(e)} (code: {error_code})")
                if openai_should_rotate(error_code, str(e)):
                    self._rotate_api_key()
                    continue
                if error_code in {401, 403}:
                    break
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt, 10)
                    time.sleep(delay)
                continue
                
            except Exception as e:
                print("=" * 100)
                print(f"[Error] Attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Messages length: {len(str(messages))}")
                    print("=" * 100)
                    delay = min(2 ** attempt, 10)
                    time.sleep(delay)
                else:
                    print(f"Messages: {messages}")
                    print("=" * 100)
                continue
        
        print(f"Warning: Evaluation failed after {max_retries} attempts")
        return "", ""

    def _request_kwargs(self, messages: List[Dict], attempt: int) -> Dict:
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "timeout": 120,
        }
        if _is_openai_reasoning_model(self.model_name):
            request_kwargs["max_completion_tokens"] = 8192
        else:
            request_kwargs["temperature"] = min(0.2 * attempt, 1.0)
            request_kwargs["max_tokens"] = 8192
        return request_kwargs


def _extract_answer_from_text(
    text: str, 
    predict_str_list: List[str], 
    extraction_mode: str
) -> Tuple[float, str]:
    """
    Extract answer from prediction text using specified extraction mode.
    
    Args:
        text: Full prediction text
        predict_str_list: List of prediction strings (for fallback)
        extraction_mode: Extraction mode ('split', 'strict', or 'strict_v2')
        
    Returns:
        Tuple of (reward, extracted_answer) where reward is 0.0 on failure
    """
    if extraction_mode == 'split':
        return 0.0, text.split("<answer>")[-1].split("</answer>")[0].strip()
    
    elif extraction_mode in ['strict', 'strict_v2']:
        # Extract the LAST <answer> tag (handles multiple answers from revisions)
        answer_pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Use the last match (most recent answer)
            return 0.0, matches[-1].strip()
        
        # Fallback: use the last turn's output if no answer tags found
        if predict_str_list:
            last_turn = predict_str_list[-1].strip()
            
            # Skip error messages
            if last_turn.startswith("Error:") or "API attempts failed" in last_turn:
                return 0.0, ""
            
            return 0.0, last_turn
        
        return 0.0, ""
    
    else:
        raise ValueError(f"Unknown extraction mode: {extraction_mode}")


def _prediction_text(prediction_list: List[str], extract_answer: bool, extra_info: Optional[Dict]) -> Tuple[bool, str]:
    prediction_text = ' '.join(prediction_list)
    if not (extract_answer and extra_info):
        return True, prediction_text

    _, extracted = _extract_answer_from_text(
        prediction_text,
        prediction_list,
        extra_info.get('extract_answer_tags', 'strict'),
    )
    return bool(extracted), extracted


def _evaluation_prompt(question: str, ground_truth: str, prediction_text: str) -> str:
    eval_prompt = QUERY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction_text
    )
    if len(eval_prompt) > 10000:
        print(f"[evaluate_answer] Warning: Very long prompt ({len(eval_prompt)} chars)")
    return eval_prompt


def _call_judge(eval_prompt: str) -> Tuple[str, str]:
    client = _get_client()
    return client.evaluate(prompt=eval_prompt, system_prompt=SYSTEM_PROMPT)


def _reward_from_score(score_str: str) -> float:
    return 1.0 if '1' in score_str else 0.0


def evaluate_answer(
    question: str,
    prediction_list: List[str],
    ground_truth: str,
    extract_answer: bool = False,
    extra_info: Optional[Dict] = None
) -> Tuple[float, str]:
    """
    Evaluate a predicted answer against ground truth using LLM-as-judge.
    
    Args:
        question: The question text
        prediction_list: List of prediction text segments
        ground_truth: Ground truth answer
        extract_answer: Whether to extract answer from <answer> tags
        extra_info: Optional dict with extraction settings
        
    Returns:
        Tuple of (reward, analysis) where:
        - reward: 1.0 for correct, 0.0 for incorrect
        - analysis: Explanation text from the judge
    """
    extracted, prediction_text = _prediction_text(prediction_list, extract_answer, extra_info)
    if not extracted:
        return 0.0, "Answer extraction failed"

    try:
        score_str, response_text = _call_judge(
            _evaluation_prompt(question, ground_truth, prediction_text)
        )
    except Exception as e:
        print(f"[evaluate_answer] Error calling LLM judge: {e}")
        return 0.0, f"Evaluation API error: {str(e)}"
    
    if not score_str:
        return 0.0, "Evaluation failed: Empty response from API"
    
    try:
        return _reward_from_score(score_str), _analysis_from_response(response_text)
        
    except Exception as e:
        print(f"[evaluate_answer] Error processing response: {e}")
        return 0.0, f"Error processing evaluation response: {str(e)}"


def compute_score(
    prompt: str, 
    predict_str_list: List[str], 
    ground_truth: str, 
    extra_info: Optional[Dict] = None
) -> Tuple[float, str]:
    """
    Compute accuracy score and return analysis text.
    
    Args:
        prompt: Question text
        predict_str_list: List of prediction strings
        ground_truth: Ground truth answer
        extra_info: Optional dict with evaluation settings:
            - gpt_extract_answer: bool, whether to extract from tags
            - extract_answer_tags: str, extraction mode
            - acc_reward_weight: float, weight for accuracy score
        
    
    Returns:
        Tuple of (accuracy_score, analysis) where:
        - accuracy_score: float, weighted accuracy (0.0 or weight*1.0)
        - analysis: str, evaluation explanation
    """
    extra = extra_info or {}
    extract_answer = extra.get("gpt_extract_answer", False)
    acc_weight = extra.get('acc_reward_weight', 1.0)
    
    try:
        reward, analysis = evaluate_answer(
            question=prompt,
            prediction_list=predict_str_list,
            ground_truth=ground_truth,
            extract_answer=extract_answer,
            extra_info=extra
        )
        
        # Validate reward type
        if not isinstance(reward, (int, float)):
            print(f"[compute_score] Warning: Invalid reward type: {type(reward)}, defaulting to 0.0")
            reward = 0.0
            if not analysis:
                analysis = f"Invalid reward value: {reward}"
        
        accuracy_score = acc_weight * float(reward)
        return accuracy_score, analysis
        
    except Exception as e:
        print(f"[compute_score] Error: {e}")
        return 0.0, f"Evaluation error: {str(e)}"
