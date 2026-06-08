"""Embedding retrieval, optional task decomposition, and experience rewrite."""

import os
import json
import hashlib
import time
import re
import threading
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from utils.json_utils import parse_llm_json_list, parse_llm_json_object

import logging
logger = logging.getLogger(__name__)

from .config import EmbeddingSettings


_EMBEDDING_SETTINGS = EmbeddingSettings.from_env()
MAX_RETRIES = _EMBEDDING_SETTINGS.max_retries
API_TIMEOUT = _EMBEDDING_SETTINGS.timeout
EMBEDDING_MAX_CONCURRENCY = _EMBEDDING_SETTINGS.max_concurrency
BATCH_SIZE = _EMBEDDING_SETTINGS.batch_size
TOP_K_RETRIEVE = 3
_EMBEDDING_SEMAPHORE = threading.BoundedSemaphore(EMBEDDING_MAX_CONCURRENCY)
MAX_EMBEDDING_INPUT_CHARS = _EMBEDDING_SETTINGS.max_input_chars
RETRY_HTTP_400 = _EMBEDDING_SETTINGS.retry_http_400


def _prepare_embedding_text(text: Any) -> str:
    value = str(text or "").replace("\x00", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return "[empty]"
    return value[:MAX_EMBEDDING_INPUT_CHARS]

MIN_SIMILARITY = 0.0

MAX_TOKENS_TASK_DECOMPOSITION = 2048
MAX_TOKENS_REWRITE = 8192

TEMPERATURE_TASK_DECOMPOSITION = 0.3
TEMPERATURE_REWRITE = 0.3
TOP_P_DEFAULT = 1.0


from prompts.experience_retrieval_templates import (
    EXPERIENCE_REWRITE_PROMPT,
    EXPERIENCE_REWRITE_SYSTEM_PROMPT,
    RETRIEVAL_IMAGE_DESCRIPTION_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
)
from .schema import (
    format_experience_for_prompt,
    get_experience_content,
    normalize_experience_library,
)


def _build_multimodal_user_content(prompt: str, images: List[Any]) -> List[dict]:
    """Build user content list for multimodal LLM call (text + image_url items)."""
    from PIL import Image
    from utils.image_utils import pil_to_base64_data_uri
    content = [{"type": "text", "text": prompt}]
    for img in images:
        if isinstance(img, Image.Image):
            content.append({
                "type": "image_url",
                "image_url": {"url": pil_to_base64_data_uri(img)}
            })
    return content


def _retrieval_info(
    *,
    original_query: str,
    decomposition_used: bool,
    subtasks: Optional[List[Dict[str, Any]]] = None,
    retrieved_experiences: Optional[List[str]] = None,
    retrieval_details: Optional[List[Dict[str, Any]]] = None,
    similarity_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    retrieved = list(retrieved_experiences or [])
    return {
        "original_query": original_query,
        "decomposition_used": decomposition_used,
        "subtasks": list(subtasks or []),
        "retrieved_experiences": retrieved,
        "retrieval_details": list(retrieval_details or []),
        "total_unique_experiences": len(retrieved),
        "similarity_scores": dict(similarity_scores or {}),
    }


class ExperienceRetriever:
    """Retrieves canonical experiences by embedding similarity."""
    
    def __init__(
        self,
        experiences: Dict[str, Any],
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        llm_client: Optional[Any] = None,  # ExperienceLLM instance for task decomposition
        experience_library_path: Optional[str] = None,  # Path to experience library JSON file
    ):
        self.experiences = normalize_experience_library(experiences)
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.llm_client = llm_client
        self.experience_library_path = experience_library_path
        
        self._last_retrieval_info: Optional[Dict[str, Any]] = None
        
        self.embedding_api_key = (
            embedding_api_key 
            or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        
        self.embedding_endpoint = (
            embedding_endpoint
            or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT")
            or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        
        if not self.embedding_endpoint.endswith("/v1"):
            self.embedding_endpoint = self.embedding_endpoint.rstrip("/") + "/v1"
        
        if not self.embedding_api_key:
            raise ValueError(
                "Embedding API key is required. Set EXPERIENCE_EMBEDDING_API_KEY "
                "or OPENAI_API_KEY environment variable, or pass embedding_api_key parameter."
            )
        
        self._experience_embeddings: Dict[str, np.ndarray] = {}
        
        self._cache_embeddings_path: Optional[str] = None
        self._cache_metadata_path: Optional[str] = None

        self._configure_cache_paths()
        self._load_or_generate_embeddings()

    def _configure_cache_paths(self) -> None:
        if not self.enable_cache:
            return

        lib_hash = self._compute_library_hash()
        cache_base_dir = self.cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "embeddings",
        )
        if self.experience_library_path:
            library_name = os.path.basename(self.experience_library_path)
            if library_name.endswith(".json"):
                library_name = library_name[:-5]
        else:
            library_name = lib_hash[:8]

        cache_subdir = os.path.join(cache_base_dir, library_name)
        os.makedirs(cache_subdir, exist_ok=True)
        cache_stem = f"experience_embeddings_{lib_hash[:8]}"
        self._cache_embeddings_path = os.path.join(cache_subdir, f"{cache_stem}.npy")
        self._cache_metadata_path = os.path.join(cache_subdir, f"{cache_stem}_metadata.json")
    
    def _compute_library_hash(self) -> str:
        """Compute hash of text that is actually embedded."""
        sorted_exps = [
            (exp_id, get_experience_content(record))
            for exp_id, record in sorted(self.experiences.items())
        ]
        content = json.dumps(sorted_exps, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _embedding_request_context(self, is_batch: bool) -> Tuple[str, Dict[str, str], str]:
        url = f"{self.embedding_endpoint}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.embedding_api_key}",
            "Content-Type": "application/json",
        }
        prefix = "Batch embedding" if is_batch else "Embedding"
        return url, headers, prefix

    def _log_embedding_timeout(self, prefix: str, url: str, exc: Exception, attempt: int, max_retries: int) -> bool:
        if attempt >= max_retries - 1:
            logger.error(
                f"Failed to generate {prefix.lower()}(s) after {max_retries} attempts (timeout): {exc}"
            )
            print(f"  Error: {prefix} API timeout. Check endpoint: {url}")
            return False
        wait_time = 2 ** attempt
        logger.warning(
            f"{prefix} API timeout (attempt {attempt + 1}/{max_retries}): {exc}. "
            f"Retrying in {wait_time}s..."
        )
        time.sleep(wait_time)
        return True

    def _handle_embedding_request_error(
        self,
        prefix: str,
        url: str,
        exc: requests.exceptions.RequestException,
        attempt: int,
        max_retries: int,
    ) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        try:
            response_text = str(response.text or "")[:500] if response is not None else ""
        except Exception:
            response_text = ""

        if status_code in {401, 403, 404} or (status_code == 400 and not RETRY_HTTP_400):
            logger.error(
                f"{prefix} API request failed with non-retryable status {status_code}: {exc}; "
                f"body={response_text}"
            )
            print(
                f"  Error: {prefix} API non-retryable failure. "
                f"Check endpoint/key: {url}, status: {status_code}"
            )
            return False

        if attempt >= max_retries - 1:
            logger.error(
                f"Failed to generate {prefix.lower()}(s) after {max_retries} attempts: "
                f"{exc}; body={response_text}"
            )
            print(f"  Error: {prefix} API request failed. Check endpoint: {url}, status: {status_code or 'N/A'}")
            return False

        wait_time = 2 ** attempt
        logger.warning(
            f"{prefix} API request failed (attempt {attempt + 1}/{max_retries}): {exc}. "
            f"Retrying in {wait_time}s..."
        )
        if status_code == 400 and response_text:
            print(f"  Warning: {prefix} API 400 body: {response_text}")
        time.sleep(wait_time)
        return True

    def _post_embedding_request(self, payload: dict, max_retries: int = MAX_RETRIES, is_batch: bool = False) -> Optional[dict]:
        """POST to the embedding API with bounded retries."""
        url, headers, prefix = self._embedding_request_context(is_batch)
        for attempt in range(max_retries):
            try:
                with _EMBEDDING_SEMAPHORE:
                    response = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                if not self._log_embedding_timeout(prefix, url, e, attempt, max_retries):
                    return None
            except requests.exceptions.RequestException as e:
                if not self._handle_embedding_request_error(prefix, url, e, attempt, max_retries):
                    return None
            except Exception as e:
                logger.error(f"Unexpected error generating {prefix.lower()}(s): {e}")
                print(f"  Error: Unexpected error: {e}")
                return None
        return None

    def _generate_embedding(self, text: str, max_retries: int = MAX_RETRIES) -> Optional[np.ndarray]:
        payload = {"model": self.embedding_model, "input": _prepare_embedding_text(text)}
        data = self._post_embedding_request(payload, max_retries=max_retries, is_batch=False)
        if not data:
            return None
        embedding = data.get("data", [{}])[0].get("embedding") if data.get("data") else None
        if embedding:
            return np.array(embedding, dtype=np.float32)
        logger.warning("Empty embedding response")
        logger.debug(f"Response data: {data}")
        return None
    
    def _generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
    ) -> List[Optional[np.ndarray]]:
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = [_prepare_embedding_text(text) for text in texts[i:i + batch_size]]
            batch_num = i // batch_size + 1
            
            batch_embeddings = self._generate_embeddings_batch_api(batch_texts, max_retries=max_retries)
            
            all_embeddings.extend(batch_embeddings)
            if total_batches > 1:
                successful = sum(1 for e in batch_embeddings if e is not None)
                print(f"  Progress: batch {batch_num}/{total_batches} ({successful}/{len(batch_texts)} successful)")
        return all_embeddings
    
    def _generate_embeddings_batch_api(self, texts: List[str], max_retries: int = MAX_RETRIES) -> List[Optional[np.ndarray]]:
        if not texts:
            return []
        prepared_texts = [_prepare_embedding_text(text) for text in texts]
        payload = {"model": self.embedding_model, "input": prepared_texts}
        data = self._post_embedding_request(payload, max_retries=max_retries, is_batch=True)
        if not data:
            if len(prepared_texts) == 1:
                return [self._generate_embedding(prepared_texts[0], max_retries=max_retries)]
            logger.warning("Batch embedding failed; falling back to per-text embedding requests")
            print("  Batch embedding failed; falling back to per-text embedding requests")
            return [self._generate_embedding(text, max_retries=max_retries) for text in prepared_texts]
        embeddings_data = data.get("data", [])
        batch_embeddings = []
        for item in embeddings_data:
            emb = item.get("embedding")
            batch_embeddings.append(np.array(emb, dtype=np.float32) if emb else None)
        while len(batch_embeddings) < len(texts):
            batch_embeddings.append(None)
        return batch_embeddings[:len(texts)]
    
    def _load_embeddings_from_cache(self) -> bool:
        """Load embeddings from disk cache if available and valid."""
        if not self.enable_cache or not self._cache_embeddings_path:
            return False
        
        if not os.path.exists(self._cache_embeddings_path):
            return False
        
        if not os.path.exists(self._cache_metadata_path):
            logger.warning("Embedding cache file exists but metadata missing, regenerating...")
            return False
        
        try:
            with open(self._cache_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            current_hash = self._compute_library_hash()
            if metadata.get("library_hash") != current_hash:
                logger.info("Experience library changed, cache invalid, regenerating...")
                return False
            
            if metadata.get("model") != self.embedding_model:
                logger.info("Embedding model changed, cache invalid, regenerating...")
                return False
            
            embeddings_dict = np.load(self._cache_embeddings_path, allow_pickle=True).item()
            exp_ids = metadata.get("experience_ids", [])
            
            self._experience_embeddings = {
                exp_id: np.array(embeddings_dict[exp_id], dtype=np.float32)
                for exp_id in exp_ids
                if exp_id in embeddings_dict and exp_id in self.experiences
            }
            
            logger.info(
                f"Loaded {len(self._experience_embeddings)} embeddings from cache "
                f"({self._cache_embeddings_path})"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load embeddings from cache: {e}, regenerating...")
            return False
    
    def _save_embeddings_to_cache(self):
        """Save embeddings to disk cache."""
        if not self.enable_cache or not self._cache_embeddings_path:
            return
        
        try:
            embeddings_dict = {
                exp_id: emb.tolist()
                for exp_id, emb in self._experience_embeddings.items()
            }
            
            np.save(self._cache_embeddings_path, embeddings_dict, allow_pickle=True)
            
            metadata = {
                "library_hash": self._compute_library_hash(),
                "model": self.embedding_model,
                "experience_ids": list(self._experience_embeddings.keys()),
                "dimension": (
                    len(list(self._experience_embeddings.values())[0])
                    if self._experience_embeddings else 0
                ),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            
            with open(self._cache_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"Saved {len(self._experience_embeddings)} embeddings to cache "
                f"({self._cache_embeddings_path})"
            )
            
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {e}")
    
    def _load_or_generate_embeddings(self):
        """Load embeddings from cache or generate new ones."""
        if self._load_embeddings_from_cache():
            print(f"Loaded {len(self._experience_embeddings)} embeddings from cache")
            return
        
        total_exps = len(self.experiences)
        print(f"\nGenerating embeddings for {total_exps} experiences using model: {self.embedding_model}")
        print("This may take a while for the first time...")
        
        exp_ids = list(self.experiences.keys())
        exp_texts = [get_experience_content(self.experiences[exp_id]) for exp_id in exp_ids]
        
        start_time = time.time()
        embeddings = self._generate_embeddings_batch(exp_texts, batch_size=BATCH_SIZE)
        elapsed = time.time() - start_time
        
        successful_count = 0
        for exp_id, emb in zip(exp_ids, embeddings):
            if emb is not None:
                self._experience_embeddings[exp_id] = emb
                successful_count += 1
            else:
                logger.warning(f"Failed to generate embedding for experience {exp_id}")
        
        print(f"Generated {successful_count}/{total_exps} embeddings in {elapsed:.1f}s")
        
        if self._experience_embeddings:
            self._save_embeddings_to_cache()
            print(f"Cached embeddings for future use")
    
    def update_experiences(self, new_experiences: Dict[str, str], incremental: bool = True):
        """Update the library and regenerate affected embeddings."""
        if incremental:
            new_exp_ids = []
            updated_exp_ids = []
            
            normalized_new = normalize_experience_library(new_experiences)
            for exp_id, exp_text in normalized_new.items():
                if exp_id not in self.experiences:
                    new_exp_ids.append(exp_id)
                elif self.experiences[exp_id] != exp_text:
                    updated_exp_ids.append(exp_id)
            
            deleted_exp_ids = set(self.experiences.keys()) - set(normalized_new.keys())
            for exp_id in deleted_exp_ids:
                if exp_id in self._experience_embeddings:
                    del self._experience_embeddings[exp_id]
            
            self.experiences = dict(normalized_new)
            
            exp_ids_to_embed = new_exp_ids + updated_exp_ids
            if exp_ids_to_embed:
                print(f"  Updating embeddings for {len(exp_ids_to_embed)} experiences ({len(new_exp_ids)} new, {len(updated_exp_ids)} updated)...")
                exp_texts = [get_experience_content(self.experiences[exp_id]) for exp_id in exp_ids_to_embed]
                embeddings = self._generate_embeddings_batch(exp_texts, batch_size=BATCH_SIZE)
                
                for exp_id, emb in zip(exp_ids_to_embed, embeddings):
                    if emb is not None:
                        self._experience_embeddings[exp_id] = emb
                    else:
                        logger.warning(f"Failed to generate embedding for experience {exp_id}")
                
                if self._experience_embeddings:
                    self._save_embeddings_to_cache()
                    print(f"  Updated embeddings cache")
        else:
            self.experiences = normalize_experience_library(new_experiences)
            self._experience_embeddings.clear()
            self._load_or_generate_embeddings()
        
    def get_last_retrieval_info(self) -> Optional[Dict[str, Any]]:
        return self._last_retrieval_info

    def _query_with_visual_context(self, query: str, images: Optional[List[Any]]) -> str:
        if images and self.llm_client:
            visual_ctx = self._describe_images_for_query(images)
            if visual_ctx:
                return f"{visual_ctx} {query}"
        return query

    def _score_experiences(self, query_embedding: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
        exp_ids = list(self._experience_embeddings.keys())
        exp_matrix = np.vstack([self._experience_embeddings[eid] for eid in exp_ids])
        sim_scores = cosine_similarity(query_embedding.reshape(1, -1), exp_matrix)[0]
        return exp_ids, exp_matrix, sim_scores

    def _select_top_similarities(
        self,
        exp_ids: List[str],
        exp_matrix: np.ndarray,
        sim_scores: np.ndarray,
        top_k: int,
        min_similarity: float,
        mmr_lambda: float,
    ) -> List[Tuple[str, float]]:
        if mmr_lambda < 1.0 and len(exp_ids) > top_k:
            return self._mmr_select(exp_ids, exp_matrix, sim_scores, top_k, mmr_lambda, min_similarity)

        similarities = [
            (exp_ids[i], float(sim_scores[i]))
            for i in range(len(exp_ids))
            if sim_scores[i] >= min_similarity
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _remember_direct_retrieval(self, retrieval_info: Dict[str, Any]) -> None:
        if not self._last_retrieval_info or not self._last_retrieval_info.get("decomposition_used", False):
            self._last_retrieval_info = retrieval_info.copy()

    def _log_direct_retrieval(self, result: Dict[str, Any], top_similarities: List[Tuple[str, float]]) -> None:
        if result:
            logger.debug(
                f"Retrieved {len(result)} experiences for query "
                f"(similarities: {[f'{s:.3f}' for _, s in top_similarities[:3]]}...)"
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        min_similarity: float = MIN_SIMILARITY,
        images: Optional[List[Any]] = None,
        mmr_lambda: float = 0.7,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Retrieve top-k experiences for a query using MMR for diversity."""
        empty_info = _retrieval_info(original_query=query, decomposition_used=False)
        if not self._experience_embeddings:
            logger.warning("No experience embeddings available, returning empty result")
            return {}, empty_info

        query = self._query_with_visual_context(query, images)
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding, returning empty result")
            return {}, empty_info

        exp_ids, exp_matrix, sim_scores = self._score_experiences(query_embedding)
        top_similarities = self._select_top_similarities(
            exp_ids, exp_matrix, sim_scores, top_k, min_similarity, mmr_lambda
        )
        result = {exp_id: self.experiences[exp_id] for exp_id, _ in top_similarities}
        retrieval_info = _retrieval_info(
            original_query=query,
            decomposition_used=False,
            retrieved_experiences=list(result.keys()),
            similarity_scores={exp_id: sim for exp_id, sim in top_similarities},
        )

        self._remember_direct_retrieval(retrieval_info)
        self._log_direct_retrieval(result, top_similarities)
        return result, retrieval_info

    def _retrieve_direct_and_remember(
        self,
        query: str,
        top_k: int,
        min_similarity: float,
        images: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        result, retrieval_info = self.retrieve(
            query, top_k=top_k, min_similarity=min_similarity, images=images
        )
        self._last_retrieval_info = retrieval_info.copy()
        return result, retrieval_info

    def _retrieve_subtask(
        self,
        subtask: Dict[str, Any],
        top_k: int,
        min_similarity: float,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        query = str(subtask.get("query") or "").strip()
        if not query:
            return {}, {}
        result, sub_info = self.retrieve(query=query, top_k=top_k, min_similarity=min_similarity)
        detail = {
            "subtask_type": subtask.get("type", "unknown"),
            "query": query,
            "retrieved_experience_ids": list(result.keys()),
            "count": len(result),
        }
        return result, {"detail": detail, "similarity_scores": sub_info.get("similarity_scores", {})}

    def _retrieve_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
        subtask_top_k: int,
        min_similarity: float,
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, float]]:
        all_results: Dict[str, str] = {}
        retrieval_details: List[Dict[str, Any]] = []
        all_similarity_scores: Dict[str, float] = {}

        for subtask in subtasks:
            result, sub_info = self._retrieve_subtask(subtask, subtask_top_k, min_similarity)
            if not result and not sub_info:
                continue
            retrieval_details.append(sub_info["detail"])
            self._merge_subtask_results(all_results, all_similarity_scores, result, sub_info)

        return all_results, retrieval_details, all_similarity_scores

    def _merge_subtask_results(
        self,
        all_results: Dict[str, str],
        all_similarity_scores: Dict[str, float],
        result: Dict[str, str],
        sub_info: Dict[str, Any],
    ) -> None:
        sub_scores = sub_info.get("similarity_scores", {})
        for exp_id, exp_text in result.items():
            if exp_id not in all_results:
                all_results[exp_id] = exp_text
            all_similarity_scores[exp_id] = max(
                all_similarity_scores.get(exp_id, 0.0),
                sub_scores.get(exp_id, 0.0),
            )

    def _log_decomposition_result(self, all_results: Dict[str, str], subtasks: List[Dict[str, Any]], subtask_top_k: int) -> None:
        if all_results:
            print(f"  [Final] Merged {len(all_results)} unique experiences from {len(subtasks)} subtasks")
            logger.info(
                f"Retrieved {len(all_results)} unique experiences "
                f"from {len(subtasks)} subtasks (each retrieved up to {subtask_top_k} experiences)"
            )
        else:
            print(f"  [Final] No experiences retrieved from any subtask")

    def _mmr_select(
        self,
        exp_ids: List[str],
        exp_matrix: np.ndarray,
        sim_scores: np.ndarray,
        top_k: int,
        lambda_param: float,
        min_sim: float,
    ) -> List[Tuple[str, float]]:
        """Maximal Marginal Relevance: balance relevance with diversity."""
        candidates = [i for i in range(len(exp_ids)) if sim_scores[i] >= min_sim]
        if not candidates:
            return []
        selected: List[Tuple[str, float]] = []
        selected_indices: List[int] = []
        for _ in range(min(top_k, len(candidates))):
            best_idx, best_score = -1, -float('inf')
            for i in candidates:
                if i in selected_indices:
                    continue
                if not selected_indices:
                    mmr = float(sim_scores[i])
                else:
                    sel_matrix = exp_matrix[selected_indices]
                    max_sim_to_selected = float(
                        cosine_similarity(exp_matrix[i:i+1], sel_matrix).max()
                    )
                    mmr = lambda_param * float(sim_scores[i]) - (1 - lambda_param) * max_sim_to_selected
                if mmr > best_score:
                    best_score, best_idx = mmr, i
            if best_idx < 0:
                break
            selected.append((exp_ids[best_idx], float(sim_scores[best_idx])))
            selected_indices.append(best_idx)
        return selected

    def _describe_images_for_query(self, images: List[Any]) -> str:
        """Generate a short visual description of images to enrich the retrieval query."""
        try:
            user_content = _build_multimodal_user_content(RETRIEVAL_IMAGE_DESCRIPTION_PROMPT, images[:1])
            resp = self.llm_client._call_api(
                user_content=user_content,
                max_tokens=128,
                temperature=0.1,
                top_p=1.0,
                api_name="Experience API",
                require_chat_completions=True,
                return_placeholder_on_error=False,
            )
            return resp.strip() if resp else ""
        except Exception as e:
            logger.debug(f"Visual description failed: {e}")
            return ""
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded embeddings."""
        return {
            "total_experiences": len(self.experiences),
            "embedded_count": len(self._experience_embeddings),
            "missing_count": len(self.experiences) - len(self._experience_embeddings),
            "embedding_model": self.embedding_model,
            "cache_enabled": self.enable_cache,
            "cache_path": self._cache_embeddings_path,
        }
    
    def _decompose_task_llm(self, task_description: str, images: Optional[List[Any]] = None) -> List[Dict[str, str]]:
        """Ask the LLM for retrieval subtasks shaped as {type, query} objects."""
        if not self.llm_client:
            raise ValueError("LLM client is required for task decomposition. Pass llm_client to ExperienceRetriever.__init__")
        
        prompt = TASK_DECOMPOSITION_PROMPT.format(task_description=task_description)
        
        for attempt in range(MAX_RETRIES):
            try:
                if images:
                    user_content = _build_multimodal_user_content(prompt, images)
                    response = self.llm_client._call_api(
                        user_content=user_content,
                        max_tokens=MAX_TOKENS_TASK_DECOMPOSITION,
                        temperature=TEMPERATURE_TASK_DECOMPOSITION,
                        top_p=TOP_P_DEFAULT,
                        api_name="Experience API",
                        require_chat_completions=True
                    )
                else:
                    response = self.llm_client.chat(prompt, max_tokens=MAX_TOKENS_TASK_DECOMPOSITION, temperature=TEMPERATURE_TASK_DECOMPOSITION)
                
                subtasks = parse_llm_json_list(response.strip())
                if subtasks and all(
                    isinstance(item, dict) and "type" in item and "query" in item
                    for item in subtasks
                ):
                    logger.debug(f"Decomposed task into {len(subtasks)} subtasks: {[s['type'] for s in subtasks]}")
                    return subtasks
                else:
                    raise ValueError("LLM returned invalid subtask format")
                    
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Task decomposition failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(f"Task decomposition failed after {MAX_RETRIES} attempts: {e}, falling back to single query")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"LLM decomposition failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(f"LLM decomposition failed after {MAX_RETRIES} attempts: {e}, falling back to single query")
        
        return [{
            "type": "general",
            "query": f"Methodology for handling this type of task: {task_description}"
        }]
    
    def retrieve_with_decomposition(
        self,
        task_description: str,
        top_k: int = TOP_K_RETRIEVE,
        min_similarity: float = MIN_SIMILARITY,
        subtask_top_k: Optional[int] = None,
        images: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Retrieve by LLM task decomposition, then merge unique subtask results."""
        if not self.llm_client:
            raise ValueError(
                "LLM client is required for decomposition-based retrieval. "
                "Pass llm_client to ExperienceRetriever.__init__ or use retrieve() instead"
            )

        if len(self.experiences) <= 20:
            print(f"  [Decomposition] Library size {len(self.experiences)} <= 20, using direct MMR retrieval")
            return self._retrieve_direct_and_remember(task_description, top_k, min_similarity, images)

        if subtask_top_k is None:
            subtask_top_k = top_k

        print(f"  [Decomposition] Decomposing task into subtasks...")
        subtasks = self._decompose_task_llm(task_description, images=images)

        if not subtasks:
            print(f"  [Decomposition] Warning: Task decomposition returned no subtasks, falling back to direct retrieval")
            logger.warning("Task decomposition returned no subtasks, falling back to direct retrieval")
            return self._retrieve_direct_and_remember(task_description, top_k, min_similarity, images)

        all_results, retrieval_details, all_similarity_scores = self._retrieve_subtasks(
            subtasks, subtask_top_k, min_similarity
        )

        retrieval_info = _retrieval_info(
            original_query=task_description,
            decomposition_used=True,
            subtasks=subtasks,
            retrieved_experiences=list(all_results.keys()),
            retrieval_details=retrieval_details,
            similarity_scores=all_similarity_scores,
        )
        
        self._last_retrieval_info = retrieval_info.copy()
        self._log_decomposition_result(all_results, subtasks, subtask_top_k)
        return all_results, retrieval_info


def _rewrite_prompt(experiences: Dict[str, str], task_description: str) -> str:
    experiences_text = "\n\n".join(
        f"[{exp_id}]\n{format_experience_for_prompt(exp_id, exp_text)}"
        for exp_id, exp_text in experiences.items()
    )
    return EXPERIENCE_REWRITE_PROMPT.format(
        task_description=task_description,
        experiences_text=experiences_text,
    )


def _call_rewrite_llm(llm_client: Any, prompt: str, images: Optional[List[Any]]) -> str:
    if images:
        return llm_client._call_api(
            user_content=_build_multimodal_user_content(prompt, images),
            max_tokens=MAX_TOKENS_REWRITE,
            temperature=TEMPERATURE_REWRITE,
            top_p=TOP_P_DEFAULT,
            system_prompt=EXPERIENCE_REWRITE_SYSTEM_PROMPT,
            api_name="Experience Rewrite API",
            require_chat_completions=True,
            return_placeholder_on_error=False,
        )
    return llm_client.chat(prompt, max_tokens=MAX_TOKENS_REWRITE, temperature=TEMPERATURE_REWRITE)


def rewrite_experiences_for_task(
    experiences: Dict[str, str],
    task_description: str,
    llm_client: Any,
    images: Optional[List[Any]] = None,
    force_rewrite: bool = False,
) -> Dict[str, str]:
    """Rewrite retrieved experiences when explicitly requested."""
    if not experiences or not force_rewrite:
        return experiences
    
    try:
        response = _call_rewrite_llm(llm_client, _rewrite_prompt(experiences, task_description), images).strip()
        rewritten_dict = parse_llm_json_object(response.strip())
        if rewritten_dict:
            original_count = len(experiences)
            rewritten_count = len(rewritten_dict)
            missing_ids = set(experiences.keys()) - set(rewritten_dict.keys())
            
            if missing_ids:
                logger.info(f"Rewrite: Model skipped {len(missing_ids)} experience(s) (from {original_count} to {rewritten_count})")
                print(f"  [Rewrite] Model skipped {len(missing_ids)} experience(s): {', '.join(sorted(missing_ids))}")
            else:
                print(f"  [Rewrite] Successfully rewritten all {rewritten_count} experiences")
            
            return rewritten_dict
        else:
            logger.warning("Rewrite response is not a dictionary, using original experiences")
    
    except Exception as e:
        logger.warning(f"Failed to rewrite experiences: {e}", exc_info=True)
        print(f"  [Rewrite] Warning: Failed to rewrite experiences: {e}, using original experiences")
    
    return experiences
