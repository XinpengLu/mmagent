"""Memory retrieval, prompt injection, and retrieval logging."""

import os
import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from qwen_vl_utils import fetch_image

from experience import (
    ExperienceLLM,
    ExperienceRetriever,
    format_for_prompt,
    load_experiences,
    rewrite_experiences_for_task,
)
from experience.schema import format_experience_for_prompt
from experience.utils import sample_metadata_context, sample_metadata_tags
from skill import (
    allow_unvalidated_recovery_inference,
    collect_linked_experiences,
    experience_failed_validation,
    experience_low_validation_quality,
    experience_unvalidated_recovery,
    filter_experiences_for_inference,
    format_default_tool_plan_for_prompt,
    format_executable_tool_plan_for_prompt,
    format_selected_skills_for_prompt,
    load_skill_index,
    rerank_memory_with_llm,
    retrieve_skills_for_task,
    skill_index_path,
)
from skill.utils import enabled_callable_tools
from engine.api_processors import set_global_prompts as set_processors_prompts
from prompts.skill_injection_templates import SKILL_INJECTION_HEADER
from utils.image_utils import process_image
from utils.env import env_bool, env_float, env_int, safe_float, safe_int
from utils.artifacts import (
    MEMORY_USAGE_FILENAME,
    RETRIEVAL_INFO_FILENAME,
    write_json,
)

logger = logging.getLogger(__name__)


def _candidate_pool_top_k(base_top_k: int, explicit_value: Optional[int], env_name: str) -> int:
    """Return a candidate recall size without changing final injection limits."""
    base = max(1, int(base_top_k or 1))
    if explicit_value is not None:
        return max(base, int(explicit_value or base))

    env_value = os.environ.get(env_name)
    if env_value:
        try:
            return max(base, int(env_value))
        except ValueError:
            pass

    if not env_bool("MEMORY_RETRIEVAL_OVERSAMPLE_FOR_FILTERS", True):
        return base
    multiplier = env_int("MEMORY_RETRIEVAL_CANDIDATE_MULTIPLIER", 2)
    max_pool = env_int("MEMORY_RETRIEVAL_MAX_CANDIDATES_PER_QUERY", 8)
    return min(max_pool, max(base, base * max(1, multiplier)))


def _experience_library_for_skill_retrieval(args, experience_retriever) -> Dict[str, Any]:
    if experience_retriever is not None:
        return experience_retriever.experiences
    if getattr(args, 'experience_library', None):
        return load_experiences(args.experience_library)
    return {}


def _available_experience_ids(experiences: Dict[str, Any]) -> Optional[set]:
    if not experiences:
        return None
    return {
        exp_id
        for exp_id, record in experiences.items()
        if not experience_failed_validation(record)
        and not experience_low_validation_quality(record)
        and (
            not experience_unvalidated_recovery(record)
            or allow_unvalidated_recovery_inference()
        )
    }


def _embedding_endpoint(args) -> str:
    endpoint = (
        getattr(args, 'experience_embedding_endpoint', None)
        or os.environ.get("EXPERIENCE_EMBEDDING_ENDPOINT")
        or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    )
    if endpoint and not endpoint.endswith("/v1"):
        endpoint = endpoint.rstrip("/") + "/v1"
    return endpoint


def _embedding_config(args) -> Dict[str, Any]:
    return {
        "enable_cache": getattr(args, 'experience_embedding_cache_enable', True),
        "cache_dir": os.path.dirname(args.experience_library) if args.experience_library else None,
        "embedding_model": os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_api_key": (
            getattr(args, 'experience_embedding_api_key', None)
            or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
        "embedding_endpoint": _embedding_endpoint(args),
    }


def _llm_client_for_retrieval_features(args) -> Optional[ExperienceLLM]:
    features = []
    if getattr(args, 'experience_retrieval_decomposition', False):
        features.append("task decomposition")
    if getattr(args, 'experience_retrieval_rewrite', False):
        features.append("experience rewrite")
    if not features:
        return None

    print(f"Initializing LLM client for {', '.join(features)}...")
    llm_client = ExperienceLLM()
    print("LLM client initialized.")
    return llm_client


def _print_retriever_header(experience_library: str, experience_count: int) -> None:
    print(f"\n{'='*80}")
    print("Initializing Experience Retriever (retrieval mode enabled)...")
    print(f"{'='*80}\n")
    print(f"Loading experience library: {experience_library}")
    print(f"Total experiences in library: {experience_count}")


def _build_experience_retriever(args, experiences: Dict[str, Any]) -> ExperienceRetriever:
    embedding = _embedding_config(args)
    print("\nInitializing Experience Retriever (this may take a moment for first-time embedding generation)...")
    return ExperienceRetriever(
        experiences=experiences,
        embedding_model=embedding["embedding_model"],
        embedding_api_key=embedding["embedding_api_key"],
        embedding_endpoint=embedding["embedding_endpoint"],
        cache_dir=embedding["cache_dir"],
        enable_cache=embedding["enable_cache"],
        llm_client=_llm_client_for_retrieval_features(args),
        experience_library_path=args.experience_library,
    )


def _print_retriever_stats(args, retriever: ExperienceRetriever) -> None:
    stats = retriever.get_embedding_stats()
    print("Experience Retriever initialized:")
    print(f"  - Total experiences: {stats['total_experiences']}")
    print(f"  - Embedded count: {stats['embedded_count']}")
    print(f"  - Embedding model: {stats['embedding_model']}")
    print(f"  - Cache enabled: {stats['cache_enabled']}")
    if stats['cache_path']:
        print(f"  - Cache path: {stats['cache_path']}")
    if getattr(args, 'experience_retrieval_decomposition', False):
        print("  - Task decomposition: ENABLED")
    if getattr(args, 'experience_retrieval_rewrite', False):
        print("  - Experience rewrite: ENABLED")
    print(f"\n{'='*80}\n")


def _full_injection_prompt(args, experiences: Dict[str, Any]) -> Optional[str]:
    injection_text = format_for_prompt(experiences, max_items=getattr(args, 'experience_max_items', 256))
    if injection_text:
        print(f"Loaded {len(experiences)} experiences for injection (full injection mode).")
        return injection_text
    return None


def initialize_experience_retriever(args):
    """Initialize retrieval mode or return a full memory injection prompt."""
    if not (getattr(args, 'experience_enable', False) and args.experience_library):
        return None, None

    try:
        exps = load_experiences(args.experience_library)
    except Exception as e:
        print(f"Experience injection skipped: {e}")
        return None, None

    use_retrieval = getattr(args, 'experience_retrieval', False)
    if use_retrieval:
        _print_retriever_header(args.experience_library, len(exps))
        try:
            experience_retriever = _build_experience_retriever(args, exps)
            _print_retriever_stats(args, experience_retriever)
            return experience_retriever, None
        except Exception as e:
            print(f"Warning: Failed to initialize Experience Retriever: {e}")
            print("  Falling back to full experience injection mode.")

    return None, _full_injection_prompt(args, exps)


@dataclass
class _MemoryQuery:
    query_text: str
    rerank_query_text: str
    rerank_metadata_context: str
    images: Optional[List[Any]]
    metadata_tags: List[str]


@dataclass
class _MemoryPipelineState:
    retrieved_exps: Dict[str, Any] = field(default_factory=dict)
    retrieval_info: Dict[str, Any] = field(default_factory=dict)
    original_retrieved_exps: Optional[Dict[str, Any]] = None
    direct_retrieved_ids: List[str] = field(default_factory=list)
    linked_experience_ids: List[str] = field(default_factory=list)
    quality_dropped_experience_ids: List[str] = field(default_factory=list)
    skill_candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_skills: List[Dict[str, Any]] = field(default_factory=list)
    skill_retrieval_info: Dict[str, Any] = field(default_factory=dict)
    memory_rerank_info: Dict[str, Any] = field(default_factory=dict)
    all_experiences: Dict[str, Any] = field(default_factory=dict)
    use_rewrite: bool = False
    llm_client: Optional[Any] = None
    mandatory_tool_plan: str = ""
    mandatory_tool_plan_tools: List[str] = field(default_factory=list)
    default_tool_plan_used: bool = False
    mandatory_tool_required: bool = False
    mandatory_tool_required_tools: List[str] = field(default_factory=list)
    direct_tool_plan_experience_ids: List[str] = field(default_factory=list)
    memory_policy_status: str = "none"
    memory_policy_temperature: Optional[float] = None
    full_injection: str = ""


def _build_query(sample, args) -> _MemoryQuery:
    """Build the text, metadata, and images used by memory retrieval."""
    base_query_text = sample.get('problem', sample.get('question', ''))
    metadata_context = (
        sample_metadata_context(sample)
        if os.environ.get("ENABLE_METADATA_RETRIEVAL_CONTEXT", "0") == "1"
        else ""
    )
    query_text = (
        f"{base_query_text}\n\n{metadata_context}"
        if metadata_context else base_query_text
    )
    rerank_metadata_context = sample_metadata_context(sample)
    rerank_query_text = (
        f"{query_text}\n\n{rerank_metadata_context}"
        if rerank_metadata_context and rerank_metadata_context not in query_text
        else query_text
    )
    return _MemoryQuery(
        query_text=query_text,
        rerank_query_text=rerank_query_text,
        rerank_metadata_context=rerank_metadata_context,
        images=_load_images_for_retrieval(sample, args),
        metadata_tags=sample_metadata_tags(sample),
    )


def _load_images_for_retrieval(sample, args) -> Optional[List[Any]]:
    """Load sample images for multimodal experience retrieval/decomposition."""
    image_folder = getattr(args, "image_folder", None)
    if not image_folder:
        return None

    image_paths = sample.get("images", [])
    if not image_paths:
        return None

    images = []
    max_pixels = getattr(args, "max_pixels", 2_000_000)
    min_pixels = getattr(args, "min_pixels", 40_000)
    for image_path in image_paths:
        full_path = os.path.join(image_folder, image_path)
        if not os.path.exists(full_path):
            print(f"  [Retrieval] Warning: Image file not found: {full_path}")
            continue
        try:
            original_image = fetch_image({"image": full_path, "max_pixels": max_pixels})
            images.append(process_image(original_image, max_pixels, min_pixels))
        except Exception as exc:
            logger.warning("Failed to load image from %s: %s", image_path, exc, exc_info=True)
            print(f"  [Retrieval] Warning: Failed to load image from {image_path}: {exc}")

    if not images:
        print("  [Retrieval] Warning: Failed to load any images from original data source")
    return images or None


def _retrieve_experiences(query: _MemoryQuery, args, experience_retriever, question_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Retrieve direct experience candidates for a sample."""
    if experience_retriever is None:
        return {}, {}

    top_k = getattr(args, 'experience_retrieval_top_k', 3)
    subtask_top_k = _candidate_pool_top_k(
        top_k,
        getattr(args, 'experience_retrieval_subtask_top_k', None),
        "EXPERIENCE_RETRIEVAL_SUBTASK_TOP_K",
    )
    if getattr(args, 'experience_retrieval_decomposition', False):
        retrieved_exps, retrieval_info = experience_retriever.retrieve_with_decomposition(
            task_description=query.query_text,
            top_k=top_k,
            min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
            subtask_top_k=subtask_top_k,
            images=query.images,
        )
    else:
        retrieved_exps, retrieval_info = experience_retriever.retrieve(
            query=query.query_text,
            top_k=subtask_top_k,
            min_similarity=getattr(args, 'experience_retrieval_min_similarity', 0.0),
            images=query.images,
        )

    if not retrieved_exps:
        print(f"  [Retrieval] No relevant experiences found for {question_id}")
    return retrieved_exps or {}, retrieval_info or {}


def _retrieve_skills(query: _MemoryQuery, args, experience_retriever, retrieval_info: Dict[str, Any], question_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Retrieve modular skill candidates and the experience library they link to."""
    skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
    if not skill_enabled:
        return [], {}, {}

    skill_index = load_skill_index(args.skill_library)
    modular_index_exists = os.path.exists(skill_index_path(args.skill_library))
    has_modular_skill_index = modular_index_exists or bool(skill_index.get("skills"))

    all_experiences = _experience_library_for_skill_retrieval(args, experience_retriever)

    skill_top_k = getattr(args, 'skill_retrieval_top_k', 3)
    skill_candidate_top_k = _candidate_pool_top_k(
        skill_top_k,
        getattr(args, 'skill_retrieval_candidate_top_k', None),
        "SKILL_RETRIEVAL_CANDIDATE_TOP_K",
    )
    skill_candidates, skill_retrieval_info = retrieve_skills_for_task(
        args.skill_library,
        query.query_text,
        retrieval_info=retrieval_info,
        top_k=skill_candidate_top_k,
        min_similarity=getattr(args, 'skill_retrieval_min_similarity', 0.0),
        available_experience_ids=_available_experience_ids(all_experiences),
        query_metadata_tags=query.metadata_tags,
        embedding_model=os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_api_key=(
            getattr(args, 'experience_embedding_api_key', None)
            or os.environ.get("EXPERIENCE_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
        embedding_endpoint=_embedding_endpoint(args),
        enable_cache=getattr(args, 'experience_embedding_cache_enable', True),
    )
    skill_candidates = skill_candidates or []
    if skill_candidates:
        candidate_skill_ids = [skill["skill_id"] for skill in skill_candidates]
        print(f"  [Skill] Retrieved {len(skill_candidates)} modular skill candidate(s): {', '.join(candidate_skill_ids)}")
    elif has_modular_skill_index:
        print(f"  [Skill] No modular skill matched {question_id}; skipping skill injection.")

    return skill_candidates, skill_retrieval_info or {}, all_experiences


def _candidate_experiences_for_rerank(state: _MemoryPipelineState) -> Dict[str, Any]:
    linked_exps = collect_linked_experiences(state.skill_candidates, state.all_experiences)
    state.linked_experience_ids = list(linked_exps.keys())
    candidate_exps = dict(state.retrieved_exps or {})
    for exp_id, record in linked_exps.items():
        candidate_exps.setdefault(exp_id, record)
    return candidate_exps


def _llm_client_for_memory(state: _MemoryPipelineState, experience_retriever) -> Any:
    if state.llm_client is not None:
        return state.llm_client
    state.llm_client = experience_retriever.llm_client if experience_retriever is not None else None
    if state.llm_client is None:
        state.llm_client = ExperienceLLM()
    return state.llm_client


def _selected_linked_experience_ids(selected_skills: List[Dict[str, Any]], all_experiences: Dict[str, Any], retained_ids: set) -> List[str]:
    selected_linked_exps = collect_linked_experiences(selected_skills, all_experiences)
    return [exp_id for exp_id in selected_linked_exps.keys() if exp_id in retained_ids]


def _drop_skills_without_retained_experiences(state: _MemoryPipelineState) -> List[str]:
    if not state.selected_skills:
        return []

    retained_exp_ids = set((state.retrieved_exps or {}).keys())
    kept_skills = []
    dropped_skill_ids = []
    for skill in state.selected_skills:
        skill_id = str(skill.get("skill_id") or "")
        linked_ids = [
            str(exp_id)
            for exp_id in skill.get("metadata", {}).get("linked_experience_ids", []) or []
        ]
        if linked_ids and not any(exp_id in retained_exp_ids for exp_id in linked_ids):
            dropped_skill_ids.append(skill_id)
            continue
        kept_skills.append(skill)
    if dropped_skill_ids:
        state.selected_skills = kept_skills
        state.linked_experience_ids = _selected_linked_experience_ids(
            state.selected_skills,
            state.all_experiences,
            retained_exp_ids,
        )
    return dropped_skill_ids


def _rerank_memory(query: _MemoryQuery, args, state: _MemoryPipelineState, experience_retriever, sample_dir: str, question_id: str) -> None:
    """Use the LLM to select the final skills and direct/linked experiences."""
    if not state.skill_candidates:
        return

    try:
        candidate_exps = _candidate_experiences_for_rerank(state)
        state.selected_skills, state.retrieved_exps, state.memory_rerank_info = rerank_memory_with_llm(
            query_text=query.rerank_query_text,
            candidate_skills=state.skill_candidates,
            candidate_experiences=candidate_exps,
            llm=_llm_client_for_memory(state, experience_retriever),
            max_skills=getattr(args, 'memory_rerank_top_skills', 2),
            max_experiences=getattr(args, 'memory_rerank_top_experiences', 8),
            debug_dir=sample_dir,
            question_id=question_id,
            query_metadata_tags=query.metadata_tags,
        )
        selected_skill_ids = [skill["skill_id"] for skill in state.selected_skills]
        state.linked_experience_ids = _selected_linked_experience_ids(
            state.selected_skills,
            state.all_experiences,
            set((state.retrieved_exps or {}).keys()),
        )
        print(f"  [Memory Rerank] Selected {len(state.selected_skills)} skill(s) and {len(state.retrieved_exps)} experience(s)")
        if selected_skill_ids:
            print(f"  [Skill] Selected modular skill(s): {', '.join(selected_skill_ids)}")
    except Exception as e:
        print(f"  [Skill] Warning: failed to rerank skill-linked memories: {e}")
        state.selected_skills = state.skill_candidates[:getattr(args, 'memory_rerank_top_skills', 2)]
        linked_exps = collect_linked_experiences(state.selected_skills, state.all_experiences)
        state.linked_experience_ids = list(linked_exps.keys())
        state.retrieved_exps = dict(state.retrieved_exps or {})
        for exp_id, record in linked_exps.items():
            if exp_id not in state.retrieved_exps:
                state.retrieved_exps[exp_id] = record

    # Modular skill path keeps structured experiences and skips rewrite/adaptation.
    state.use_rewrite = False


def _filter_and_rewrite_memory(query: _MemoryQuery, args, state: _MemoryPipelineState, experience_retriever, question_id: str) -> None:
    """Apply quality filters and optional LLM rewrite to retrieved experiences."""
    if state.retrieved_exps:
        before_filter_ids = set(state.retrieved_exps.keys())
        state.retrieved_exps = filter_experiences_for_inference(
            state.retrieved_exps,
            allow_failure_only=bool(state.selected_skills),
            similarity_scores=state.retrieval_info.get("similarity_scores", {}),
            query_metadata_tags=query.metadata_tags,
        )
        dropped_ids = sorted(before_filter_ids - set((state.retrieved_exps or {}).keys()))
        if dropped_ids:
            state.quality_dropped_experience_ids = dropped_ids
            print(
                f"  [Memory Quality] Dropped {len(dropped_ids)} weak/failure-only experience(s): "
                + ", ".join(dropped_ids[:8])
                + ("..." if len(dropped_ids) > 8 else "")
            )

    dropped_skill_ids = _drop_skills_without_retained_experiences(state)
    if dropped_skill_ids:
        print(
            "  [Memory Quality] Dropped selected skill(s) without retained linked experiences: "
            + ", ".join(dropped_skill_ids)
        )

    if state.retrieved_exps and state.use_rewrite:
        state.original_retrieved_exps = copy.deepcopy(state.retrieved_exps)
        state.llm_client = experience_retriever.llm_client or ExperienceLLM()
        print(f"  [Retrieval] Rewriting {len(state.retrieved_exps)} experiences...")
        state.retrieved_exps = rewrite_experiences_for_task(
            state.retrieved_exps, query.query_text, state.llm_client, query.images
        )

def reload_experiences(args, base_system_prompt, experience_retriever=None):
    """Reload the memory library after training updates."""
    if not (getattr(args, 'experience_enable', False) and args.experience_library):
        return base_system_prompt
    
    try:
        exps = load_experiences(args.experience_library)
        use_retrieval = getattr(args, 'experience_retrieval', False)

        if use_retrieval and experience_retriever is not None:
            print(f"  Updating experience retriever with {len(exps)} experiences...")
            experience_retriever.update_experiences(exps, incremental=True)
            print(f"  Experience retriever updated successfully.")
            return base_system_prompt

        injection_text = format_for_prompt(exps, max_items=getattr(args, 'experience_max_items', 256))
        if injection_text:
            system_prompt = (injection_text + "\n\n" + base_system_prompt) if base_system_prompt else injection_text
            set_processors_prompts(system_prompt)
            print(f"  Reloaded {len(exps)} experiences for next batch.")
            return system_prompt
    except Exception as e:
        print(f"  Warning: Failed to reload experiences: {e}")
        import traceback
        traceback.print_exc()

    return base_system_prompt


def _direct_tool_plan_score(exp_id: str, record: Dict[str, Any], similarity_scores: Dict[str, float]) -> Optional[float]:
    """Rank standalone executable experiences when no modular skill is selected."""
    if not isinstance(record, dict) or record.get("kind") != "success":
        return None
    if not record.get("tool_sequence"):
        return None
    enabled_tools = enabled_callable_tools()
    has_callable_tool = any(
        str(tool).strip() in enabled_tools
        for tool in record.get("tool_tags", []) or []
    ) or any(
        isinstance(step, dict) and str(step.get("tool") or "").strip() in enabled_tools
        for step in record.get("tool_sequence", []) or []
    )
    if not has_callable_tool:
        return None
    if experience_failed_validation(record) or experience_low_validation_quality(record):
        return None
    if experience_unvalidated_recovery(record) and not allow_unvalidated_recovery_inference():
        return None

    stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    score = safe_float(similarity_scores.get(exp_id), 0.0)
    score += 0.30 * safe_float(stats.get("validation_average_score"), 0.0)
    score += 0.15 * safe_float(stats.get("validation_pass_rate"), 0.0)
    score += 0.02 * min(5, safe_int(stats.get("support_count"), 1))
    score += 0.05 * safe_float(record.get("confidence"), 0.0)
    if stats.get("supervised_correction"):
        score += 0.05
    if stats.get("recovered_from_failure") or record.get("recovery_status"):
        score += 0.02
    return score


def _select_direct_tool_plan_experiences(
    experiences: Dict[str, Any],
    retrieval_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Pick a compact executable policy source from direct retrieval results."""
    if not experiences:
        return {}
    similarity_scores = retrieval_info.get("similarity_scores", {}) if isinstance(retrieval_info, dict) else {}
    max_items = env_int("DIRECT_EXPERIENCE_TOOL_PLAN_MAX_EXPERIENCES", 2)
    scored = []
    for exp_id, record in experiences.items():
        score = _direct_tool_plan_score(exp_id, record, similarity_scores)
        if score is not None:
            scored.append((score, exp_id, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return {exp_id: record for _, exp_id, record in scored[:max(1, max_items)]}


def _plan_has_unverified_policy(state: _MemoryPipelineState, plan_experiences: Dict[str, Any]) -> bool:
    return any(
        str((skill.get("metadata", {}) or {}).get("recovery_status") or "") == "candidate_unverified"
        for skill in state.selected_skills
    ) or any(
        experience_unvalidated_recovery(record)
        for record in (plan_experiences or {}).values()
        if isinstance(record, dict)
    )


def _apply_tool_plan_policy(args, state: _MemoryPipelineState, plan_experiences: Dict[str, Any]) -> None:
    if state.default_tool_plan_used:
        state.memory_policy_status = "default_tool_required"
    elif state.mandatory_tool_plan:
        state.memory_policy_status = "candidate_probe" if _plan_has_unverified_policy(state, plan_experiences) else "validated_policy"
    else:
        state.memory_policy_status = "none"
    if state.memory_policy_status == "validated_policy":
        try:
            state.memory_policy_temperature = env_float("VALIDATED_MEMORY_POLICY_TEMPERATURE", 0.2)
        except ValueError:
            state.memory_policy_temperature = 0.2

    state.mandatory_tool_required = bool(state.mandatory_tool_plan)
    state.mandatory_tool_required_tools = state.mandatory_tool_plan_tools if state.mandatory_tool_plan else []
    args._memory_policy_status = state.memory_policy_status
    args._memory_policy_temperature = state.memory_policy_temperature
    args._mandatory_tool_plan_required = state.mandatory_tool_required
    args._mandatory_tool_plan_tools = state.mandatory_tool_required_tools


def _build_tool_plan(state: _MemoryPipelineState, question_id: str) -> Dict[str, Any]:
    plan_experiences: Dict[str, Any] = {}
    if state.retrieved_exps:
        plan_experiences = state.retrieved_exps or {}
        if not state.selected_skills:
            plan_experiences = _select_direct_tool_plan_experiences(
                state.retrieved_exps or {},
                state.retrieval_info or {},
            )
            state.direct_tool_plan_experience_ids = list(plan_experiences.keys())
        state.mandatory_tool_plan, state.mandatory_tool_plan_tools = format_executable_tool_plan_for_prompt(
            state.selected_skills,
            plan_experiences,
            max_steps=6 if state.selected_skills else env_int("DIRECT_EXPERIENCE_TOOL_PLAN_MAX_STEPS", 3),
        )
        if not state.mandatory_tool_plan:
            state.direct_tool_plan_experience_ids = []

    if not state.mandatory_tool_plan:
        state.mandatory_tool_plan, state.mandatory_tool_plan_tools = format_default_tool_plan_for_prompt()
        if state.mandatory_tool_plan:
            state.default_tool_plan_used = True
            print(f"  [Tool Plan] Added default tool-use plan for {question_id}")
    return plan_experiences


def _build_injection(query: _MemoryQuery, args, state: _MemoryPipelineState, sample_dir: str, question_id: str) -> str:
    """Build the final prompt injection from tool plan, selected skills, and experiences."""
    skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)
    skill_injection = ""
    if skill_enabled and state.selected_skills:
        skill_injection = _process_skill(state.selected_skills, sample_dir, question_id)

    plan_experiences = _build_tool_plan(state, question_id)
    _apply_tool_plan_policy(args, state, plan_experiences)

    exp_injection = format_for_prompt(state.retrieved_exps, max_items=len(state.retrieved_exps)) if state.retrieved_exps else ""
    tool_plan_injection = (
        f"<mandatory_tool_plan>\n{state.mandatory_tool_plan}\n</mandatory_tool_plan>"
        if state.mandatory_tool_plan else ""
    )

    parts = [part for part in (tool_plan_injection, skill_injection, exp_injection) if part]
    state.full_injection = "\n\n".join(parts)
    if tool_plan_injection and skill_injection and exp_injection:
        print(f"  [Injection] Combined mandatory tool plan + skill + experiences for {question_id}")
    elif tool_plan_injection and skill_injection:
        print(f"  [Injection] Combined mandatory tool plan + skill for {question_id}")
    elif tool_plan_injection and exp_injection:
        print(f"  [Injection] Combined mandatory tool plan + experiences for {question_id}")
    elif skill_injection and exp_injection:
        print(f"  [Injection] Combined skill + experiences for {question_id}")

    return state.full_injection


def _write_memory_usage(state: _MemoryPipelineState, query: _MemoryQuery, question_id: str, sample_dir: str) -> None:
    """Write machine-readable memory usage for later analysis and training updates."""
    if not sample_dir:
        return

    action_experience_ids: List[str] = []
    if state.mandatory_tool_plan:
        for exp_id in state.direct_tool_plan_experience_ids + state.linked_experience_ids:
            if exp_id and exp_id not in action_experience_ids:
                action_experience_ids.append(exp_id)

    usage = {
        "question_id": question_id,
        "selected_skill_ids": [skill["skill_id"] for skill in state.selected_skills],
        "selected_skills": [
            {
                "skill_id": skill["skill_id"],
                "score": skill.get("score"),
                "linked_experience_ids": skill.get("metadata", {}).get("linked_experience_ids", []),
            }
            for skill in state.selected_skills
        ],
        "direct_retrieved_experience_ids": state.direct_retrieved_ids,
        "linked_experience_ids": state.linked_experience_ids,
        "quality_dropped_experience_ids": state.quality_dropped_experience_ids,
        "retrieved_experience_ids": list((state.retrieved_exps or {}).keys()),
        "rewrite_used": bool(state.use_rewrite),
        "similarity_scores": state.retrieval_info.get("similarity_scores", {}),
        "skill_retrieval_info": state.skill_retrieval_info,
        "memory_rerank_info": state.memory_rerank_info,
        "rerank_metadata_context": query.rerank_metadata_context,
        "mandatory_tool_plan_required": state.mandatory_tool_required,
        "mandatory_tool_plan_tools": state.mandatory_tool_required_tools,
        "memory_tool_plan_available": bool(state.mandatory_tool_plan),
        "default_tool_plan_used": state.default_tool_plan_used,
        "direct_tool_plan_experience_ids": state.direct_tool_plan_experience_ids,
        "action_experience_ids": action_experience_ids,
        "memory_policy_status": state.memory_policy_status,
        "memory_policy_temperature": state.memory_policy_temperature,
    }
    write_json(os.path.join(sample_dir, MEMORY_USAGE_FILENAME), usage, indent=2)


def retrieve_experiences_for_sample(sample, args, experience_retriever, base_system_prompt,
                                     question_id, sample_dir, update_global_prompts=False):
    """
    Run sample-level memory retrieval and prompt injection.
    Pipeline: build query -> retrieve memories -> rerank -> filter -> inject.
    """
    exp_enabled = experience_retriever is not None
    skill_enabled = getattr(args, 'skill_inference', False) and getattr(args, 'skill_library', None)

    if not (exp_enabled or skill_enabled):
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)

    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)

    try:
        query = _build_query(sample, args)
        state = _MemoryPipelineState(use_rewrite=getattr(args, 'experience_retrieval_rewrite', False))

        state.retrieved_exps, state.retrieval_info = _retrieve_experiences(
            query, args, experience_retriever, question_id
        )
        state.direct_retrieved_ids = list(state.retrieved_exps.keys())

        (
            state.skill_candidates,
            state.skill_retrieval_info,
            state.all_experiences,
        ) = _retrieve_skills(query, args, experience_retriever, state.retrieval_info, question_id)

        _rerank_memory(query, args, state, experience_retriever, sample_dir, question_id)
        _filter_and_rewrite_memory(query, args, state, experience_retriever, question_id)
        full_injection = _build_injection(query, args, state, sample_dir, question_id)

        dynamic_system_prompt = (
            f"{full_injection}\n\n{base_system_prompt}" if full_injection and base_system_prompt
            else (full_injection or base_system_prompt)
        )

        if exp_enabled and sample_dir:
            _save_retrieval_info(
                experience_retriever,
                sample_dir,
                state.retrieved_exps or None,
                state.original_retrieved_exps,
                state.use_rewrite,
                retrieval_info=state.retrieval_info,
            )
        _write_memory_usage(state, query, question_id, sample_dir)

        return _finalize_prompt(args, dynamic_system_prompt, update_global_prompts)

    except Exception as e:
        logger.warning(f"Failed to process for {question_id}: {e}", exc_info=True)
        print(f"  [Pipeline] Warning: {e}")
        return _finalize_prompt(args, base_system_prompt, update_global_prompts)


def _process_skill(selected_skills: List[Dict[str, Any]], sample_dir: str, question_id: str) -> str:
    """Format selected modular skills for prompt injection."""
    if not selected_skills:
        return ""
    try:
        skill_content = format_selected_skills_for_prompt(selected_skills)
        print(f"  [Skill] Injecting selected modular skills for {question_id}")
        if sample_dir:
            with open(os.path.join(sample_dir, "tt_selected_skills.md"), 'w', encoding='utf-8') as f:
                f.write(skill_content)
        return SKILL_INJECTION_HEADER.format(skill_content=skill_content)
    except Exception as e:
        print(f"  [Skill] Warning: {e}")
        return ""


def _finalize_prompt(args, prompt, update_global_prompts):
    """Update global prompts or attach the sample prompt to args."""
    if update_global_prompts:
        set_processors_prompts(prompt)
    else:
        args._sample_system_prompt = prompt
    return prompt


def _write_experience_section(file_obj, title: str, experiences: Dict[str, Any]) -> None:
    file_obj.write(f"\n\n{'=' * 80}\n")
    file_obj.write(f"{title}\n")
    file_obj.write("=" * 80 + "\n")
    for exp_id, exp_text in experiences.items():
        file_obj.write(f"\n[{exp_id}]\n")
        file_obj.write("-" * 80 + "\n")
        file_obj.write(format_experience_for_prompt(exp_id, exp_text))
        file_obj.write("\n\n")


def _write_retrieval_header(file_obj, retrieval_info: Dict[str, Any], rewrite_used: bool) -> None:
    file_obj.write("=" * 80 + "\n")
    file_obj.write("EXPERIENCE RETRIEVAL INFORMATION\n")
    file_obj.write("=" * 80 + "\n\n")
    file_obj.write(f"Original Query:\n{retrieval_info.get('original_query', 'N/A')}\n\n")
    file_obj.write(f"Decomposition Used: {retrieval_info.get('decomposition_used', False)}\n")
    file_obj.write(f"Rewrite Used: {rewrite_used}\n\n")


def _write_retrieval_subtasks(file_obj, retrieval_info: Dict[str, Any]) -> None:
    if not retrieval_info.get('decomposition_used', False):
        return
    file_obj.write(f"Subtasks ({len(retrieval_info.get('subtasks', []))}):\n")
    for i, subtask in enumerate(retrieval_info.get('subtasks', []), 1):
        file_obj.write(f"\n  {i}. Type: {subtask.get('type', 'unknown')}\n")
        file_obj.write(f"     Query: {subtask.get('query', '')}\n")

    file_obj.write(f"\n\nRetrieval Details:\n")
    for i, detail in enumerate(retrieval_info.get('retrieval_details', []), 1):
        file_obj.write(f"\n  {i}. Subtask: {detail.get('subtask_type', 'unknown')}\n")
        file_obj.write(f"     Query: {detail.get('query', '')}\n")
        file_obj.write(f"     Retrieved {detail.get('count', 0)} experiences:\n")
        for exp_id in detail.get('retrieved_experience_ids', []):
            file_obj.write(f"       - {exp_id}\n")


def _write_final_retrieved_ids(file_obj, retrieval_info: Dict[str, Any]) -> None:
    file_obj.write(f"\n\nFinal Retrieved Experiences ({retrieval_info.get('total_unique_experiences', 0)} total):\n")
    for exp_id in retrieval_info.get('retrieved_experiences', []):
        file_obj.write(f"  - {exp_id}\n")


def _save_retrieval_info(experience_retriever, sample_dir, retrieved_exps=None, original_retrieved_exps=None, rewrite_used=False, retrieval_info=None):
    """Save retrieval information to a human-readable per-sample artifact."""
    if retrieval_info is None:
        retrieval_info = experience_retriever.get_last_retrieval_info() if experience_retriever else None
    if not retrieval_info:
        return

    retrieval_info_file = os.path.join(sample_dir, RETRIEVAL_INFO_FILENAME)
    os.makedirs(sample_dir, exist_ok=True)

    with open(retrieval_info_file, 'w', encoding='utf-8') as f:
        _write_retrieval_header(f, retrieval_info, rewrite_used)
        _write_retrieval_subtasks(f, retrieval_info)
        _write_final_retrieved_ids(f, retrieval_info)

        if rewrite_used and original_retrieved_exps:
            _write_experience_section(
                f,
                "ORIGINAL RETRIEVED EXPERIENCES (Before Rewrite)",
                original_retrieved_exps,
            )

        if retrieved_exps:
            section_title = "REWRITTEN EXPERIENCES (After Rewrite)" if rewrite_used else "RETRIEVED EXPERIENCE CONTENT"
            _write_experience_section(f, section_title, retrieved_exps)
        else:
            f.write("\nNo relevant experiences were retrieved for this query.\n")


def prepare_sample_args(args):
    """Create a thread-safe shallow args copy for parallel execution."""
    sample_args = copy.copy(args)
    if hasattr(args, 'tool_configs') and args.tool_configs:
        sample_args.tool_configs = copy.deepcopy(args.tool_configs)
    return sample_args


def get_sample_metadata(sample, sample_idx, output_dir):
    """Extract question_id and output directory for one sample."""
    question_id = sample.get('doc_id', sample.get('question_id', f"sample_{sample_idx}"))
    sample_dir = os.path.join(output_dir, question_id)
    return question_id, sample_dir
