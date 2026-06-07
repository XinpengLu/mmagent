"""Single train/validation/test phase execution."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .context import (
    MainFlowContext,
    _count_json_items,
    _default_prompt_and_tools,
)
from .health import check_phase_health
from utils.env import env_bool, env_float, env_int
from utils.artifacts import AVERAGE_PASS_FILENAME, DATASET_SUMMARY_FILENAME, read_json, write_json


@dataclass(frozen=True)
class _PhasePaths:
    data_path: Path
    image_dir: Path
    output_dir: Path
    log_file: Path


@dataclass(frozen=True)
class _MemoryPaths:
    memory_name: str
    memory_root: Path
    experience_library: Path
    skill_library: Path


@dataclass(frozen=True)
class _ExecutionSettings:
    skip_completed: bool
    temperature: float
    top_p: float
    max_completion_tokens: int
    max_turns: int
    max_images: int
    max_total_tokens: int
    max_pixels: int
    min_pixels: int
    num_workers: int
    rollouts_per_sample: int
    seed_base: int
    system_prompts_path: str
    tool_config_path: str
    image_search_max_calls: int
    web_search_max_calls: int
    code_interpreter_max_calls: int
    experience_max_ops: int
    experience_large_batch: int
    experience_max_items: int
    experience_generation_workers: int
    skill_cluster_max_clusters: int
    skill_cluster_min_experiences: int
    skill_retrieval_top_k: int
    skill_retrieval_min_similarity: float
    memory_rerank_top_skills: int
    memory_rerank_top_experiences: int
    experience_retrieval_top_k: int
    experience_retrieval_min_similarity: float

    @classmethod
    def from_env(cls) -> "_ExecutionSettings":
        num_workers = env_int("NUM_WORKERS", 8)
        return cls(
            skip_completed=env_bool("SKIP_COMPLETED", True),
            temperature=env_float("TEMPERATURE", 0.6),
            top_p=env_float("TOP_P", 1.0),
            max_completion_tokens=env_int("MAX_COMPLETION_TOKENS", 8192),
            max_turns=env_int("MAX_TURNS", 20),
            max_images=env_int("MAX_IMAGES", 50),
            max_total_tokens=env_int("MAX_TOTAL_TOKENS", 32768),
            max_pixels=env_int("MAX_PIXELS", 2000000),
            min_pixels=env_int("MIN_PIXELS", 40000),
            num_workers=num_workers,
            rollouts_per_sample=env_int("ROLLOUTS_PER_SAMPLE", 4),
            seed_base=env_int("SEED_BASE", 42),
            system_prompts_path=os.environ.get("SYSTEM_PROMPTS_PATH", "prompts/system_prompts.yaml"),
            tool_config_path=os.environ.get("TOOL_CONFIG_PATH", "configs/tool_configs.yaml"),
            image_search_max_calls=env_int("IMAGE_SEARCH_MAX_CALLS", 5),
            web_search_max_calls=env_int("WEB_SEARCH_MAX_CALLS", 7),
            code_interpreter_max_calls=env_int("CODE_INTERPRETER_MAX_CALLS", 10),
            experience_max_ops=env_int("EXPERIENCE_MAX_OPS", 3),
            experience_large_batch=env_int("EXPERIENCE_LARGE_BATCH", 8),
            experience_max_items=env_int("EXPERIENCE_MAX_ITEMS", 90),
            experience_generation_workers=env_int("EXPERIENCE_GENERATION_WORKERS", num_workers),
            skill_cluster_max_clusters=env_int("SKILL_CLUSTER_MAX_CLUSTERS", 12),
            skill_cluster_min_experiences=env_int("SKILL_CLUSTER_MIN_EXPERIENCES", 2),
            skill_retrieval_top_k=env_int("SKILL_RETRIEVAL_TOP_K", 3),
            skill_retrieval_min_similarity=env_float("SKILL_RETRIEVAL_MIN_SIMILARITY", 0.0),
            memory_rerank_top_skills=env_int("MEMORY_RERANK_TOP_SKILLS", 2),
            memory_rerank_top_experiences=env_int("MEMORY_RERANK_TOP_EXPERIENCES", 8),
            experience_retrieval_top_k=env_int("EXPERIENCE_RETRIEVAL_TOP_K", 3),
            experience_retrieval_min_similarity=env_float("EXPERIENCE_RETRIEVAL_MIN_SIMILARITY", 0.0),
        )


class _Tee:
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


@contextmanager
def _patched_environ(overrides: dict[str, str]):
    previous = os.environ.copy()
    for key, value in overrides.items():
        os.environ[key] = str(value)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


@contextmanager
def _tee_output(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(previous_stdout, log)
        sys.stderr = _Tee(previous_stderr, log)
        try:
            yield
        finally:
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr


def _phase_path_value(
    phase: str,
    kind_key: str,
    default_value: Path,
    ctx: MainFlowContext,
    phase_env: dict[str, str],
) -> Path:
    phase_key = phase.upper()
    phase_specific = phase_env.get(f"{phase_key}_{kind_key}") or os.environ.get(f"{phase_key}_{kind_key}")
    if phase_specific:
        return Path(phase_specific)

    if ctx.benchmark_count == 1:
        generic = os.environ.get(kind_key)
        if generic:
            return Path(generic)

    return default_value


def _phase_data_path(phase: str, benchmark: str, ctx: MainFlowContext, phase_env: dict[str, str]) -> Path:
    split_file = os.environ.get(f"{phase.upper()}_SPLIT_FILE", f"{phase}.json")
    return _phase_path_value(
        phase,
        "DATA_PATH",
        ctx.split_root / benchmark / split_file,
        ctx,
        phase_env,
    )


def _phase_output_dir(phase: str, benchmark: str, ctx: MainFlowContext, phase_env: dict[str, str]) -> Path:
    return _phase_path_value(
        phase,
        "OUTPUT_DIR",
        ctx.output_root / benchmark / phase,
        ctx,
        phase_env,
    )


def _phase_log_output_dir(phase: str, benchmark: str, ctx: MainFlowContext, phase_env: dict[str, str]) -> Path:
    return _phase_path_value(
        phase,
        "LOG_OUTPUT_DIR",
        ctx.log_root / benchmark / phase,
        ctx,
        phase_env,
    )


def _count_samples_for_phase(phase: str, data_path: Path, phase_env: dict[str, str]) -> int:
    phase_value = phase_env.get(f"MAX_SAMPLES_{phase.upper()}") or os.environ.get(f"MAX_SAMPLES_{phase.upper()}")
    value = phase_value or os.environ.get("MAX_SAMPLES", "")
    return int(value) if value else _count_json_items(data_path)


def _set_phase_env_default(env: dict[str, str], key: str, value: str) -> None:
    if key not in env and key not in os.environ:
        env[key] = value


def _resolve_memory_paths(benchmark: str, ctx: MainFlowContext, phase_env: dict[str, str]) -> _MemoryPaths:
    memory_name = os.environ.get("MEMORY_NAME") or benchmark
    phase_memory_root = phase_env.get("MEMORY_ROOT_OVERRIDE", "")
    memory_root_base = os.environ.get("MEMORY_ROOT_BASE", "")
    default_memory_root = (
        Path(memory_root_base) / memory_name
        if memory_root_base
        else ctx.data_root / "memory_bank" / memory_name
    )
    memory_root = Path(phase_memory_root or os.environ.get("MEMORY_ROOT") or default_memory_root)
    if phase_memory_root:
        experience_library = memory_root / "experiences.json"
        skill_library = memory_root / "skill_library.anchor"
    else:
        experience_library = Path(os.environ.get("EXPERIENCE_LIBRARY") or memory_root / "experiences.json")
        skill_library = Path(os.environ.get("SKILL_LIBRARY") or memory_root / "skill_library.anchor")
    return _MemoryPaths(
        memory_name=memory_name,
        memory_root=memory_root,
        experience_library=experience_library,
        skill_library=skill_library,
    )


def _resolve_phase_paths(
    phase: str,
    benchmark: str,
    ctx: MainFlowContext,
    phase_env: dict[str, str],
) -> _PhasePaths:
    log_output_dir = _phase_log_output_dir(phase, benchmark, ctx, phase_env)
    return _PhasePaths(
        data_path=_phase_data_path(phase, benchmark, ctx, phase_env),
        image_dir=Path(os.environ.get("IMAGE_DIR") or ctx.bench_root),
        output_dir=_phase_output_dir(phase, benchmark, ctx, phase_env),
        log_file=log_output_dir.with_suffix(".log"),
    )


def _write_phase_summary(output_dir: Path, k: int) -> None:
    metrics_path = output_dir / DATASET_SUMMARY_FILENAME
    summary_path = output_dir / AVERAGE_PASS_FILENAME
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found; no pass@k/average@k summary written.")
        return

    metrics = read_json(metrics_path)
    selected = (metrics.get("k_metrics") or {}).get(str(k)) or (metrics.get("k_metrics") or {}).get(k)
    if selected is None:
        print(f"Warning: k={k} not found in {metrics_path}.")
        return

    summary = {
        "k": int(k),
        f"pass@{k}": selected.get("pass_at_k", 0.0),
        f"average@{k}": selected.get("average_at_k", 0.0),
        "metrics_file": str(metrics_path),
    }
    write_json(summary_path, summary, indent=2)
    print(f"Recorded metrics: pass@{k}={summary[f'pass@{k}']:.4f}, average@{k}={summary[f'average@{k}']:.4f}")
    print(f"Summary file: {summary_path}")


def _build_execution_args(
    *,
    phase: str,
    data_path: Path,
    image_dir: Path,
    output_dir: Path,
    max_samples: int,
    settings: _ExecutionSettings,
    system_prompt: str,
    skill_library: Path,
    experience_library: Path,
) -> SimpleNamespace:
    is_train = phase == "train"
    return SimpleNamespace(
        input_file=str(data_path),
        image_folder=str(image_dir),
        output_dir=str(output_dir),
        skip_completed=settings.skip_completed,
        disable_memory_usage_update=phase in {"val", "test"},
        max_samples=max_samples,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_completion_tokens=settings.max_completion_tokens,
        max_turns=settings.max_turns,
        max_images=settings.max_images,
        max_total_tokens=settings.max_total_tokens,
        max_pixels=settings.max_pixels,
        min_pixels=settings.min_pixels,
        num_workers=settings.num_workers,
        rollouts_per_sample=settings.rollouts_per_sample,
        seed_base=settings.seed_base,
        system_prompts_path=settings.system_prompts_path,
        system_prompt_key=system_prompt,
        tool_config_path=settings.tool_config_path,
        image_search_max_calls=settings.image_search_max_calls,
        web_search_max_calls=settings.web_search_max_calls,
        code_interpreter_max_calls=settings.code_interpreter_max_calls,
        experience_enable=True,
        experience_library=str(experience_library),
        experience_online_generate=is_train,
        experience_library_update=is_train,
        experience_max_ops=settings.experience_max_ops,
        experience_large_batch=settings.experience_large_batch,
        experience_max_items=settings.experience_max_items,
        experience_refine=is_train,
        experience_generation_workers=settings.experience_generation_workers,
        skill_enable=True,
        skill_library=str(skill_library),
        skill_inference=True,
        skill_cluster_max_clusters=settings.skill_cluster_max_clusters,
        skill_cluster_min_experiences=settings.skill_cluster_min_experiences,
        skill_retrieval_top_k=settings.skill_retrieval_top_k,
        skill_retrieval_candidate_top_k=None,
        skill_retrieval_min_similarity=settings.skill_retrieval_min_similarity,
        memory_rerank_top_skills=settings.memory_rerank_top_skills,
        memory_rerank_top_experiences=settings.memory_rerank_top_experiences,
        experience_retrieval=True,
        experience_retrieval_top_k=settings.experience_retrieval_top_k,
        experience_retrieval_subtask_top_k=None,
        experience_retrieval_min_similarity=settings.experience_retrieval_min_similarity,
        experience_embedding_api_key=None,
        experience_embedding_endpoint=None,
        experience_embedding_cache_enable=True,
        experience_retrieval_decomposition=True,
        experience_retrieval_rewrite=True,
    )


def _save_phase_metrics(args, data, all_results) -> None:
    """Write rollout results and aggregate pass@k dataset metrics."""
    from main_flow.metrics import compute_dataset_summary, _print_summary, _save_results

    _save_results(all_results, args.output_dir)
    _print_summary(all_results, args.output_dir)
    print(f"\n{'='*80}")
    print("Aggregating pass@k, average@k, and diagnostic metrics across all samples")
    print(f"{'='*80}\n")
    compute_dataset_summary(data, args)


def _run_main_flow_execution(args):
    """Run one train/validation/test phase."""
    from main_flow.config import ExperimentConfig
    from main_flow.memory_update import TrainingMemoryUpdater
    from main_flow.rollout_runner import MemoryRuntime, RolloutRunner

    output_name = os.path.basename(os.path.normpath(args.output_dir or "")).lower()
    is_validation_phase = (
        output_name == "val"
        or output_name.startswith("val_")
        or output_name.startswith("validation")
    )
    is_train_phase = output_name == "train" or output_name.startswith("train_")
    os.environ["MMAGENT_TRAIN_PHASE"] = "1" if is_train_phase else "0"
    os.environ["MMAGENT_VALIDATION_PHASE"] = "1" if is_validation_phase else "0"

    try:
        config = ExperimentConfig(args).load()
    except RuntimeError as e:
        print(e)
        raise SystemExit(1) from e

    memory_runtime = MemoryRuntime(args, config.base_system_prompt).initialize()
    training_updater = TrainingMemoryUpdater(args, memory_runtime)
    all_results = RolloutRunner(config, memory_runtime, training_updater).run()
    _save_phase_metrics(args, config.data, all_results)
    print("Phase inference finished.")


def _print_phase_banner(
    *,
    phase: str,
    benchmark: str,
    paths: _PhasePaths,
    memory: _MemoryPaths,
    settings: _ExecutionSettings,
    max_samples: int,
    system_prompt: str,
    enabled_tools: str,
) -> None:
    print("=" * 70)
    print(f"{phase} benchmark")
    print(f"Benchmark: {benchmark}")
    print(f"Data: {paths.data_path}")
    print(f"Image root: {paths.image_dir}")
    print(f"Output: {paths.output_dir}")
    print(f"Log: {paths.log_file}")
    print(f"Memory: {memory.memory_root}")
    print(f"Rollouts per sample: {settings.rollouts_per_sample}")
    print(f"Max samples: {max_samples}")
    print(f"System prompt: {system_prompt}")
    print(f"Enabled tools: {enabled_tools}")
    print("=" * 70)


def _build_phase_env(
    *,
    phase: str,
    benchmark: str,
    phase_env: dict[str, str],
    system_prompt: str,
    enabled_tools: str,
    memory: _MemoryPaths,
) -> dict[str, str]:
    env = {key: str(value) for key, value in phase_env.items() if value is not None}
    env.update(
        {
            "BENCHMARK": benchmark,
            "SYSTEM_PROMPT_TYPE": system_prompt,
            "ENABLED_TOOLS": enabled_tools,
            "MEMORY_NAME": memory.memory_name,
            "MEMORY_ROOT": str(memory.memory_root),
            "EXPERIENCE_LIBRARY": str(memory.experience_library),
            "SKILL_LIBRARY": str(memory.skill_library),
        }
    )
    if phase == "val":
        _set_phase_env_default(env, "ENABLE_UNVALIDATED_RECOVERY_VALIDATION_PROBE", "1")
        _set_phase_env_default(env, "ENABLE_UNVALIDATED_RECOVERY_TOOL_PLAN", "1")
    elif phase == "train":
        _set_phase_env_default(env, "ENABLE_UNVALIDATED_RECOVERY_TRAINING_PROBE", "1")
    return env


def run_phase(
    phase: str,
    benchmark: str,
    ctx: MainFlowContext,
    phase_env: dict[str, str] | None = None,
) -> None:
    phase_env = phase_env or {}
    default_system_prompt, default_tools = _default_prompt_and_tools(benchmark)
    system_prompt = os.environ.get("SYSTEM_PROMPT_TYPE") or default_system_prompt
    enabled_tools = os.environ.get("ENABLED_TOOLS") or default_tools

    paths = _resolve_phase_paths(phase, benchmark, ctx, phase_env)
    memory = _resolve_memory_paths(benchmark, ctx, phase_env)
    if not paths.data_path.exists():
        raise SystemExit(f"Error: missing {phase} data file: {paths.data_path}")
    max_samples = _count_samples_for_phase(phase, paths.data_path, phase_env)
    settings = _ExecutionSettings.from_env()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    memory.memory_root.mkdir(parents=True, exist_ok=True)

    _print_phase_banner(
        phase=phase,
        benchmark=benchmark,
        paths=paths,
        memory=memory,
        settings=settings,
        max_samples=max_samples,
        system_prompt=system_prompt,
        enabled_tools=enabled_tools,
    )
    env = _build_phase_env(
        phase=phase,
        benchmark=benchmark,
        phase_env=phase_env,
        system_prompt=system_prompt,
        enabled_tools=enabled_tools,
        memory=memory,
    )

    args = _build_execution_args(
        phase=phase,
        data_path=paths.data_path,
        image_dir=paths.image_dir,
        output_dir=paths.output_dir,
        max_samples=max_samples,
        settings=settings,
        system_prompt=system_prompt,
        skill_library=memory.skill_library,
        experience_library=memory.experience_library,
    )
    with _patched_environ(env), _tee_output(paths.log_file):
        _run_main_flow_execution(args)
    check_phase_health(phase, paths.output_dir, memory.memory_root, paths.log_file)

    if phase in {"val", "test"}:
        _write_phase_summary(paths.output_dir, settings.rollouts_per_sample)
