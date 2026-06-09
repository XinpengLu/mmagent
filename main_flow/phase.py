from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from .context import (
    MainFlowContext,
    _count_json_items,
    _default_prompt_and_tools,
)
from utils.artifacts import AVERAGE_PASS_FILENAME, DATASET_SUMMARY_FILENAME, read_json, write_json


def _setting(settings: dict, path: str):
    current = settings
    for key in path.split("."):
        if not isinstance(current, dict):
            raise KeyError(f"Invalid runtime setting path: {path}")
        current = current[key]
    return current


def _setting_int(settings: dict, path: str) -> int:
    return int(_setting(settings, path))


def _setting_float(settings: dict, path: str) -> float:
    return float(_setting(settings, path))


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
    def from_context(cls, ctx: MainFlowContext) -> "_ExecutionSettings":
        settings = ctx.settings
        num_workers = _setting_int(settings, "runtime.num_workers")
        return cls(
            temperature=0.6,
            top_p=1.0,
            max_completion_tokens=8192,
            max_turns=20,
            max_images=50,
            max_total_tokens=32768,
            max_pixels=2_000_000,
            min_pixels=40_000,
            num_workers=num_workers,
            rollouts_per_sample=_setting_int(settings, "rollout.per_sample"),
            seed_base=42,
            system_prompts_path=str(ctx.system_prompts),
            tool_config_path=str(ctx.tool_config),
            image_search_max_calls=5,
            web_search_max_calls=7,
            code_interpreter_max_calls=10,
            experience_max_ops=3,
            experience_large_batch=_setting_int(settings, "experience.large_batch"),
            experience_max_items=_setting_int(settings, "experience.max_items"),
            experience_generation_workers=_setting_int(settings, "experience.generation_workers"),
            skill_cluster_max_clusters=_setting_int(settings, "skill.cluster.max_clusters"),
            skill_cluster_min_experiences=_setting_int(settings, "skill.cluster.min_experiences"),
            skill_retrieval_top_k=_setting_int(settings, "skill.retrieval.top_k"),
            skill_retrieval_min_similarity=_setting_float(settings, "skill.retrieval.min_similarity"),
            memory_rerank_top_skills=_setting_int(settings, "memory.rerank_top_skills"),
            memory_rerank_top_experiences=_setting_int(settings, "memory.rerank_top_experiences"),
            experience_retrieval_top_k=_setting_int(settings, "experience.retrieval.top_k"),
            experience_retrieval_min_similarity=_setting_float(settings, "experience.retrieval.min_similarity"),
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


def _count_samples_for_phase(phase: str, data_path: Path, phase_env: dict[str, str]) -> int:
    value = phase_env.get(f"MAX_SAMPLES_{phase.upper()}", "")
    return int(value) if value else _count_json_items(data_path)


def _resolve_memory_paths(benchmark: str, ctx: MainFlowContext, phase_env: dict[str, str]) -> _MemoryPaths:
    memory_root = Path(phase_env.get("MEMORY_ROOT_OVERRIDE") or ctx.memory_root_base / benchmark)
    return _MemoryPaths(
        memory_name=benchmark,
        memory_root=memory_root,
        experience_library=memory_root / "experiences.json",
        skill_library=memory_root / "skill_library.anchor",
    )


def _resolve_phase_paths(
    phase: str,
    benchmark: str,
    ctx: MainFlowContext,
    phase_env: dict[str, str],
) -> _PhasePaths:
    prefix = phase.upper()
    data_path = Path(phase_env.get(f"{prefix}_DATA_PATH") or ctx.split_root / benchmark / f"{phase}.json")
    output_dir = Path(phase_env.get(f"{prefix}_OUTPUT_DIR") or ctx.output_root / benchmark / phase)
    log_output_dir = Path(phase_env.get(f"{prefix}_LOG_OUTPUT_DIR") or ctx.log_root / benchmark / phase)
    return _PhasePaths(
        data_path=data_path,
        image_dir=ctx.bench_root,
        output_dir=output_dir,
        log_file=log_output_dir.with_suffix(".log"),
    )


def write_phase_summary(output_dir: Path, k: int) -> None:
    metrics_path = output_dir / DATASET_SUMMARY_FILENAME
    summary_path = output_dir / AVERAGE_PASS_FILENAME
    metrics = read_json(metrics_path)
    selected = metrics["k_metrics"][str(k)]

    summary = {
        "k": k,
        f"pass@{k}": selected["pass_at_k"],
        f"average@{k}": selected["average_at_k"],
        "metrics_file": str(metrics_path),
    }
    write_json(summary_path, summary, indent=2)


def _build_execution_args(
    *,
    phase: str,
    data_path: Path,
    image_dir: Path,
    output_dir: Path,
    max_samples: int,
    settings: _ExecutionSettings,
    system_prompt: str,
    enabled_tools: str,
    skill_library: Path,
    experience_library: Path,
) -> SimpleNamespace:
    is_train = phase == "train"
    return SimpleNamespace(
        phase=phase,
        input_file=str(data_path),
        image_folder=str(image_dir),
        output_dir=str(output_dir),
        skip_completed=True,
        disable_memory_usage_update=phase in {"val", "test"},
        max_samples=max_samples,
        **asdict(settings),
        system_prompt_key=system_prompt,
        enabled_tools=enabled_tools,
        experience_enable=True,
        experience_library=str(experience_library),
        experience_online_generate=is_train,
        experience_library_update=is_train,
        experience_refine=is_train,
        skill_enable=True,
        skill_library=str(skill_library),
        skill_inference=True,
        skill_retrieval_candidate_top_k=None,
        experience_retrieval=True,
        experience_retrieval_subtask_top_k=None,
        experience_embedding_api_key=None,
        experience_embedding_endpoint=None,
        experience_embedding_cache_enable=True,
        experience_retrieval_decomposition=True,
        experience_retrieval_rewrite=True,
    )


def run_main_flow_execution(args):
    """Run one train/validation/test phase."""
    from main_flow.config import ExperimentConfig
    from main_flow.memory_update import TrainingMemoryUpdater
    from main_flow.rollout_runner import MemoryRuntime, RolloutRunner
    from main_flow.metrics import compute_dataset_summary, _save_results
    from skill.utils import set_configured_callable_tools

    set_configured_callable_tools(args.enabled_tools)

    config = ExperimentConfig(args).load()
    memory_runtime = MemoryRuntime(args, config.base_system_prompt).initialize()
    training_updater = TrainingMemoryUpdater(args, memory_runtime)
    all_results = RolloutRunner(config, memory_runtime, training_updater).run()
    _save_results(all_results, args.output_dir)
    compute_dataset_summary(config.data, args)

def _build_phase_env(
    *,
    phase: str,
    benchmark: str,
    memory: _MemoryPaths,
) -> dict[str, str]:
    return {
        "BENCHMARK": benchmark,
        "MEMORY_NAME": memory.memory_name,
        "MEMORY_ROOT": str(memory.memory_root),
        "EXPERIENCE_LIBRARY": str(memory.experience_library),
        "SKILL_LIBRARY": str(memory.skill_library),
        "MMAGENT_TRAIN_PHASE": str(phase == "train").lower(),
        "MMAGENT_VALIDATION_PHASE": str(phase == "val").lower(),
    }


def run_phase(
    phase: str,
    benchmark: str,
    ctx: MainFlowContext,
    phase_env: dict[str, str] | None = None,
) -> None:
    phase_env = phase_env or {}
    system_prompt, enabled_tools = _default_prompt_and_tools(benchmark)

    paths = _resolve_phase_paths(phase, benchmark, ctx, phase_env)
    memory = _resolve_memory_paths(benchmark, ctx, phase_env)

    if not paths.data_path.exists():
        raise SystemExit(f"Error: missing {phase} data file: {paths.data_path}")
    max_samples = _count_samples_for_phase(phase, paths.data_path, phase_env)
    settings = _ExecutionSettings.from_context(ctx)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    memory.memory_root.mkdir(parents=True, exist_ok=True)

    env = _build_phase_env(
        phase=phase,
        benchmark=benchmark,
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
        enabled_tools=enabled_tools,
        skill_library=memory.skill_library,
        experience_library=memory.experience_library,
    )
    with _patched_environ(env), _tee_output(paths.log_file):
        run_main_flow_execution(args)

    if phase in {"val", "test"}:
        write_phase_summary(paths.output_dir, settings.rollouts_per_sample)
