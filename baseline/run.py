"""Run minimal no-tool and simple-tool baselines with standard outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from dotenv import load_dotenv

from configs.runtime import PROJECT_ROOT, RuntimeConfig, RuntimePaths


CONFIG_DIR = PROJECT_ROOT / "configs"
ENV_FILE_PATH = CONFIG_DIR / ".env"
SEARCH_BENCHMARKS = {
    "AgentVista",
    "MMSearch-Plus",
    "MMBrowseComp",
    "BrowseComp-V3",
    "VisBrowse-Bench",
    "Agentic-MME",
}
GENERAL_TOOL_BENCHMARKS = {"VisualToolBench"}
DISABLED_MEMORY_SETTINGS = {
    "experience_large_batch": 1,
    "experience_max_items": 0,
    "experience_generation_workers": 1,
    "skill_cluster_max_clusters": 0,
    "skill_cluster_min_experiences": 0,
    "skill_retrieval_top_k": 0,
    "skill_retrieval_min_similarity": 0.0,
    "memory_rerank_top_skills": 0,
    "memory_rerank_top_experiences": 0,
    "experience_retrieval_top_k": 0,
    "experience_retrieval_min_similarity": 0.0,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _resolve(root: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _safe_path_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {".", "-", "_"} else "_" for ch in value).strip("_") or "model"


def _runtime_from_yaml(path: Path) -> RuntimeConfig:
    payload = _load_yaml(path)
    paths = payload["paths"]
    settings = {key: value for key, value in payload.items() if key not in {"mode", "benchmarks", "paths"}}
    reasoning_model_name = str(settings["models"]["reasoning_model_name"])
    baseline_name = str(settings["baseline"]["name"])
    output_root = PROJECT_ROOT / "output" / _safe_path_name(reasoning_model_name) / baseline_name
    runtime_paths = RuntimePaths(
        data_root=PROJECT_ROOT,
        bench_root=_resolve(PROJECT_ROOT, paths["benchmark_processed"]),
        split_root=_resolve(PROJECT_ROOT, paths["benchmark_split"]),
        output_root=output_root,
        log_root=output_root / "_logs",
        memory_root_base=output_root / "_memory",
        system_prompts=_resolve(PROJECT_ROOT, paths["system_prompts"]),
        tool_config=_resolve(PROJECT_ROOT, paths["tool_config"]),
    )
    runtime = RuntimeConfig(
        mode=str(payload.get("mode") or "test"),
        benchmarks=list(payload["benchmarks"]),
        paths=runtime_paths,
        settings=settings,
        compat_env=_flatten_settings(settings),
    )
    return runtime


def _flatten_settings(settings: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}

    def visit(prefix: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit((*prefix, str(key)), child)
            return
        env["_".join(part.upper() for part in prefix)] = "true" if value is True else "false" if value is False else str(value)

    for section, values in settings.items():
        prefix = () if section in {"runtime", "models"} else (section,)
        visit(prefix, values)
    env["ROLLOUTS_PER_SAMPLE"] = env.pop("ROLLOUT_PER_SAMPLE")
    if "BASELINE_ENABLE_FUNCTION_CALLING" in env:
        env["ENABLE_FUNCTION_CALLING"] = env["BASELINE_ENABLE_FUNCTION_CALLING"]
    return env


def _apply_baseline_env(runtime: RuntimeConfig) -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").rstrip("/")
    chat_url = os.environ.get("OPENAI_CHAT_COMPLETIONS_URL", "").rstrip("/") or f"{base_url}/chat/completions"

    os.environ.update(
        {
            "DATA_ROOT": str(runtime.paths.data_root),
            "BENCH_ROOT": str(runtime.paths.bench_root),
            "BENCHMARK_SPLIT_ROOT": str(runtime.paths.split_root),
            "OUTPUT_ROOT": str(runtime.paths.output_root),
            "LOG_ROOT": str(runtime.paths.log_root),
            "SYSTEM_PROMPTS_PATH": str(runtime.paths.system_prompts),
            "TOOL_CONFIG_PATH": str(runtime.paths.tool_config),
            "OPENAI_BASE_URL": base_url,
            "OPENAI_API_BASE": base_url,
            "OPENAI_CHAT_COMPLETIONS_URL": chat_url,
            "REASONING_API_KEY": api_key,
            "REASONING_END_POINT": chat_url,
            "VERIFIER_API_KEY": api_key,
            "VERIFIER_END_POINT": base_url,
        }
    )
    os.environ.update(runtime.compat_env)


def _baseline_prompt_and_tools(benchmark: str, settings: dict[str, Any]) -> tuple[str, str]:
    baseline = settings["baseline"]
    if baseline["name"] == "no_tool":
        return str(baseline["system_prompt"]), ""
    if benchmark in GENERAL_TOOL_BENCHMARKS:
        return str(baseline["general_prompt"]), str(baseline["general_tools"])
    if benchmark in SEARCH_BENCHMARKS:
        return str(baseline["search_prompt"]), str(baseline["search_tools"])
    return str(baseline["code_prompt"]), str(baseline["code_tools"])


def _sample_count(data_path: Path, max_samples: int | None) -> int:
    if max_samples is not None:
        return max_samples
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    return len(data)


def _build_args(runtime: RuntimeConfig, benchmark: str, max_samples: int | None) -> SimpleNamespace:
    paths = runtime.paths
    settings = runtime.settings
    system_prompt, enabled_tools = _baseline_prompt_and_tools(benchmark, settings)
    data_path = paths.split_root / benchmark / "test.json"
    output_dir = paths.output_root / benchmark / "test"
    return SimpleNamespace(
        phase="test",
        input_file=str(data_path),
        image_folder=str(paths.bench_root),
        output_dir=str(output_dir),
        max_samples=_sample_count(data_path, max_samples),
        num_workers=int(settings["runtime"]["num_workers"]),
        rollouts_per_sample=int(settings["rollout"]["per_sample"]),
        system_prompts_path=str(paths.system_prompts),
        tool_config_path=str(paths.tool_config),
        system_prompt_key=system_prompt,
        enabled_tools=enabled_tools,
        **DISABLED_MEMORY_SETTINGS,
        experience_library="",
        skill_library="",
    )


def _run_benchmark(runtime: RuntimeConfig, benchmark: str, max_samples: int | None) -> None:
    from main_flow.config import ExperimentConfig
    from main_flow.metrics import _save_results, compute_dataset_summary
    from main_flow.phase import write_phase_summary
    from main_flow.rollout_runner import MemoryRuntime, RolloutRunner
    from main_flow.memory_update import TrainingMemoryUpdater
    from skill.utils import set_configured_callable_tools

    args = _build_args(runtime, benchmark, max_samples)
    data_path = Path(args.input_file)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    os.environ["BENCHMARK"] = benchmark
    os.environ["MMAGENT_TRAIN_PHASE"] = "false"
    os.environ["MMAGENT_VALIDATION_PHASE"] = "false"
    set_configured_callable_tools(args.enabled_tools)

    config = ExperimentConfig(args).load()
    memory_runtime = MemoryRuntime(args, config.base_system_prompt).initialize()
    all_results = RolloutRunner(config, memory_runtime, TrainingMemoryUpdater(args, memory_runtime)).run()
    _save_results(all_results, args.output_dir)
    compute_dataset_summary(config.data, args)
    write_phase_summary(Path(args.output_dir), args.rollouts_per_sample)


def run(config_path: Path, max_samples: int | None) -> None:
    load_dotenv(ENV_FILE_PATH, override=True)
    runtime = _runtime_from_yaml(config_path)
    _apply_baseline_env(runtime)
    for benchmark in runtime.benchmarks:
        print(f"\n[Baseline] {runtime.settings['baseline']['name']} / {benchmark}")
        _run_benchmark(runtime, benchmark, max_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "runtime_no_tool.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run(args.config if args.config.is_absolute() else PROJECT_ROOT / args.config, args.max_samples)


if __name__ == "__main__":
    main()
