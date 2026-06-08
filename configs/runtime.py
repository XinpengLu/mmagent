"""Runtime YAML loader for benchmark runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


VALID_MODES = {"train", "val", "test", "train_test", "train_val", "train_val_test"}

REQUIRED_ENV_KEYS = (
    "REASONING_MODEL_NAME",
    "VERIFIER_MODEL_NAME",
    "EXPERIENCE_MODEL_NAME",
    "EXPERIENCE_EMBEDDING_MODEL",
    "ENABLE_FUNCTION_CALLING",
    "ENABLE_VALIDATION",
    "NUM_WORKERS",
    "ROLLOUTS_PER_SAMPLE",
    "EXPERIENCE_LARGE_BATCH",
    "EXPERIENCE_GENERATION_WORKERS",
    "EXPERIENCE_MAX_ITEMS",
    "SKIP_COMPLETED",
)

BOOL_ENV_KEYS = {
    "ENABLE_FUNCTION_CALLING",
    "ENABLE_VALIDATION",
    "VALIDATION_EVAL_MEMORY_ISOLATED",
    "VALIDATION_RESTORE_BEST_EACH_STAGE",
    "VALIDATION_FINAL_SELECT_BEST",
    "ENABLE_VALIDATION_REFLECTION",
    "VALIDATION_REFLECTION_WRITE_BACK",
    "SKIP_COMPLETED",
    "TRAJ_SUMMARY_IMAGE_CAPTION_INCLUDE_SYSTEM_PROMPT",
    "TRAJ_SUMMARY_CAPTION_TOOL_WITH_ORIGINAL",
    "ENABLE_FAST_EXPERIENCE_LIMIT_PRUNE",
    "ENABLE_UNVALIDATED_RECOVERY_TRAINING_PROBE",
    "ENABLE_UNVALIDATED_RECOVERY_VALIDATION_PROBE",
    "ENABLE_UNVALIDATED_RECOVERY_TOOL_PLAN",
}

INT_ENV_KEYS = {
    "VALIDATION_REFLECTION_MAX_SAMPLES",
    "NUM_WORKERS",
    "ROLLOUTS_PER_SAMPLE",
    "ROLLOUT_MAX_ATTEMPTS",
    "REASONING_API_MAX_RETRIES",
    "FINAL_ANSWER_API_MAX_RETRIES",
    "EXPERIENCE_LARGE_BATCH",
    "EXPERIENCE_GENERATION_WORKERS",
    "EXPERIENCE_MAX_ITEMS",
    "EXPERIENCE_API_MAX_RETRIES",
    "EXPERIENCE_CRITIQUE_MAX_RETRIES",
    "EXPERIENCE_EMBEDDING_MAX_RETRIES",
    "TRAJ_SUMMARY_MAX_TOKENS",
    "TRAJ_SUMMARY_MAX_RETRIES",
    "EXPERIENCE_CRITIQUE_MAX_TOKENS",
    "EXPERIENCE_IMAGE_CAPTION_MAX_TOKENS",
    "EXPERIENCE_IMAGE_CAPTION_WORKERS",
    "TRAJ_SUMMARY_IMAGE_BUDGET",
    "TRAJ_SUMMARY_TOOL_IMAGES_PER_ROLLOUT",
    "FAST_EXPERIENCE_PRUNE_MAX_OVERFLOW",
    "FAILURE_RECOVERY_MAX_LLM_CALLS",
    "FAILURE_RECOVERY_MAX_CANDIDATES",
    "FAILURE_RECOVERY_SUCCESS_CANDIDATES",
    "FAILURE_RECOVERY_MAX_TOKENS",
}

FLOAT_ENV_KEYS = {
    "VALIDATION_TARGET_AVERAGE_MIN",
    "VALIDATION_TARGET_MIN_TASK_AVERAGE",
    "ROLLOUT_RETRY_WAIT_SECONDS",
    "REASONING_API_BASE_WAIT",
    "REASONING_API_MAX_WAIT",
    "EXPERIENCE_API_BASE_WAIT",
    "EXPERIENCE_API_MAX_WAIT",
}

PATH_ENV_FIELDS = {
    "DATA_ROOT": "data_root",
    "BENCH_ROOT": "bench_root",
    "BENCHMARK_SPLIT_ROOT": "split_root",
    "OUTPUT_ROOT": "output_root",
    "LOG_ROOT": "log_root",
    "MEMORY_ROOT_BASE": "memory_root_base",
    "SYSTEM_PROMPTS_PATH": "system_prompts",
    "TOOL_CONFIG_PATH": "tool_config",
}


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    bench_root: Path
    split_root: Path
    output_root: Path
    log_root: Path
    memory_root_base: Path
    system_prompts: Path
    tool_config: Path
    env_file: Path


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    benchmarks: list[str]
    path: Path
    project_root: Path
    paths: RuntimePaths
    env: dict[str, str]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_config_path() -> Path:
    return Path(os.environ.get("MMAGENT_RUNTIME_CONFIG") or _project_root() / "configs" / "runtime.yaml").resolve()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            raise ValueError(f"Invalid .env key in {path}: {raw_line!r}")
        os.environ[key] = value


def _require_mapping(payload: Any, key: str, source: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must define mapping section: {key}")
    return value


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def _validate_bool_env(key: str, value: Any, source: Path) -> None:
    if isinstance(value, bool):
        return
    normalized = str(value).strip().lower()
    if normalized in {"1", "0", "true", "false", "yes", "no", "y", "n"}:
        return
    raise ValueError(f"Invalid boolean env value in {source}: {key}={value!r}")


def _validate_int_env(key: str, value: Any, source: Path) -> None:
    if isinstance(value, bool):
        raise ValueError(f"Invalid integer env value in {source}: {key}={value!r}")
    try:
        int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer env value in {source}: {key}={value!r}") from exc


def _validate_float_env(key: str, value: Any, source: Path) -> None:
    if isinstance(value, bool):
        raise ValueError(f"Invalid float env value in {source}: {key}={value!r}")
    try:
        float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float env value in {source}: {key}={value!r}") from exc


def _validate_env_values(env: dict[str, Any], source: Path) -> None:
    for key, value in env.items():
        if value in (None, ""):
            continue
        if key in BOOL_ENV_KEYS:
            _validate_bool_env(key, value, source)
        elif key in INT_ENV_KEYS:
            _validate_int_env(key, value, source)
        elif key in FLOAT_ENV_KEYS:
            _validate_float_env(key, value, source)


def _resolve(root: Path, raw: Any, key: str) -> Path:
    if raw in (None, ""):
        raise ValueError(f"paths.{key} must be set in runtime config")
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _load_runtime_payload(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a YAML mapping: {config_path}")
    return payload


def _load_runtime_paths(root: Path, payload: dict[str, Any], source: Path) -> RuntimePaths:
    paths = _require_mapping(payload, "paths", source)
    return RuntimePaths(
        data_root=root,
        bench_root=_resolve(root, paths.get("benchmark_processed"), "benchmark_processed"),
        split_root=_resolve(root, paths.get("benchmark_split"), "benchmark_split"),
        output_root=_resolve(root, paths.get("output_root"), "output_root"),
        log_root=_resolve(root, paths.get("log_root"), "log_root"),
        memory_root_base=_resolve(root, paths.get("memory_root_base"), "memory_root_base"),
        system_prompts=_resolve(root, paths.get("system_prompts"), "system_prompts"),
        tool_config=_resolve(root, paths.get("tool_config"), "tool_config"),
        env_file=_resolve(root, paths.get("env_file"), "env_file"),
    )


def _load_api_env(env_file: Path) -> tuple[str, str, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").rstrip("/")
    chat_url = os.environ.get("OPENAI_CHAT_COMPLETIONS_URL", "").rstrip("/")
    if base_url and not chat_url:
        chat_url = f"{base_url}/chat/completions"
        os.environ["OPENAI_CHAT_COMPLETIONS_URL"] = chat_url
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY is not set. Put it in {env_file}.")
    if not base_url or not chat_url:
        raise ValueError(
            "OPENAI_BASE_URL must be set in .env. "
            "OPENAI_CHAT_COMPLETIONS_URL is optional and defaults to <OPENAI_BASE_URL>/chat/completions."
        )
    return api_key, base_url, chat_url


def _load_runtime_env(payload: dict[str, Any], source: Path) -> dict[str, str]:
    raw_env = _require_mapping(payload, "env", source)
    _validate_env_values(raw_env, source)
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid env key in {source}: {key!r}")
        env[key] = _stringify(value)
    return env


def _normalize_mode(mode: Any, env: dict[str, str], source: Path) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"runtime mode must be one of {sorted(VALID_MODES)}; got {mode!r}")

    effective_mode = str(mode)
    if effective_mode == "train_val":
        env["ENABLE_VALIDATION"] = "1"
        return "train"
    if effective_mode == "train_val_test":
        env["ENABLE_VALIDATION"] = "1"
    return effective_mode


def _load_benchmarks(payload: dict[str, Any]) -> list[str]:
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks or not all(isinstance(item, str) and item for item in benchmarks):
        raise ValueError("runtime config must define a non-empty benchmarks list")
    return list(benchmarks)


def _apply_env_path_overrides(paths: RuntimePaths, env: dict[str, str]) -> RuntimePaths:
    updates = {
        field_name: Path(env_key_value)
        for env_key, field_name in PATH_ENV_FIELDS.items()
        if (env_key_value := env.get(env_key)) not in (None, "")
    }
    return replace(paths, **updates) if updates else paths


def _apply_run_root_override(root: Path, paths: RuntimePaths, env: dict[str, str]) -> RuntimePaths:
    effective_env = os.environ.copy()
    effective_env.update(env)
    run_id = effective_env.get("RUN_ID", "")
    run_root_raw = effective_env.get("RUN_ROOT", "")
    if not (run_id or run_root_raw):
        return paths

    run_id = run_id or "benchmark_run"
    run_root = Path(run_root_raw) if run_root_raw else root / "output" / "repro_benchmark_runs" / run_id
    run_root = run_root if run_root.is_absolute() else root / run_root
    env["RUN_ID"] = run_id
    env["RUN_ROOT"] = str(run_root)
    env["OUTPUT_ROOT"] = str(run_root / "output")
    env["LOG_ROOT"] = str(run_root / "logs")
    env["MEMORY_ROOT_BASE"] = str(run_root / "memory")
    env["VALIDATION_SNAPSHOT_ROOT"] = str(run_root / "snapshots")
    return RuntimePaths(
        data_root=paths.data_root,
        bench_root=paths.bench_root,
        split_root=paths.split_root,
        output_root=run_root / "output",
        log_root=run_root / "logs",
        memory_root_base=run_root / "memory",
        system_prompts=paths.system_prompts,
        tool_config=paths.tool_config,
        env_file=paths.env_file,
    )


def apply_runtime_env(runtime: RuntimeConfig) -> None:
    """Populate environment variables consumed by the runtime modules."""
    api_key, openai_base_url, chat_url = _load_api_env(runtime.paths.env_file)
    os.environ["DATA_ROOT"] = str(runtime.paths.data_root)
    os.environ["BENCH_ROOT"] = str(runtime.paths.bench_root)
    os.environ["BENCHMARK_SPLIT_ROOT"] = str(runtime.paths.split_root)
    os.environ["OUTPUT_ROOT"] = str(runtime.paths.output_root)
    os.environ["LOG_ROOT"] = str(runtime.paths.log_root)
    os.environ["MEMORY_ROOT_BASE"] = str(runtime.paths.memory_root_base)
    os.environ["SYSTEM_PROMPTS_PATH"] = str(runtime.paths.system_prompts)
    os.environ["TOOL_CONFIG_PATH"] = str(runtime.paths.tool_config)

    for key, value in runtime.env.items():
        os.environ[key] = value

    missing = [key for key in REQUIRED_ENV_KEYS if os.environ.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required runtime env values in {runtime.path}: {', '.join(missing)}")

    os.environ.setdefault("REASONING_API_KEY", api_key)
    os.environ.setdefault("REASONING_END_POINT", chat_url)
    os.environ.setdefault("VERIFIER_API_KEY", api_key)
    os.environ.setdefault("VERIFIER_END_POINT", openai_base_url)
    os.environ.setdefault("EXPERIENCE_API_KEY", api_key)
    os.environ.setdefault("EXPERIENCE_END_POINT", chat_url)
    os.environ.setdefault("EXPERIENCE_EMBEDDING_API_KEY", api_key)
    os.environ.setdefault("EXPERIENCE_EMBEDDING_ENDPOINT", openai_base_url)


def load_runtime_config() -> RuntimeConfig:
    config_path = _runtime_config_path()
    payload = _load_runtime_payload(config_path)
    root = _project_root()
    paths = _load_runtime_paths(root, payload, config_path)
    _load_env_file(paths.env_file)

    env = _load_runtime_env(payload, config_path)
    mode = _normalize_mode(payload.get("mode"), env, config_path)
    paths = _apply_env_path_overrides(paths, env)
    paths = _apply_run_root_override(root, paths, env)
    runtime = RuntimeConfig(
        mode=mode,
        benchmarks=_load_benchmarks(payload),
        path=config_path,
        project_root=root,
        paths=paths,
        env=env,
    )
    apply_runtime_env(runtime)
    return runtime
