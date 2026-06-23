"""Runtime YAML loader for benchmark runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


VALID_MODES = {"train", "val", "test", "train_test", "train_val", "train_val_test"}
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
RUNTIME_CONFIG_PATH = CONFIG_DIR / "runtime.yaml"
ENV_FILE_PATH = CONFIG_DIR / ".env"
_ACTIVE_RUNTIME_CONFIG: RuntimeConfig | None = None

REQUIRED_ENV_KEYS = (
    "REASONING_MODEL_NAME",
    "VERIFIER_MODEL_NAME",
    "EXPERIENCE_MODEL_NAME",
    "EXPERIENCE_EMBEDDING_MODEL",
    "ENABLE_FUNCTION_CALLING",
    "NUM_WORKERS",
    "ROLLOUTS_PER_SAMPLE",
    "EXPERIENCE_LARGE_BATCH",
    "EXPERIENCE_GENERATION_WORKERS",
    "EXPERIENCE_MAX_ITEMS",
)

RESERVED_CONFIG_KEYS = {"name", "mode", "benchmarks", "paths"}

COMPAT_ENV_ALIASES = {
    "ROLLOUT_PER_SAMPLE": "ROLLOUTS_PER_SAMPLE",
    "EXPERIENCE_FAST_PRUNE_ENABLED": "ENABLE_FAST_EXPERIENCE_LIMIT_PRUNE",
    "EXPERIENCE_FAST_PRUNE_MAX_OVERFLOW": "FAST_EXPERIENCE_PRUNE_MAX_OVERFLOW",
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


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    benchmarks: list[str]
    paths: RuntimePaths
    settings: dict[str, Any]
    compat_env: dict[str, str]


def active_runtime_config() -> RuntimeConfig | None:
    return _ACTIVE_RUNTIME_CONFIG


def runtime_setting(path: str, default: Any = None) -> Any:
    runtime = active_runtime_config()
    current: Any = runtime.settings if runtime else {}
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _require_mapping(payload: Any, key: str, source: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must define mapping section: {key}")
    return value


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _resolve(root: Path, raw: Any, key: str) -> Path:
    if raw in (None, ""):
        raise ValueError(f"paths.{key} must be set in runtime config")
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _safe_path_name(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in {".", "-", "_"} else "_" for ch in text).strip("_") or "run"


def _reasoning_model_name(payload: dict[str, Any], source: Path) -> str:
    models = _require_mapping(payload, "models", source)
    return str(models.get("reasoning_model_name") or "model")


def _run_name(payload: dict[str, Any]) -> str:
    return str(payload.get("name") or "mmagent")


def _load_runtime_payload(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a YAML mapping: {config_path}")
    return payload


def _load_runtime_paths(root: Path, payload: dict[str, Any], source: Path) -> RuntimePaths:
    paths = _require_mapping(payload, "paths", source)
    run_root = root / "output" / _safe_path_name(_reasoning_model_name(payload, source)) / _safe_path_name(_run_name(payload))
    return RuntimePaths(
        data_root=root,
        bench_root=_resolve(root, paths.get("benchmark_processed"), "benchmark_processed"),
        split_root=_resolve(root, paths.get("benchmark_split"), "benchmark_split"),
        output_root=run_root,
        log_root=run_root,
        memory_root_base=run_root,
        system_prompts=_resolve(root, paths.get("system_prompts"), "system_prompts"),
        tool_config=_resolve(root, paths.get("tool_config"), "tool_config"),
    )


def _load_api_env(env_file: Path) -> tuple[str, str, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    chat_url = f"{base_url}/chat/completions" if base_url else ""
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY is not set. Put it in {env_file}.")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL must be set in .env.")
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_CHAT_COMPLETIONS_URL"] = chat_url
    return api_key, base_url, chat_url


def _load_runtime_settings(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    settings = {key: value for key, value in payload.items() if key not in RESERVED_CONFIG_KEYS}
    invalid = [key for key, value in settings.items() if not isinstance(value, dict)]
    if invalid:
        raise ValueError(f"Runtime settings must be YAML mappings in {source}: {', '.join(invalid)}")
    return settings


def _flatten_compat_env(settings: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}

    def visit(prefix: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit((*prefix, str(key)), child)
            return

        env_key = "_".join(part.upper() for part in prefix)
        env[env_key] = _env_value(value)

    for section, values in settings.items():
        prefix = () if section in {"runtime", "models"} else (section,)
        visit(prefix, values)

    for generated_name, legacy_name in COMPAT_ENV_ALIASES.items():
        if generated_name in env:
            env[legacy_name] = env.pop(generated_name)
    return env


def _load_benchmarks(payload: dict[str, Any]) -> list[str]:
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks or not all(isinstance(item, str) and item for item in benchmarks):
        raise ValueError("runtime config must define a non-empty benchmarks list")
    return list(benchmarks)


def _apply_run_root_override(root: Path, paths: RuntimePaths) -> RuntimePaths:
    run_id = os.environ.get("RUN_ID", "")
    run_root_raw = os.environ.get("RUN_ROOT", "")
    if not (run_id or run_root_raw):
        return paths

    run_id = _safe_path_name(run_id or "benchmark_run")
    run_root = Path(run_root_raw) if run_root_raw else paths.output_root.parent / run_id
    run_root = run_root if run_root.is_absolute() else root / run_root
    os.environ["RUN_ID"] = run_id
    os.environ["RUN_ROOT"] = str(run_root)
    return replace(
        paths,
        output_root=run_root,
        log_root=run_root,
        memory_root_base=run_root,
    )


def _path_env(paths: RuntimePaths) -> dict[str, str]:
    return {
        "DATA_ROOT": str(paths.data_root),
        "BENCH_ROOT": str(paths.bench_root),
        "BENCHMARK_SPLIT_ROOT": str(paths.split_root),
        "OUTPUT_ROOT": str(paths.output_root),
        "LOG_ROOT": str(paths.log_root),
        "MEMORY_ROOT_BASE": str(paths.memory_root_base),
        "SYSTEM_PROMPTS_PATH": str(paths.system_prompts),
        "TOOL_CONFIG_PATH": str(paths.tool_config),
    }


def _apply_api_defaults(api_key: str, openai_base_url: str, chat_url: str) -> None:
    defaults = {
        "REASONING_API_KEY": api_key,
        "REASONING_END_POINT": chat_url,
        "VERIFIER_API_KEY": api_key,
        "VERIFIER_END_POINT": openai_base_url,
        "EXPERIENCE_API_KEY": api_key,
        "EXPERIENCE_END_POINT": chat_url,
        "EXPERIENCE_EMBEDDING_API_KEY": api_key,
        "EXPERIENCE_EMBEDDING_ENDPOINT": openai_base_url,
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def apply_runtime_env(runtime: RuntimeConfig) -> None:
    """Populate environment variables consumed by legacy runtime modules."""
    api_key, openai_base_url, chat_url = _load_api_env(ENV_FILE_PATH)
    os.environ.update(_path_env(runtime.paths))
    os.environ.update(runtime.compat_env)

    missing = [key for key in REQUIRED_ENV_KEYS if os.environ.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required runtime env values in {RUNTIME_CONFIG_PATH}: {', '.join(missing)}")

    _apply_api_defaults(api_key, openai_base_url, chat_url)


def load_runtime_config() -> RuntimeConfig:
    global _ACTIVE_RUNTIME_CONFIG

    config_path = RUNTIME_CONFIG_PATH
    payload = _load_runtime_payload(config_path)
    root = PROJECT_ROOT
    load_dotenv(ENV_FILE_PATH, override=True)
    paths = _load_runtime_paths(root, payload, config_path)
    paths = _apply_run_root_override(root, paths)

    settings = _load_runtime_settings(payload, config_path)
    runtime = RuntimeConfig(
        mode = payload.get("mode", "train_val"),
        benchmarks=_load_benchmarks(payload),
        paths=paths,
        settings=settings,
        compat_env=_flatten_compat_env(settings),
    )
    apply_runtime_env(runtime)
    _ACTIVE_RUNTIME_CONFIG = runtime
    return runtime
