"""Runtime YAML loader for benchmark runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from utils.key_pool import configure_key_file


VALID_MODES = {"train", "val", "test", "train_test", "train_val", "train_val_test"}
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
RUNTIME_CONFIG_PATH = CONFIG_DIR / "runtime.yaml"
_ACTIVE_RUNTIME_CONFIG: RuntimeConfig | None = None

REQUIRED_ENV_KEYS = (
    "REASONING_MODEL_NAME",
    "REASONING_API_KEY",
    "REASONING_END_POINT",
    "VERIFIER_MODEL_NAME",
    "VERIFIER_API_KEY",
    "VERIFIER_END_POINT",
    "EXPERIENCE_MODEL_NAME",
    "EXPERIENCE_API_KEY",
    "EXPERIENCE_END_POINT",
    "EXPERIENCE_EMBEDDING_MODEL",
    "EXPERIENCE_EMBEDDING_API_KEY",
    "EXPERIENCE_EMBEDDING_ENDPOINT",
    "ENABLE_FUNCTION_CALLING",
    "NUM_WORKERS",
    "ROLLOUTS_PER_SAMPLE",
    "EXPERIENCE_LARGE_BATCH",
    "EXPERIENCE_GENERATION_WORKERS",
    "EXPERIENCE_MAX_ITEMS",
)

RESERVED_CONFIG_KEYS = {"extends", "name", "mode", "benchmarks", "paths"}

COMPAT_ENV_ALIASES = {
    "ROLLOUT_PER_SAMPLE": "ROLLOUTS_PER_SAMPLE",
    "EXPERIENCE_FAST_PRUNE_ENABLED": "ENABLE_FAST_EXPERIENCE_LIMIT_PRUNE",
    "EXPERIENCE_FAST_PRUNE_MAX_OVERFLOW": "FAST_EXPERIENCE_PRUNE_MAX_OVERFLOW",
}

API_KEY_POOLS = (
    "reasoning",
    "verifier",
    "experience",
    "experience_embedding",
    "serp",
    "jina",
)

API_ROLES = {
    "REASONING": ("reasoning", "REASONING_END_POINT", "chat"),
    "VERIFIER": ("verifier", "VERIFIER_END_POINT", "base"),
    "EXPERIENCE": ("experience", "EXPERIENCE_END_POINT", "chat"),
    "EXPERIENCE_EMBEDDING": ("experience_embedding", "EXPERIENCE_EMBEDDING_ENDPOINT", "base"),
}

KEY_FILE_ENV_BY_POOL = {
    "reasoning": "REASONING_KEY_FILE",
    "verifier": "VERIFIER_KEY_FILE",
    "experience": "EXPERIENCE_KEY_FILE",
    "experience_embedding": "EXPERIENCE_EMBEDDING_KEY_FILE",
    "serp": "SERPAPI_KEY_FILE",
    "jina": "JINA_KEY_FILE",
}

API_KEY_ENV_BY_POOL = {
    "reasoning": "REASONING_API_KEY",
    "verifier": "VERIFIER_API_KEY",
    "experience": "EXPERIENCE_API_KEY",
    "experience_embedding": "EXPERIENCE_EMBEDDING_API_KEY",
    "serp": "SERPAPI_KEY",
    "jina": "JINA_API_KEY",
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
    base_name = str(payload.get("name") or "mmagent")
    ablation = payload.get("ablation")
    if not isinstance(ablation, dict):
        return base_name
    ablation_name = str(ablation.get("name") or "").strip()
    if not ablation_name or ablation_name.lower() in {"none", "default"}:
        return base_name
    ablation_kind = str(ablation.get("kind") or "ablation").strip() or "ablation"
    return f"{base_name}_ablation_{ablation_kind}_{ablation_name}"


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_runtime_payload(config_path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    seen = seen or set()
    if config_path in seen:
        raise ValueError(f"Runtime config extends cycle detected: {config_path}")
    seen.add(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a YAML mapping: {config_path}")

    parent = payload.get("extends")
    if not parent:
        return payload

    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_payload = _load_runtime_payload(parent_path, seen)
    child_payload = {key: value for key, value in payload.items() if key != "extends"}
    return _deep_merge_config(parent_payload, child_payload)


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


def _base_url(env: dict[str, str], prefix: str) -> str:
    return (env.get(f"{prefix}_BASE_URL") or "").rstrip("/")


def _chat_url(base_url: str) -> str:
    return f"{base_url}/chat/completions" if base_url else ""


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
        prefix = () if section in {"runtime", "models", "api"} else (section,)
        visit(prefix, values)

    for generated_name, legacy_name in COMPAT_ENV_ALIASES.items():
        if generated_name in env:
            env[legacy_name] = env.pop(generated_name)
    _apply_key_files(env)
    return env


def _apply_key_files(env: dict[str, str]) -> None:
    for pool in API_KEY_POOLS:
        key_file = env.get(KEY_FILE_ENV_BY_POOL[pool])
        if key_file:
            env[API_KEY_ENV_BY_POOL[pool]] = configure_key_file(pool, key_file)


def _apply_role_api_config(env: dict[str, str]) -> None:
    for prefix, (pool, endpoint_key, endpoint_kind) in API_ROLES.items():
        if not env.get(f"{prefix}_API_KEY"):
            key_file = env.get(f"{prefix}_KEY_FILE")
            if key_file:
                env[f"{prefix}_API_KEY"] = configure_key_file(pool, key_file)

        base_url = _base_url(env, prefix)
        if base_url:
            env[endpoint_key] = _chat_url(base_url) if endpoint_kind == "chat" else base_url



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


def apply_runtime_env(runtime: RuntimeConfig, source: Path = RUNTIME_CONFIG_PATH) -> None:
    """Populate environment variables consumed by legacy runtime modules."""
    os.environ.update(_path_env(runtime.paths))
    env = dict(runtime.compat_env)
    _apply_role_api_config(env)
    os.environ.update(env)

    missing = [key for key in REQUIRED_ENV_KEYS if os.environ.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required runtime env values in {source}: {', '.join(missing)}")


def load_runtime_config(config_path: str | Path | None = None) -> RuntimeConfig:
    global _ACTIVE_RUNTIME_CONFIG

    config_path = Path(config_path) if config_path is not None else RUNTIME_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    payload = _load_runtime_payload(config_path)
    root = PROJECT_ROOT
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
    apply_runtime_env(runtime, config_path)
    _ACTIVE_RUNTIME_CONFIG = runtime
    return runtime
