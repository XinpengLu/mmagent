"""Run health checks after a train/validation/test phase."""

from __future__ import annotations

import os
from pathlib import Path

from utils.env import env_bool, env_float, env_int
from utils.artifacts import DATASET_SUMMARY_FILENAME, SAMPLE_SUMMARY_FILENAME, read_json


def _load_phase_metrics(output_dir: Path) -> dict:
    metrics_path = output_dir / DATASET_SUMMARY_FILENAME
    if not metrics_path.exists():
        raise SystemExit(f"Health check failed: missing metrics file {metrics_path}")
    return read_json(metrics_path)


def _health_numbers(metrics: dict) -> tuple[int, float, float, float]:
    num_samples = int(metrics.get("num_samples", 0) or 0)
    tool_usage = metrics.get("tool_usage") if isinstance(metrics.get("tool_usage"), dict) else {}
    tool_rate = float(tool_usage.get("tool_rollout_rate", 0.0) or 0.0)
    avg_tool_calls = float(tool_usage.get("avg_tool_calls_per_rollout", 0.0) or 0.0)
    k_metrics = metrics.get("k_metrics") if isinstance(metrics.get("k_metrics"), dict) else {}
    selected = k_metrics.get(str(os.environ.get("ROLLOUTS_PER_SAMPLE", "4"))) or k_metrics.get("4") or {}
    avg_score = float(selected.get("average_at_k", 0.0) or 0.0)
    return num_samples, avg_score, tool_rate, avg_tool_calls


def _check_tool_activity(num_samples: int, avg_score: float, tool_rate: float, avg_tool_calls: float) -> None:
    min_tool_rate = env_float("PHASE_HEALTH_MIN_TOOL_ROLLOUT_RATE", 0.01)
    if num_samples > 0 and avg_score == 0.0 and tool_rate <= min_tool_rate and avg_tool_calls == 0.0:
        raise SystemExit(
            "Health check failed: phase produced zero score and zero tool calls. "
            "This usually indicates upstream API failure or empty trajectories, not a valid method result."
        )


def _check_auth_errors(log_file: Path) -> None:
    if not log_file.exists():
        return
    text = log_file.read_text(encoding="utf-8", errors="ignore").lower()
    auth_patterns = (
        "api error 401",
        "401 client error",
        "unauthorized",
        "invalid api key",
        "insufficient_quota",
        "remainquota",
        "quota",
    )
    auth_error_count = sum(text.count(pattern) for pattern in auth_patterns)
    max_auth_errors = env_int("PHASE_HEALTH_MAX_AUTH_ERRORS", 0)
    if auth_error_count > max_auth_errors:
        raise SystemExit(
            f"Health check failed: detected {auth_error_count} auth/quota error marker(s) "
            f"in {log_file}. Refresh the API key or configure a fallback before resuming."
        )


def _transient_failure_sample_ids(output_dir: Path) -> list[str]:
    bad_samples = []
    for sample_summary_path in sorted(output_dir.glob(f"*/{SAMPLE_SUMMARY_FILENAME}")):
        row = read_json(sample_summary_path)
        if row.get("has_transient_failure"):
            bad_samples.append(str(sample_summary_path.parent.name))
    return bad_samples


def _check_transient_failures(output_dir: Path) -> None:
    bad_transient_samples = _transient_failure_sample_ids(output_dir)
    if not bad_transient_samples:
        return
    preview = ", ".join(bad_transient_samples[:8])
    suffix = "" if len(bad_transient_samples) <= 8 else f", ... (+{len(bad_transient_samples) - 8})"
    raise SystemExit(
        f"Health check failed: {len(bad_transient_samples)} sample(s) contain transient rollout failures "
        f"and must be rerun before scoring: {preview}{suffix}"
    )


def _experience_count(memory_root: Path) -> int:
    exp_path = memory_root / "experiences.json"
    if not exp_path.exists():
        return 0
    payload = read_json(exp_path)
    exp_lib = payload.get("experiences", payload) if isinstance(payload, dict) else {}
    return len(exp_lib) if isinstance(exp_lib, dict) else 0


def _check_train_memory(phase: str, memory_root: Path, num_samples: int) -> None:
    if phase != "train" or not env_bool("PHASE_HEALTH_REQUIRE_TRAIN_MEMORY", True):
        return
    if num_samples > 0 and _experience_count(memory_root) == 0:
        raise SystemExit(
            "Health check failed: training finished with an empty experience library. "
            "Do not continue validation on an empty memory bank."
        )


def check_phase_health(phase: str, output_dir: Path, memory_root: Path, log_file: Path) -> None:
    if not env_bool("ENABLE_PHASE_HEALTH_CHECK", True):
        return

    metrics = _load_phase_metrics(output_dir)
    num_samples, avg_score, tool_rate, avg_tool_calls = _health_numbers(metrics)
    _check_tool_activity(num_samples, avg_score, tool_rate, avg_tool_calls)
    _check_auth_errors(log_file)
    _check_transient_failures(output_dir)
    _check_train_memory(phase, memory_root, num_samples)

    print(
        f"Health check passed: phase={phase}, samples={num_samples}, "
        f"average@k={avg_score:.4f}, tool_rate={tool_rate:.4f}, avg_tool_calls={avg_tool_calls:.4f}"
    )
