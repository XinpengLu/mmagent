from __future__ import annotations

from pathlib import Path

from configs.runtime import (
    PROJECT_ROOT,
    _load_runtime_paths,
    _load_runtime_payload,
    _load_runtime_settings,
)


ABLATION_ROOT = PROJECT_ROOT / "configs" / "ablations"
MODULE_SWITCHES = {
    "experience_generation",
    "experience_retrieval",
    "experience_rewrite",
    "experience_injection",
    "skill_generation",
    "skill_retrieval",
    "linked_experience_injection",
    "memory_rerank",
    "tool_plan",
    "failure_recovery",
    "validation_reflection",
}


def _ablation_configs() -> list[Path]:
    return sorted(ABLATION_ROOT.rglob("*.yaml"))


def test_ablation_configs_resolve_to_unique_output_roots() -> None:
    seen: dict[str, Path] = {}
    for config_path in _ablation_configs():
        payload = _load_runtime_payload(config_path)
        settings = _load_runtime_settings(payload, config_path)
        paths = _load_runtime_paths(PROJECT_ROOT, payload, config_path)

        assert "ablation" in settings
        assert settings["ablation"]["name"]
        output_root = str(paths.output_root)
        assert output_root not in seen, f"{config_path} collides with {seen[output_root]}"
        seen[output_root] = config_path


def test_module_ablation_switches_are_known() -> None:
    for config_path in sorted((ABLATION_ROOT / "modules").glob("*.yaml")):
        payload = _load_runtime_payload(config_path)
        modules = payload.get("ablation", {}).get("modules", {})
        assert modules, f"{config_path} must define ablation.modules"
        unknown = set(modules) - MODULE_SWITCHES
        assert not unknown, f"{config_path} defines unknown module switches: {sorted(unknown)}"

