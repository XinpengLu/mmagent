# Minimal Baselines

This folder runs two intentionally small baselines with the same rollout and metrics format as the main pipeline.

## No Tool

Uses only multimodal model reasoning with the `no_tool_direct_cot` prompt.

```bash
python -m baseline.run --config configs/runtime_no_tool.yaml
```

## Simple Tool

Uses the base model plus the benchmark's simple tool set, without experience, skill, memory, recovery, or tool-plan modules.

```bash
python -m baseline.run --config configs/runtime_with_tool.yaml
```

For a quick smoke test:

```bash
python -m baseline.run --config configs/runtime_no_tool.yaml --max-samples 1
python -m baseline.run --config configs/runtime_with_tool.yaml --max-samples 1
```

Outputs are written under `output/<reasoning_model_name>/<baseline_name>/<benchmark>/test` and keep the standard `results.jsonl`, per-sample rollout artifacts, `metrics_at_k.json`, and `average_pass_at_k.json` files.
