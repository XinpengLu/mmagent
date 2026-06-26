# Baselines

This folder runs baselines with the same rollout artifacts and metrics format as the main pipeline.

## No Tool

Uses only multimodal model reasoning with the `no_tool_direct_cot` prompt.

```bash
python -m baseline.run --config configs/runtime_no_tool.yaml
```

## Simple Tool / ReAct

Uses the base model plus the benchmark's simple tool set, without experience, skill, memory, recovery, or tool-plan modules.

```bash
python -m baseline.run --config configs/runtime_with_tool.yaml
```

## Reflexion

Uses the same benchmark-specific tool set as `with_tool`, but runs the `k` rollouts for each sample sequentially. After a failed rollout, it asks the model to write a short reflection from the trajectory and judge feedback, then injects that reflection into the next rollout.

```bash
python -m baseline.run --config configs/runtime_reflexion_qwen.yaml
```

Intermediate files:

- `reflexion_memory.json`: reflections generated for the current sample.
- standard rollout folders: `rollout_*/traj.jsonl`, `rollout_*/metrics.json`.

## ExpeL

Uses the same benchmark-specific tool set as `with_tool`. It first runs a lightweight training pass on the benchmark's `train.json` split to extract reusable ExpeL-style rules, then retrieves top-k rules for each test sample and injects them into the prompt.

```bash
python -m baseline.run --config configs/runtime_expel_qwen.yaml
```

Intermediate files:

- `train/`: one training rollout per sampled train item.
- `train/expel_rules.json`: extracted reusable rules for that benchmark.
- `test/*/expel_selected_rules.json`: rules retrieved for each test sample.

## Trace2Skill

Uses the same benchmark-specific tool set as `with_tool`. It first runs one training rollout per sampled train item from `train.json`, distills trajectory-local lessons, consolidates them into one global skill guide, and injects that guide into every test rollout.

```bash
python -m baseline.run --config configs/runtime_trace2skill_qwen.yaml
```

Intermediate files:

- `train/`: one training rollout per sampled train item.
- `train/trace2skill_local_lessons.json`: local lessons distilled from training trajectories.
- `train/trace2skill_guide.md`: consolidated global skill guide injected at test time.

For a quick smoke test:

```bash
python -m baseline.run --config configs/runtime_no_tool.yaml --max-samples 1
python -m baseline.run --config configs/runtime_with_tool.yaml --max-samples 1
python -m baseline.run --config configs/runtime_reflexion_qwen.yaml --max-samples 1
python -m baseline.run --config configs/runtime_expel_qwen.yaml --max-samples 1
python -m baseline.run --config configs/runtime_trace2skill_qwen.yaml --max-samples 1
```

Outputs are written under `output/<reasoning_model_name>/<baseline_name>/<benchmark>/test` and keep the standard `results.jsonl`, per-sample rollout artifacts, `metrics_at_k.json`, and `average_pass_at_k.json` files.
