# Ablation configs

Ablations live in three groups:

- `modules/`: paper-level module ablations that correspond to the main HIVE mechanisms.
- `modules_backup/`: fine-grained diagnostic switches kept for later debugging, but not part of the main ablation table.
- `params/`: keep the pipeline intact, but change one hyperparameter family.

Run a module ablation:

```bash
python -m main_flow.main --config configs/ablations/modules/no_skill.yaml
```

Run a parameter ablation:

```bash
python -m main_flow.main --config configs/ablations/params/experience_max_items_30.yaml
```

Each file extends `configs/runtime.yaml`. A config with:

```yaml
ablation:
  kind: module
  name: no_skill
```

writes to:

```text
output/{reasoning_model_name}/{base_name}_ablation_{kind}_{name}/
```

For example:

```text
output/gpt-5.5/mmagent_ablation_module_no_skill/
```

This keeps module ablations, parameter ablations, and the default run from
overwriting each other.

## Test-only TIR-Bench ablations

Some ablations only change inference-time memory usage, so their configs run
`mode: test` on `TIR-Bench` and reuse the final memory from the full HIVE run:

```yaml
memory:
  reuse_full_hive_memory_for_test:
    TIR-Bench: TIR-Bench
```

The source run defaults to `output/{reasoning_model_name}/hive/`. For example,
`configs/ablations/modules/no_memory_rerank.yaml` writes its test outputs under
`output/{reasoning_model_name}/hive_ablation_module_no_memory_rerank/`, but reads
memory from `output/{reasoning_model_name}/hive/TIR-Bench/memory/`.

## Primary module ablations

Keep the main `modules/` directory aligned with the paper-level claims:

- `no_memory`: remove all memory to sanity-check the gap to a tool-only agent.
- `no_skill`: remove the skill layer and keep flat experience memory.
- `no_linked_experience`: keep skills, but remove linked supporting experiences.
- `no_memory_rerank`: replace reranked memory selection with similarity-only selection.
- `no_tool_plan`: remove the memory-derived tool-plan policy.
- `no_failure_recovery`: remove recovery-experience synthesis.
- `no_validation`: remove validation-guided reflection/evolution.

The backup directory keeps narrower implementation diagnostics such as disabling
experience generation, experience rewriting, skill generation, or skill-only
injection. Restore those only when a reviewer asks for a lower-level diagnostic
or when debugging a specific mechanism.

## Module switches

The base runtime enables all switches under `ablation.modules`. Module ablation
YAMLs override only the relevant switches:

- `experience_generation`: train-time experience generation and memory writes.
- `experience_retrieval`: test/validation-time direct experience retrieval.
- `experience_rewrite`: task-conditioned rewriting of retrieved experiences.
- `experience_injection`: prompt injection of retrieved experiences.
- `skill_generation`: train-time clustering/skill abstraction.
- `skill_retrieval`: test/validation-time skill retrieval and injection.
- `linked_experience_injection`: injection of experiences linked from selected skills.
- `memory_rerank`: LLM reranking over retrieved skills/experiences.
- `tool_plan`: mandatory executable tool-plan injection.
- `failure_recovery`: synthesis of recovery experiences before skill clustering.
- `validation_reflection`: validation-set reflection write-back.

## Sanity check

Run this after adding new ablation YAMLs:

```bash
python -m pytest tests/test_ablation_configs.py
```
