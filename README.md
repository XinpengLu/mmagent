# XSkill Agent Runner

This repository contains the code entry points for XSkill training, validation,
testing, memory learning, and benchmark reporting.

## Recommended Commands

Create the Python environment once, then install the project dependencies:

```bash
cd /mnt/f/XSkill/mmagent

conda create -n xskill python=3.11 -y
conda activate xskill
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After the environment is ready, use the single unified runner.

`run_full_benchmark.sh` assumes the Python environment is already active.
Runtime mode, benchmark list, paths, and defaults are loaded from
`configs/runtime.yaml`. The actual train/val/test and staged-validation flow
lives in:

```text
main_flow/
```

Build or refresh the fixed benchmark splits before experiments:

```bash
cd /mnt/f/XSkill/mmagent

python benchmark/build_benchmark_subset.py \
  --processed-root benchmark/processed \
  --output-root benchmark/split \
  --benchmarks TIR-Bench \
  --train-per-benchmark -1 \
  --val-per-benchmark 12 \
  --test-per-benchmark -1 \
  --stage-interval 10 \
  --seed 42
```

The runner expects these split files to exist. If they are missing, it exits
with an error instead of rebuilding them inside the experiment run.

Train with staged validation only. This does not run test:

```bash
RUN_ID=tir_fixed_v1_trainval \
bash run_full_benchmark.sh
```

Set this in `configs/runtime.yaml` before running:

```yaml
mode: train_val
benchmarks:
  - TIR-Bench
```

Test the memory from a finished run by using the same `RUN_ID` and changing
`mode` to `test`:

```bash
RUN_ID=tir_fixed_v1_trainval \
bash run_full_benchmark.sh
```

Train with staged validation, then test by setting `mode: train_val_test`:

```bash
RUN_ID=tir_fixed_v1_trainval \
bash run_full_benchmark.sh
```

Multiple benchmarks are configured in YAML:

```yaml
benchmarks:
  - TIR-Bench
  - VisualToolBench
  - AgentVista
```

## Output Location

Benchmark splits are fixed outside the run output:

```text
F:\XSkill\mmagent\benchmark\split\<Benchmark>\
  train.json
  val.json
  test.json
  stages\
    stages.json
    stage_001_10.json
    stage_002_20.json
    ...
```

When `RUN_ID` is set, the run is saved under:

```text
F:\XSkill\mmagent\output\repro_benchmark_runs\<RUN_ID>\
```

For a benchmark such as `TIR-Bench`, outputs are organized as:

```text
output\<benchmark>\
  train\
    metrics_at_k.json
    stages\
      train_stage_1_metrics_at_k.json
      train_stage_2_metrics_at_k.json
      ...
  val\
    val_stage_1\
      metrics_at_k.json
      feedback.json
      reflection.json
      status_calibration.json
      summary.json
    val_stage_2\
      ...
  test\
    metrics_at_k.json
    results.jsonl
  analysis\
    metrics\
      stage_metrics.json
      stage_metrics.csv
    figures\
      performance_curve.pdf
      efficiency_tokens_runtime.pdf
      memory_and_skill_usage.pdf
      validation_diagnostics.pdf
      ...
  validation_history.jsonl
  validation_history.csv
  best_validation_memory.json
```

Without `RUN_ID`, the script uses the default `OUTPUT_ROOT`, which defaults to:

```text
F:\XSkill\mmagent\output\full_benchmark
```

## Generate Analysis Figures

Validation analysis metrics are generated automatically at the end of `train_val`
and `train_val_test`. PDF figures are not generated during the run; generate
them manually after all runs finish:

```bash
python -m validation_tools.reporting \
  --run-root output/gpt-5.5/hive/TIR-Bench/output \
  --figures
```

The plotting code lives under:

```text
validation_tools/plot
```

## Memory Policy

During staged validation:

- Training uses the current evolving memory.
- Validation evaluates an isolated copy of the current stage memory.
- Validation feedback updates the validation copy and records stage quality.
- Final selection copies the validation-best checkpoint into the final memory location.

This keeps training meaningful while still using validation for model/memory
selection.

## Environment

The runner reads API keys and model names from:

```text
F:\XSkill\mmagent\configs\.env
```

Runtime mode, benchmarks, paths, and default parameters are loaded from:

```text
F:\XSkill\mmagent\configs\runtime.yaml
```

Benchmark data and outputs are project-local by default:

```text
F:\XSkill\mmagent\benchmark
F:\XSkill\mmagent\output
```

## Code Layout

```text
mmagent/
  run_full_benchmark.sh
  main_flow/
    main.py             # train / validation / test entry
    validation.py       # staged train-then-val loop
    phase.py            # one train / val / test phase
    rollout_runner.py   # rollout scheduling
    memory.py           # retrieval and prompt injection
    memory_update.py    # train-time experience / skill update
  experience/
    pipeline.py         # rollout-to-experience generation
    retriever.py        # experience retrieval and rewrite
    library.py          # merge / refine experience memory
  skill/
    generation.py       # experience-to-skill clustering
    retriever.py        # skill retrieval
    tool_plan.py        # executable tool-plan prompts
  configs/
    runtime.yaml
    tool_configs.yaml
  validation_tools/
    cli.py
    reporting.py
    plot/
  tools/
  engine/
  prompts/
    system_prompts.yaml                  # base system prompts
    answer_templates.py                  # final-answer synthesis template
    experience_generation_templates.py   # train-time experience generation
    experience_retrieval_templates.py    # test-time experience retrieval
    skill_generation_templates.py        # experience-to-skill generation
    skill_injection_templates.py         # test-time skill injection
    recovery_templates.py                # failure-recovery synthesis
    judge_templates.py                   # LLM judge prompts
    memory_rerank_prompts.yaml           # memory reranking prompts
```
