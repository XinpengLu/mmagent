# Benchmark Processing

MMAgent uses two benchmark directories:

```text
benchmark/processed/<benchmark>/
benchmark/split/<benchmark>/
```

`processed/` is produced from raw downloads by `prepare_xskill_benchmarks.py`.
Each sample follows the same core schema:

```json
{
  "doc_id": "...",
  "problem": "<image>\nquestion text",
  "images": ["BenchmarkName/path/to/local_image.png"],
  "solution": "final answer",
  "data_source": "BenchmarkName"
}
```

`split/` is produced from `processed/` by `build_benchmark_subset.py`.
It is the directory used by the main train/validation/test pipeline.

## Convert Raw Data

Run from the repository root:

```bash
cd /mnt/f/XSkill
python mmagent/benchmark/prepare_xskill_benchmarks.py \
  --datasets all \
  --profile paper \
  --make-paper-splits \
  --strict
```

This writes `all.json`, `train.json`, and `test.json` under
`mmagent/benchmark/processed/<benchmark>/`.

## Build Runtime Splits

```bash
python mmagent/benchmark/build_benchmark_subset.py \
  --processed-root mmagent/benchmark/processed \
  --output-root mmagent/benchmark/split \
  --benchmarks TIR-Bench AgentVista MMSearch-Plus MMBrowseComp VisualToolBench BrowseComp-V3 VisBrowse-Bench Agentic-MME \
  --train-per-benchmark -1 \
  --val-per-benchmark 12 \
  --test-per-benchmark -1 \
  --stage-interval 10
```

Validation samples are held out from `processed/train.json`. Test samples are
never used for validation.

## Current Splits

| Benchmark | processed train | processed test | runtime train | runtime val | runtime test | Group fields |
|---|---:|---:|---:|---:|---:|---|
| TIR-Bench | 100 | 200 | 88 | 12 | 200 | `task` |
| AgentVista | 100 | 109 | 88 | 12 | 109 | `domain`, `subdomain` |
| MMSearch-Plus | 100 | 211 | 88 | 12 | 211 | `category`, `subtask`, `difficulty` |
| MMBrowseComp | 0 | 130 | 0 | 0 | 130 | `category`, `subtask`, `level` |
| VisualToolBench | 100 | 214 | 88 | 12 | 214 | `prompt_category`, `eval_focus`, `turncase` |
| BrowseComp-V3 | 100 | 200 | 88 | 12 | 200 | `category`, `sub_category`, `difficulty`, `level` |
| VisBrowse-Bench | 0 | 169 | 0 | 0 | 169 | `domain`, `sub_domain` |
| Agentic-MME | 100 | 318 | 88 | 12 | 318 | `level`, `match_type`, `has_search` |

The initial train/test split in `prepare_xskill_benchmarks.py` is stratified by
the group fields above. For datasets with `train=0`, the benchmark is treated as
OOD test-only.

The runtime split in `build_benchmark_subset.py` also uses the same group fields
when it holds validation samples out from training.

## Dataset Notes

`VisualToolBench` uses only single-turn hybrid-tool-reasoning tasks in the paper
profile.

`TIR-Bench` uses the five filtered categories: `refcoco`, `maze`, `instrument`,
`ocr`, and `contrast`. The train/test split is proportional by `task`; it no
longer uses a hand-written category count table.

`MMBrowseComp` and `VisBrowse-Bench` are kept as test-only OOD benchmarks.

`VisBrowse-Bench` raw records contain remote image URLs. The converter maps
those URLs to local files under:

```text
benchmark/raw/VisBrowse-Bench/VisBrowse_Bench_images/
```

Processed samples therefore use local image paths such as:

```text
VisBrowse-Bench/VisBrowse_Bench_images/1.png
```

`BrowseComp-V3` must be decrypted before conversion. The converter reads:

```text
benchmark/raw/BrowseComp-V3/data/decrypted.jsonl
```

The official `metadata` and `sub_goals` are preserved as fields, but the main
pipeline still uses only the problem, images, and final answer for rollout and
scoring.

`Agentic-MME` stores one task per `json/*.json`. The converter merges them into
standard JSON lists. Official input images come only from `images/`; `image_cause/`
is not used as model input. `process_evaluation` is preserved for analysis, but
it is not injected into the prompt.
