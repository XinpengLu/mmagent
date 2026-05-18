#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${XSKILL_DATA_ROOT:-${DATA_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}}"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
cd "$CODE_ROOT"
export DATA_ROOT

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    for env_file in "$CODE_ROOT/.env" "$CODE_ROOT/.env.local" "$DATA_ROOT/.env" "$DATA_ROOT/.env.local"; do
        if [[ -f "$env_file" ]]; then
            source "$env_file"
            break
        fi
    done
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set. Run: source .env"
    exit 1
fi

BENCH_ROOT="${BENCH_ROOT:-$DATA_ROOT/benchmark_full/processed}"
IMAGE_DIR="${IMAGE_DIR:-$BENCH_ROOT}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/output/multibench_smoke/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs/multibench_smoke/$RUN_ID}"

# Keep default coverage small but diverse. Override with e.g.
# SMOKE_SPECS="tir:maze tir:refcoco bench:AgentVista"
SMOKE_SPECS="${SMOKE_SPECS:-tir:maze tir:refcoco tir:ocr bench:VisualToolBench bench:AgentVista}"

ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-4}"
EXPERIENCE_LARGE_BATCH="${EXPERIENCE_LARGE_BATCH:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES_TRAIN="${MAX_SAMPLES_TRAIN:-2}"
MAX_SAMPLES_TEST="${MAX_SAMPLES_TEST:-4}"
MAX_TURNS="${MAX_TURNS:-6}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-1.0}"
EXPERIENCE_MAX_OPS="${EXPERIENCE_MAX_OPS:-3}"
EXPERIENCE_GENERATION_WORKERS="${EXPERIENCE_GENERATION_WORKERS:-1}"
SKILL_CLUSTER_MIN_EXPERIENCES="${SKILL_CLUSTER_MIN_EXPERIENCES:-1}"
SKILL_CLUSTER_MAX_CLUSTERS="${SKILL_CLUSTER_MAX_CLUSTERS:-12}"
MEMORY_RERANK_TOP_SKILLS="${MEMORY_RERANK_TOP_SKILLS:-2}"
MEMORY_RERANK_TOP_EXPERIENCES="${MEMORY_RERANK_TOP_EXPERIENCES:-6}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-multi_tool_agent}"
ENABLED_TOOLS="${ENABLED_TOOLS:-web_search, visit, code_interpreter}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
K="${K:-$ROLLOUTS_PER_SAMPLE}"

if (( EXPERIENCE_LARGE_BATCH % ROLLOUTS_PER_SAMPLE != 0 )); then
    echo "Error: EXPERIENCE_LARGE_BATCH ($EXPERIENCE_LARGE_BATCH) must be a multiple of ROLLOUTS_PER_SAMPLE ($ROLLOUTS_PER_SAMPLE)."
    exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

summarize_one() {
    local label="$1"
    local root="$2"
    local summary_path="$3"
    python3 - "$label" "$root" "$summary_path" "$K" <<'PY'
import json
import sys
from pathlib import Path

label = sys.argv[1]
root = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
k = int(sys.argv[4])

def read_split(split: str):
    out = root / "output" / split
    metrics_at_k = out / "metrics_at_k.json"
    metrics_summary = out / "metrics_summary.json"
    if metrics_at_k.exists():
        data = json.load(open(metrics_at_k, encoding="utf-8"))
        selected = data.get("k_metrics", {}).get(str(k)) or data.get("k_metrics", {}).get(k) or {}
        return {
            "num_samples": data.get("num_samples"),
            "num_rollouts": data.get("num_rollouts"),
            f"pass@{k}": selected.get("pass_at_k", 0.0),
            f"average@{k}": selected.get("average_at_k", 0.0),
            "metrics_file": str(metrics_at_k),
        }
    if metrics_summary.exists():
        data = json.load(open(metrics_summary, encoding="utf-8"))
        return {
            "num_samples": data.get("total_samples"),
            "overall_accuracy_score": data.get("overall_accuracy_score"),
            "metrics_file": str(metrics_summary),
        }
    return {"warning": f"No metrics found under {out}"}

summary = {
    "label": label,
    "k": k,
    "train": read_split("train"),
    "test": read_split("test"),
    "run_root": str(root),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

run_tir_spec() {
    local task="$1"
    local label="tir_${task}"
    local root="$RUN_ROOT/$label"
    local log="$LOG_ROOT/$label.outer.log"
    echo "======================================================================"
    echo "Running $label"
    echo "root=$root"
    echo "log=$log"
    echo "======================================================================"
    TASK="$task" \
    RUN_ID="$RUN_ID" \
    RUN_ROOT="$root" \
    LOG_ROOT="$root/logs" \
    MEMORY_ROOT="$root/memory_bank" \
    IMAGE_DIR="$IMAGE_DIR" \
    RUN_TRAIN="$RUN_TRAIN" \
    RUN_TEST="$RUN_TEST" \
    MAX_SAMPLES_TRAIN="$MAX_SAMPLES_TRAIN" \
    MAX_SAMPLES_TEST="$MAX_SAMPLES_TEST" \
    NUM_WORKERS="$NUM_WORKERS" \
    ROLLOUTS_PER_SAMPLE="$ROLLOUTS_PER_SAMPLE" \
    EXPERIENCE_LARGE_BATCH="$EXPERIENCE_LARGE_BATCH" \
    EXPERIENCE_GENERATION_WORKERS="$EXPERIENCE_GENERATION_WORKERS" \
    EXPERIENCE_MAX_OPS="$EXPERIENCE_MAX_OPS" \
    SKILL_CLUSTER_MIN_EXPERIENCES="$SKILL_CLUSTER_MIN_EXPERIENCES" \
    SKILL_CLUSTER_MAX_CLUSTERS="$SKILL_CLUSTER_MAX_CLUSTERS" \
    MEMORY_RERANK_TOP_SKILLS="$MEMORY_RERANK_TOP_SKILLS" \
    MEMORY_RERANK_TOP_EXPERIENCES="$MEMORY_RERANK_TOP_EXPERIENCES" \
    MAX_TURNS="$MAX_TURNS" \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    SYSTEM_PROMPT_TYPE="$SYSTEM_PROMPT_TYPE" \
    ENABLED_TOOLS="$ENABLED_TOOLS" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    bash scripts/run_tir_maze_realapi_train_test.sh 2>&1 | tee "$log"
    summarize_one "$label" "$root" "$root/metrics_summary_smoke.json"
}

run_bench_spec() {
    local bench="$1"
    local safe="${bench//[^A-Za-z0-9_]/_}"
    local label="bench_${safe}"
    local root="$RUN_ROOT/$label"
    local train_data="$BENCH_ROOT/$bench/train.json"
    local test_data="$BENCH_ROOT/$bench/test.json"
    local memory_root="$root/memory_bank"
    local train_output="$root/output/train"
    local test_output="$root/output/test"
    local train_log="$LOG_ROOT/$label.train"
    local test_log="$LOG_ROOT/$label.test"

    echo "======================================================================"
    echo "Running $label"
    echo "root=$root"
    echo "======================================================================"
    mkdir -p "$memory_root" "$root/output" "$LOG_ROOT"

    if [[ "$RUN_TRAIN" == "1" || "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "TRUE" ]]; then
        if [[ -f "$train_data" ]]; then
            BENCHMARK="$bench" \
            MEMORY_ROOT="$memory_root" \
            IMAGE_DIR="$IMAGE_DIR" \
            DATA_PATH="$train_data" \
            OUTPUT_DIR="$train_output" \
            LOG_OUTPUT_DIR="$train_log" \
            MAX_SAMPLES="$MAX_SAMPLES_TRAIN" \
            NUM_WORKERS="$NUM_WORKERS" \
            ROLLOUTS_PER_SAMPLE="$ROLLOUTS_PER_SAMPLE" \
            EXPERIENCE_LARGE_BATCH="$EXPERIENCE_LARGE_BATCH" \
            EXPERIENCE_GENERATION_WORKERS="$EXPERIENCE_GENERATION_WORKERS" \
            EXPERIENCE_MAX_OPS="$EXPERIENCE_MAX_OPS" \
            SKILL_CLUSTER_MIN_EXPERIENCES="$SKILL_CLUSTER_MIN_EXPERIENCES" \
            SKILL_CLUSTER_MAX_CLUSTERS="$SKILL_CLUSTER_MAX_CLUSTERS" \
            MEMORY_RERANK_TOP_SKILLS="$MEMORY_RERANK_TOP_SKILLS" \
            MEMORY_RERANK_TOP_EXPERIENCES="$MEMORY_RERANK_TOP_EXPERIENCES" \
            MAX_TURNS="$MAX_TURNS" \
            TEMPERATURE="$TEMPERATURE" \
            TOP_P="$TOP_P" \
            SYSTEM_PROMPT_TYPE="$SYSTEM_PROMPT_TYPE" \
            ENABLED_TOOLS="$ENABLED_TOOLS" \
            SKIP_COMPLETED="$SKIP_COMPLETED" \
            bash eval/run_exskill_train.sh
        else
            echo "Skipping train for $bench: $train_data not found"
        fi
    fi

    if [[ "$RUN_TEST" == "1" || "$RUN_TEST" == "true" || "$RUN_TEST" == "TRUE" ]]; then
        if [[ -f "$test_data" ]]; then
            BENCHMARK="$bench" \
            MEMORY_ROOT="$memory_root" \
            IMAGE_DIR="$IMAGE_DIR" \
            DATA_PATH="$test_data" \
            OUTPUT_DIR="$test_output" \
            LOG_OUTPUT_DIR="$test_log" \
            MAX_SAMPLES="$MAX_SAMPLES_TEST" \
            NUM_WORKERS="$NUM_WORKERS" \
            ROLLOUTS_PER_SAMPLE="$ROLLOUTS_PER_SAMPLE" \
            EXPERIENCE_LARGE_BATCH="$EXPERIENCE_LARGE_BATCH" \
            EXPERIENCE_MAX_OPS="$EXPERIENCE_MAX_OPS" \
            SKILL_CLUSTER_MIN_EXPERIENCES="$SKILL_CLUSTER_MIN_EXPERIENCES" \
            SKILL_CLUSTER_MAX_CLUSTERS="$SKILL_CLUSTER_MAX_CLUSTERS" \
            MEMORY_RERANK_TOP_SKILLS="$MEMORY_RERANK_TOP_SKILLS" \
            MEMORY_RERANK_TOP_EXPERIENCES="$MEMORY_RERANK_TOP_EXPERIENCES" \
            MAX_TURNS="$MAX_TURNS" \
            TEMPERATURE="$TEMPERATURE" \
            TOP_P="$TOP_P" \
            SYSTEM_PROMPT_TYPE="$SYSTEM_PROMPT_TYPE" \
            ENABLED_TOOLS="$ENABLED_TOOLS" \
            SKIP_COMPLETED="$SKIP_COMPLETED" \
            bash eval/run_exskill_inference.sh
        else
            echo "Skipping test for $bench: $test_data not found"
        fi
    fi

    summarize_one "$label" "$root" "$root/metrics_summary_smoke.json"
}

echo "Run root: $RUN_ROOT"
echo "Specs: $SMOKE_SPECS"
echo "Rollouts: $ROLLOUTS_PER_SAMPLE, train samples: $MAX_SAMPLES_TRAIN, test samples: $MAX_SAMPLES_TEST"

for spec in $SMOKE_SPECS; do
    case "$spec" in
        tir:*) run_tir_spec "${spec#tir:}" ;;
        bench:*) run_bench_spec "${spec#bench:}" ;;
        *) echo "Unknown spec: $spec" >&2; exit 2 ;;
    esac
done

python3 - "$RUN_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
items = []
for path in sorted(root.glob("*/metrics_summary_smoke.json")):
    try:
        items.append(json.load(open(path, encoding="utf-8")))
    except Exception as exc:
        items.append({"path": str(path), "error": str(exc)})
summary_path = root / "multibench_summary.json"
json.dump(items, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("======================================================================")
print(f"Multi-benchmark smoke summary: {summary_path}")
for item in items:
    label = item.get("label", item.get("path", "unknown"))
    test = item.get("test", {})
    k = item.get("k")
    avg = test.get(f"average@{k}") if k else None
    pas = test.get(f"pass@{k}") if k else None
    print(f"{label}: pass@{k}={pas} average@{k}={avg} warning={test.get('warning')}")
print("======================================================================")
PY
