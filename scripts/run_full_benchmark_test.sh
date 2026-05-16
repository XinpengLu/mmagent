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

BENCHMARK="${BENCHMARK:-TIR-Bench}"
BENCH_ROOT="${BENCH_ROOT:-$DATA_ROOT/benchmark_full/processed}"
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/output/full_benchmark}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs/full_benchmark}"
MEMORY_NAME="${MEMORY_NAME:-$BENCHMARK}"
MEMORY_ROOT="${MEMORY_ROOT:-$DATA_ROOT/memory_bank/$MEMORY_NAME}"
export MEMORY_NAME MEMORY_ROOT

DATA_PATH="${DATA_PATH:-$BENCH_ROOT/$BENCHMARK/$TEST_SPLIT_FILE}"
IMAGE_DIR="${IMAGE_DIR:-$BENCH_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$BENCHMARK/test}"
LOG_OUTPUT_DIR="${LOG_OUTPUT_DIR:-$LOG_ROOT/$BENCHMARK/test}"

NUM_WORKERS="${NUM_WORKERS:-8}"
ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-4}"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: missing test data file: $DATA_PATH"
    exit 1
fi

if [[ "${CHECK_PAPER_SPLIT:-1}" == "1" && "$BENCHMARK" == "TIR-Bench" ]]; then
    python3 - "$DATA_PATH" <<'PY'
from collections import Counter
import json
import sys

expected = {"refcoco": 60, "maze": 51, "instrument": 39, "ocr": 28, "contrast": 22}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    counts = Counter(item.get("task") for item in json.load(f))

if dict(counts) != expected:
    print("Error: TIR-Bench test split does not match XSkill Table 7.")
    print(f"Expected: {expected}")
    print(f"Found:    {dict(counts)}")
    print("Rerun scripts/prepare_xskill_benchmarks.py with --profile paper --make-paper-splits --seed 42.")
    raise SystemExit(2)

print(f"TIR-Bench test split check passed: {dict(counts)}")
PY
fi

count_json_items() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(len(json.load(f)))
PY
}

MAX_SAMPLES="${MAX_SAMPLES_TEST:-${MAX_SAMPLES:-$(count_json_items "$DATA_PATH")}}"

echo "======================================================================"
echo "Full benchmark inference"
echo "Benchmark: $BENCHMARK"
echo "Data: $DATA_PATH"
echo "Image root: $IMAGE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Rollouts per sample: $ROLLOUTS_PER_SAMPLE"
echo "Max samples: $MAX_SAMPLES"
echo "Memory bank: $MEMORY_ROOT"
echo "======================================================================"

IMAGE_DIR="$IMAGE_DIR" \
DATA_PATH="$DATA_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
LOG_OUTPUT_DIR="$LOG_OUTPUT_DIR" \
MAX_SAMPLES="$MAX_SAMPLES" \
NUM_WORKERS="$NUM_WORKERS" \
ROLLOUTS_PER_SAMPLE="$ROLLOUTS_PER_SAMPLE" \
DATA_ROOT="$DATA_ROOT" \
MEMORY_NAME="$MEMORY_NAME" \
MEMORY_ROOT="$MEMORY_ROOT" \
bash eval/run_exskill_inference.sh

python3 - "$OUTPUT_DIR" "$ROLLOUTS_PER_SAMPLE" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
k = sys.argv[2]
metrics_path = output_dir / "metrics_at_k.json"
summary_path = output_dir / "average_pass_at_k.json"

if not metrics_path.exists():
    print(f"Warning: {metrics_path} not found; no pass@k/average@k summary written.")
    raise SystemExit(0)

with metrics_path.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

k_metrics = metrics.get("k_metrics", {})
selected = k_metrics.get(k) or k_metrics.get(int(k))
if selected is None:
    print(f"Warning: k={k} not found in {metrics_path}.")
    raise SystemExit(0)

summary = {
    "k": int(k),
    f"pass@{k}": selected.get("pass_at_k", 0.0),
    f"average@{k}": selected.get("average_at_k", 0.0),
    "metrics_file": str(metrics_path),
}
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("======================================================================")
print(f"Recorded metrics: pass@{k}={summary[f'pass@{k}']:.4f}, average@{k}={summary[f'average@{k}']:.4f}")
print(f"Summary file: {summary_path}")
print("======================================================================")
PY
