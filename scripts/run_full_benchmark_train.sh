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
TRAIN_SPLIT_FILE="${TRAIN_SPLIT_FILE:-train.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/output/full_benchmark}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs/full_benchmark}"
MEMORY_NAME="${MEMORY_NAME:-$BENCHMARK}"
MEMORY_ROOT="${MEMORY_ROOT:-$DATA_ROOT/memory_bank/$MEMORY_NAME}"
export MEMORY_NAME MEMORY_ROOT

DATA_PATH="${DATA_PATH:-$BENCH_ROOT/$BENCHMARK/$TRAIN_SPLIT_FILE}"
IMAGE_DIR="${IMAGE_DIR:-$BENCH_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$BENCHMARK/train}"
LOG_OUTPUT_DIR="${LOG_OUTPUT_DIR:-$LOG_ROOT/$BENCHMARK/train}"

NUM_WORKERS="${NUM_WORKERS:-8}"
EXPERIENCE_LARGE_BATCH="${EXPERIENCE_LARGE_BATCH:-8}"
ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-4}"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: missing train data file: $DATA_PATH"
    exit 1
fi

if [[ "${CHECK_PAPER_SPLIT:-1}" == "1" && "$BENCHMARK" == "TIR-Bench" ]]; then
    python3 - "$DATA_PATH" <<'PY'
from collections import Counter
import json
import sys

expected = {"refcoco": 27, "maze": 27, "instrument": 22, "ocr": 15, "contrast": 9}
with open(sys.argv[1], "r", encoding="utf-8") as f:
    counts = Counter(item.get("task") for item in json.load(f))

if dict(counts) != expected:
    print("Error: TIR-Bench train split does not match XSkill Table 7.")
    print(f"Expected: {expected}")
    print(f"Found:    {dict(counts)}")
    print("Rerun scripts/prepare_xskill_benchmarks.py with --profile paper --make-paper-splits --seed 42.")
    raise SystemExit(2)

print(f"TIR-Bench train split check passed: {dict(counts)}")
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

MAX_SAMPLES="${MAX_SAMPLES_TRAIN:-${MAX_SAMPLES:-$(count_json_items "$DATA_PATH")}}"

echo "======================================================================"
echo "Full benchmark training"
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
EXPERIENCE_LARGE_BATCH="$EXPERIENCE_LARGE_BATCH" \
DATA_ROOT="$DATA_ROOT" \
MEMORY_NAME="$MEMORY_NAME" \
MEMORY_ROOT="$MEMORY_ROOT" \
bash eval/run_exskill_train.sh

echo "Training finished. Updated memory bank:"
echo "  $MEMORY_ROOT/experiences.json"
echo "  $MEMORY_ROOT/SKILL.md"
