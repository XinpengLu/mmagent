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

# eval/run_exskill_*.sh reads these with shell expansion; keep empty defaults safe.
export SERPAPI_KEY="${SERPAPI_KEY:-}"
export JINA_API_KEY="${JINA_API_KEY:-}"

BENCHMARK="TIR-Bench"
TASK="${TASK:-maze}"
BENCH_ROOT="${BENCH_ROOT:-$DATA_ROOT/benchmark_full/processed}"
TRAIN_SOURCE="${TRAIN_SOURCE:-$BENCH_ROOT/$BENCHMARK/train.json}"
TEST_SOURCE="${TEST_SOURCE:-$BENCH_ROOT/$BENCHMARK/test.json}"
IMAGE_DIR="${IMAGE_DIR:-$BENCH_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/output/tir_${TASK}_realapi/$RUN_ID}"
FILTERED_DIR="${FILTERED_DIR:-$RUN_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/output}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs/tir_${TASK}_realapi/$RUN_ID}"

# Keep memory inside this run directory so existing $DATA_ROOT/memory_bank is untouched.
MEMORY_ROOT="${MEMORY_ROOT:-$RUN_ROOT/memory_bank}"
MEMORY_NAME="${MEMORY_NAME:-tir_${TASK}_realapi_$RUN_ID}"
EXPERIENCE_LIBRARY="$MEMORY_ROOT/experiences.json"
SKILL_LIBRARY="$MEMORY_ROOT/SKILL.md"
export MEMORY_ROOT MEMORY_NAME EXPERIENCE_LIBRARY SKILL_LIBRARY

ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-4}"
EXPERIENCE_LARGE_BATCH="${EXPERIENCE_LARGE_BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"

mkdir -p "$FILTERED_DIR" "$OUTPUT_ROOT" "$LOG_ROOT" "$MEMORY_ROOT"

TRAIN_DATA="$FILTERED_DIR/tir_${TASK}_train.json"
TEST_DATA="$FILTERED_DIR/tir_${TASK}_test.json"

case "$TASK" in
    maze)
        TRAIN_EXPECTED="${TRAIN_EXPECTED:-27}"
        TEST_EXPECTED="${TEST_EXPECTED:-51}"
        ;;
    refcoco)
        TRAIN_EXPECTED="${TRAIN_EXPECTED:-27}"
        TEST_EXPECTED="${TEST_EXPECTED:-60}"
        ;;
    instrument)
        TRAIN_EXPECTED="${TRAIN_EXPECTED:-22}"
        TEST_EXPECTED="${TEST_EXPECTED:-39}"
        ;;
    ocr)
        TRAIN_EXPECTED="${TRAIN_EXPECTED:-15}"
        TEST_EXPECTED="${TEST_EXPECTED:-28}"
        ;;
    contrast)
        TRAIN_EXPECTED="${TRAIN_EXPECTED:-9}"
        TEST_EXPECTED="${TEST_EXPECTED:-22}"
        ;;
    *)
        echo "Error: unsupported TIR-Bench task: $TASK"
        echo "Supported tasks: maze, refcoco, instrument, ocr, contrast"
        exit 2
        ;;
esac

filter_split() {
    local src="$1"
    local dst="$2"
    local split="$3"
    local expected="$4"

    if [[ ! -f "$src" ]]; then
        echo "Error: missing source split: $src"
        exit 1
    fi

    python3 - "$src" "$dst" "$TASK" "$split" "$expected" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
task = sys.argv[3]
split = sys.argv[4]
expected = int(sys.argv[5])

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

filtered = [item for item in data if item.get("task") == task]
if len(filtered) != expected:
    print(f"Error: expected {expected} {task} samples in {split}, found {len(filtered)}.")
    print(f"Source: {src}")
    raise SystemExit(2)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(filtered)} {task} samples for {split}: {dst}")
PY
}

count_json_items() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(len(json.load(f)))
PY
}

summarize_metrics() {
    local train_output="$1"
    local test_output="$2"
    local summary_path="$3"
    local k="$4"
    local task="$5"

    python3 - "$train_output" "$test_output" "$summary_path" "$k" "$task" <<'PY'
import json
import sys
from pathlib import Path

train_output = Path(sys.argv[1])
test_output = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
k = int(sys.argv[4])
task = sys.argv[5]

def read_metrics(output_dir: Path):
    metrics_path = output_dir / "metrics_at_k.json"
    fallback_path = output_dir / "metrics_summary.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        selected = metrics.get("k_metrics", {}).get(str(k)) or metrics.get("k_metrics", {}).get(k)
        if selected:
            return {
                "num_samples": metrics.get("num_samples"),
                "num_rollouts": metrics.get("num_rollouts"),
                f"pass@{k}": selected.get("pass_at_k", 0.0),
                f"average@{k}": selected.get("average_at_k", 0.0),
                "metrics_file": str(metrics_path),
            }

    if fallback_path.exists():
        with fallback_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        return {
            "num_samples": metrics.get("total_samples"),
            "overall_accuracy_score": metrics.get("overall_accuracy_score"),
            "metrics_file": str(fallback_path),
        }

    return {
        "warning": f"No metrics file found under {output_dir}",
    }

summary = {
    "task": task,
    "k": k,
    "train": read_metrics(train_output),
    "test": read_metrics(test_output),
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("======================================================================")
print("TIR-Bench real-API metrics")
for split in ("train", "test"):
    item = summary[split]
    print(f"{split}:")
    if f"pass@{k}" in item:
        print(f"  samples:   {item.get('num_samples')}")
        print(f"  rollouts:  {item.get('num_rollouts')}")
        print(f"  pass@{k}:   {item[f'pass@{k}']:.4f}")
        print(f"  average@{k}: {item[f'average@{k}']:.4f}")
    elif "overall_accuracy_score" in item:
        print(f"  samples:   {item.get('num_samples')}")
        print(f"  accuracy:  {item.get('overall_accuracy_score')}")
    else:
        print(f"  {item.get('warning')}")
    print(f"  file:      {item.get('metrics_file', 'N/A')}")
print(f"summary: {summary_path}")
print("======================================================================")
PY
}

if (( EXPERIENCE_LARGE_BATCH % ROLLOUTS_PER_SAMPLE != 0 )); then
    echo "Error: EXPERIENCE_LARGE_BATCH ($EXPERIENCE_LARGE_BATCH) must be a multiple of ROLLOUTS_PER_SAMPLE ($ROLLOUTS_PER_SAMPLE)."
    exit 2
fi

filter_split "$TRAIN_SOURCE" "$TRAIN_DATA" "train" "$TRAIN_EXPECTED"
filter_split "$TEST_SOURCE" "$TEST_DATA" "test" "$TEST_EXPECTED"

MAX_SAMPLES_TRAIN="${MAX_SAMPLES_TRAIN:-$(count_json_items "$TRAIN_DATA")}"
MAX_SAMPLES_TEST="${MAX_SAMPLES_TEST:-$(count_json_items "$TEST_DATA")}"

TRAIN_OUTPUT="$OUTPUT_ROOT/train"
TEST_OUTPUT="$OUTPUT_ROOT/test"
TRAIN_LOG="$LOG_ROOT/train"
TEST_LOG="$LOG_ROOT/test"

echo "======================================================================"
echo "TIR-Bench $TASK real-API train/test"
echo "Data root:      $DATA_ROOT"
echo "Train data:     $TRAIN_DATA"
echo "Test data:      $TEST_DATA"
echo "Image root:     $IMAGE_DIR"
echo "Run root:       $RUN_ROOT"
echo "Memory root:    $MEMORY_ROOT"
echo "Output root:    $OUTPUT_ROOT"
echo "Log root:       $LOG_ROOT"
echo "Rollouts:       $ROLLOUTS_PER_SAMPLE"
echo "Workers:        $NUM_WORKERS"
echo "Train samples:  $MAX_SAMPLES_TRAIN"
echo "Test samples:   $MAX_SAMPLES_TEST"
echo "======================================================================"

if [[ "$RUN_TRAIN" == "1" || "$RUN_TRAIN" == "true" || "$RUN_TRAIN" == "TRUE" ]]; then
    IMAGE_DIR="$IMAGE_DIR" \
    DATA_PATH="$TRAIN_DATA" \
    OUTPUT_DIR="$TRAIN_OUTPUT" \
    LOG_OUTPUT_DIR="$TRAIN_LOG" \
    MAX_SAMPLES="$MAX_SAMPLES_TRAIN" \
    NUM_WORKERS="$NUM_WORKERS" \
    ROLLOUTS_PER_SAMPLE="$ROLLOUTS_PER_SAMPLE" \
    EXPERIENCE_LARGE_BATCH="$EXPERIENCE_LARGE_BATCH" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    BENCHMARK="$BENCHMARK" \
    MEMORY_NAME="$MEMORY_NAME" \
    MEMORY_ROOT="$MEMORY_ROOT" \
    EXPERIENCE_LIBRARY="$EXPERIENCE_LIBRARY" \
    SKILL_LIBRARY="$SKILL_LIBRARY" \
    bash eval/run_exskill_train.sh
fi

if [[ "$RUN_TEST" == "1" || "$RUN_TEST" == "true" || "$RUN_TEST" == "TRUE" ]]; then
    IMAGE_DIR="$IMAGE_DIR" \
    DATA_PATH="$TEST_DATA" \
    OUTPUT_DIR="$TEST_OUTPUT" \
    LOG_OUTPUT_DIR="$TEST_LOG" \
    MAX_SAMPLES="$MAX_SAMPLES_TEST" \
    NUM_WORKERS="$NUM_WORKERS" \
    ROLLOUTS_PER_SAMPLE="$ROLLOUTS_PER_SAMPLE" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    BENCHMARK="$BENCHMARK" \
    MEMORY_NAME="$MEMORY_NAME" \
    MEMORY_ROOT="$MEMORY_ROOT" \
    EXPERIENCE_LIBRARY="$EXPERIENCE_LIBRARY" \
    SKILL_LIBRARY="$SKILL_LIBRARY" \
    bash eval/run_exskill_inference.sh
fi

summarize_metrics "$TRAIN_OUTPUT" "$TEST_OUTPUT" "$RUN_ROOT/metrics_tir_${TASK}_realapi.json" "$ROLLOUTS_PER_SAMPLE" "$TASK"
