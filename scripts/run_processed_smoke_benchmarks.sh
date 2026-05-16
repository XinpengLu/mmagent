#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${XSKILL_DATA_ROOT:-${DATA_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}}"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
cd "$CODE_ROOT" || exit 1
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

BENCH_ROOT="${BENCH_ROOT:-$DATA_ROOT/benchmark_full/processed_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/output/gpt_processed_smoke}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs/gpt_processed_smoke}"
MEMORY_ROOT="${MEMORY_ROOT:-$DATA_ROOT/memory_bank/test}"
export MEMORY_ROOT

MAX_SAMPLES="${MAX_SAMPLES:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-1}"

BENCHMARKS=(
    "AgentVista"
    "MMBrowseComp"
    "MMSearch-Plus"
    "TIR-Bench"
    "VisualToolBench"
)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

FAILED=()

for BENCH in "${BENCHMARKS[@]}"; do
    DATA_PATH="$BENCH_ROOT/$BENCH/all.json"

    echo
    echo "======================================================================"
    echo "Testing benchmark: $BENCH"
    echo "Data path: $DATA_PATH"
    echo "Image root: $BENCH_ROOT"
    echo "Output: $OUTPUT_ROOT/$BENCH"
    echo "Memory bank: $MEMORY_ROOT"
    echo "======================================================================"

    if [[ ! -f "$DATA_PATH" ]]; then
        echo "Missing data file: $DATA_PATH"
        FAILED+=("$BENCH")
        continue
    fi

    IMAGE_DIR="$BENCH_ROOT" \
    DATA_PATH="$DATA_PATH" \
    OUTPUT_DIR="$OUTPUT_ROOT/$BENCH" \
    LOG_OUTPUT_DIR="$LOG_ROOT/$BENCH" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    NUM_WORKERS="$NUM_WORKERS" \
    ROLLOUTS_PER_SAMPLE="$ROLLOUTS_PER_SAMPLE" \
    DATA_ROOT="$DATA_ROOT" \
    MEMORY_ROOT="$MEMORY_ROOT" \
    bash eval/run_exskill_inference.sh

    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
        echo "Benchmark failed: $BENCH (exit code: $STATUS)"
        FAILED+=("$BENCH")
    fi
done

echo
echo "======================================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All processed_smoke benchmarks finished."
    echo "Outputs: $OUTPUT_ROOT"
    echo "Logs: $LOG_ROOT"
else
    echo "Some benchmarks failed: ${FAILED[*]}"
    echo "Check logs under: $LOG_ROOT"
    exit 1
fi
