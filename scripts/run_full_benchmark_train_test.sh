#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${XSKILL_DATA_ROOT:-${DATA_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}}"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
cd "$CODE_ROOT"

BENCHMARK="${BENCHMARK:-TIR-Bench}"
BENCH_ROOT="${BENCH_ROOT:-$DATA_ROOT/benchmark_full/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/output/full_benchmark}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs/full_benchmark}"
MEMORY_NAME="${MEMORY_NAME:-$BENCHMARK}"
MEMORY_ROOT="${MEMORY_ROOT:-$DATA_ROOT/memory_bank/$MEMORY_NAME}"
ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXPERIENCE_LARGE_BATCH="${EXPERIENCE_LARGE_BATCH:-8}"

export BENCHMARK BENCH_ROOT OUTPUT_ROOT LOG_ROOT DATA_ROOT MEMORY_NAME MEMORY_ROOT
export ROLLOUTS_PER_SAMPLE NUM_WORKERS EXPERIENCE_LARGE_BATCH

echo "Running train + test for $BENCHMARK"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "MEMORY_NAME=$MEMORY_NAME"
echo "MEMORY_ROOT=$MEMORY_ROOT"
echo

bash scripts/run_full_benchmark_train.sh
bash scripts/run_full_benchmark_test.sh
