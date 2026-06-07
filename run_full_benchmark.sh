#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNTIME_CONFIG="${MMAGENT_RUNTIME_CONFIG:-$SCRIPT_DIR/configs/runtime.yaml}"
echo "Runtime config: $RUNTIME_CONFIG"

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m main_flow.main
