#!/bin/bash

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

API_KEY_1="${OPENAI_API_KEY}"

API_KEY_2="${OPENAI_API_KEY_2:-}"

OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_CHAT_COMPLETIONS_URL="${OPENAI_BASE_URL%/}/chat/completions"

# ============================================================================
# 推理模型配置
# ============================================================================

export REASONING_MODEL_NAME="${REASONING_MODEL_NAME:-gpt-4o}"

export REASONING_API_KEY=$API_KEY_1
export REASONING_END_POINT=$OPENAI_CHAT_COMPLETIONS_URL

# 可选：备用推理 API，用于轮询请求
export REASONING_API_KEY_2=$API_KEY_2
export REASONING_END_POINT_2=$OPENAI_CHAT_COMPLETIONS_URL

# ============================================================================
# 评测模型配置
# ============================================================================
export VERIFIER_MODEL_NAME="${VERIFIER_MODEL_NAME:-gpt-4o}"
export VERIFIER_API_KEY=$API_KEY_1
export VERIFIER_END_POINT=$OPENAI_BASE_URL

# ============================================================================
# 经验生成模型配置
# ============================================================================
export EXPERIENCE_MODEL_NAME="${EXPERIENCE_MODEL_NAME:-gpt-4o}"

export EXPERIENCE_API_KEY=$API_KEY_1
export EXPERIENCE_END_POINT=$OPENAI_CHAT_COMPLETIONS_URL

# 可选：经验生成的备用 API
export EXPERIENCE_API_KEY_2=$API_KEY_2
export EXPERIENCE_END_POINT_2=$OPENAI_CHAT_COMPLETIONS_URL

# ============================================================================
# 相似度检索嵌入模型配置
# ============================================================================
export EXPERIENCE_EMBEDDING_MODEL="${EXPERIENCE_EMBEDDING_MODEL:-text-embedding-3-small}"
export EXPERIENCE_EMBEDDING_API_KEY="${OPENAI_API_KEY}"
export EXPERIENCE_EMBEDDING_ENDPOINT="$OPENAI_BASE_URL"

# ============================================================================
# 函数调用与工具配置
# ============================================================================
export SERPAPI_KEY="${SERPAPI_KEY}"
export JINA_API_KEY="${JINA_API_KEY}"

export ENABLE_FUNCTION_CALLING="true"

# 可用工具：web_search、image_search、visit、code_interpreter、zoom
# 如需启用图片搜索，可使用下面这一行
# export ENABLED_TOOLS="web_search, image_search, visit, code_interpreter"
export ENABLED_TOOLS="web_search, visit, code_interpreter"

TOOL_CONFIG_PATH="eval/configs/tool_configs.yaml"

IMAGE_SEARCH_MAX_CALLS=5
WEB_SEARCH_MAX_CALLS=7

# ============================================================================
# 推理参数
# ============================================================================

MAX_TOTAL_TOKENS=32768
MAX_TURNS=20
MAX_IMAGES=100
TEMPERATURE=0.6
TOP_P=1.0

# ============================================================================
# 经验库参数
# ============================================================================

EXPERIENCE_MAX_OPS=3
EXPERIENCE_MAX_ITEMS=120

# 经验检索参数
EXPERIENCE_RETRIEVAL_TOP_K=3
MEMORY_NAME="${MEMORY_NAME:-${BENCHMARK:-test}}"
MEMORY_ROOT="${MEMORY_ROOT:-$DATA_ROOT/memory_bank/$MEMORY_NAME}"
export MEMORY_NAME MEMORY_ROOT
EXPERIENCE_LIBRARY="${EXPERIENCE_LIBRARY:-$MEMORY_ROOT/experiences.json}"

# ============================================================================
# 技能库参数
# ============================================================================

SKILL_LIBRARY="${SKILL_LIBRARY:-$MEMORY_ROOT/SKILL.md}"
SKILL_MAX_LENGTH=1000

# ============================================================================
# 运行设置
# ============================================================================

# SYSTEM_PROMPT_TYPE="multi_tool_agent_search"
SYSTEM_PROMPT_TYPE="multi_tool_agent"
# SYSTEM_PROMPT_TYPE="multi_tool_agent_code"
# SYSTEM_PROMPT_TYPE="agent_zoom"
# SYSTEM_PROMPT_TYPE="direct_cot"

IMAGE_DIR="${IMAGE_DIR:-$DATA_ROOT/benchmark}"
DATA_PATH="${DATA_PATH:-$DATA_ROOT/benchmark/VisualProbe_Test/val.json}"

OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/output/test_exskill_inference}"
LOG_OUTPUT_DIR="${LOG_OUTPUT_DIR:-$DATA_ROOT/logs/test_exskill_inference}"

NUM_WORKERS="${NUM_WORKERS:-8}"
EXPERIENCE_LARGE_BATCH="${EXPERIENCE_LARGE_BATCH:-8}"
ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-2}"

MAX_SAMPLES="${MAX_SAMPLES:-16}"

mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$LOG_OUTPUT_DIR")" "$(dirname "$EXPERIENCE_LIBRARY")" "$(dirname "$SKILL_LIBRARY")"

# ============================================================================
# 启动纯推理流程
# ============================================================================

python3 -u eval/infer_api.py \
    --input-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --output-dir $OUTPUT_DIR \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-turns $MAX_TURNS \
    --max-images $MAX_IMAGES \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --system-prompt-key $SYSTEM_PROMPT_TYPE \
    --num-workers $NUM_WORKERS \
    --tool-config-path $TOOL_CONFIG_PATH \
    --max-samples $MAX_SAMPLES \
    --rollouts-per-sample $ROLLOUTS_PER_SAMPLE \
    --image-search-max-calls $IMAGE_SEARCH_MAX_CALLS \
    --web-search-max-calls $WEB_SEARCH_MAX_CALLS \
    --skill-enable \
    --skill-library $SKILL_LIBRARY \
    --skill-inference \
    --experience-enable \
    --experience-library $EXPERIENCE_LIBRARY \
    --experience-retrieval \
    --experience-retrieval-top-k $EXPERIENCE_RETRIEVAL_TOP_K \
    --experience-retrieval-decomposition \
    --experience-retrieval-rewrite \
    2>&1 | tee $LOG_OUTPUT_DIR.log
    # --experience-online-generate \
    # --experience-library-update \
    # --experience-max-ops $EXPERIENCE_MAX_OPS \
    # --experience-large-batch $EXPERIENCE_LARGE_BATCH \
    # --experience-refine \
    # --experience-max-items $EXPERIENCE_MAX_ITEMS \
    # --skill-refine \
    # --skill-max-length $SKILL_MAX_LENGTH \
