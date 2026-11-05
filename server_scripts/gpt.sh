#!/bin/bash

EXTRA_ARGS=($@)
model=${MODEL:-openai/gpt-oss-120b}
API_PORT=${API_PORT:-8001}

$RUN_PREFIX uv run vllm serve \
    "$model" \
    --host 0.0.0.0 \
    --port $API_PORT \
    --max-model-len "$HT_MAX_MODEL_LEN" \
    --max-num-seqs "$HT_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$HT_MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization 0.85 \
    "${EXTRA_ARGS[@]}"
