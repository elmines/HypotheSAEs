#!/bin/bash

EXTRA_ARGS=($@)
model=${MODEL:-openai/gpt-oss-20b}
API_PORT=${API_PORT:-8001}

$RUN_PREFIX uv run vllm serve \
    "$model" \
    --host 0.0.0.0 \
    --port $API_PORT \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.40 \
    "${EXTRA_ARGS[@]}"
