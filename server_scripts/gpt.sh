#!/bin/bash

EXTRA_ARGS=($@)
model=${MODEL:-openai/gpt-oss-120b}
mem_usage=${MEM_USAGE:-0.80}

API_PORT=${API_PORT:-8001}

$RUN_PREFIX uv run vllm serve \
    "$model" \
    --host 0.0.0.0 \
    --port $API_PORT \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization $mem_usage \
    "${EXTRA_ARGS[@]}"
