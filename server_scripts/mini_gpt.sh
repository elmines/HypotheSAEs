#!/bin/bash

EXTRA_ARGS=($@)
model=${MODEL:-openai/gpt-oss-20b}
mem_usage=${MEM_USAGE:-0.20}

API_PORT=${API_PORT:-8001}
set -x
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(($MAX_NUM_SEQS << 9))}


$RUN_PREFIX uv run --active vllm serve \
    "$model" \
    --host 0.0.0.0 \
    --port $API_PORT \
    --max-model-len 16384 \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --gpu-memory-utilization $mem_usage \
    "${EXTRA_ARGS[@]}"
