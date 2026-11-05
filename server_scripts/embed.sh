#!/bin/bash

uv run vllm serve \
    nomic-ai/modernbert-embed-base \
    --host 0.0.0.0 \
    --port 8002 --gpu-memory-utilization 0.20