#!/bin/bash

env LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true OPENAI_API_KEY=local-no-auth \
    uv run litellm --host 0.0.0.0 --port 8003 \
    --model openai/gpt-oss-20b --api_base http://0.0.0.0:8001/v1 \
    --model openai/nomic-ai/modernbert-embed-base --api_base http://0.0.0.0:8002/v1 \