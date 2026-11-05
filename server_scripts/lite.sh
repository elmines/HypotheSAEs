#!/bin/bash

env LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true OPENAI_API_KEY=local-no-auth \
    uv run litellm --host 0.0.0.0 --port 8003 --config $(dirname $0)/config_lite.yml