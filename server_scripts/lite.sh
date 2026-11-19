#!/bin/bash

YAML_CONFIG=${1:-$(dirname $0)/config_lite.yml}
pids=()

cleanup() {
	kill "${pids[@]}" || true
	wait || true
}

max_num_seqs=16
max_batched=$(($max_num_seqs << 9))
HOST=0.0.0.0
GPT_COMMON=()
GPT_COMMON+=(--host $HOST)
GPT_COMMON+=(--max-model-len 16384)
GPT_COMMON+=(--max-num-seqs $max_num_seqs)
GPT_COMMON+=(--max-num-batched-tokens $max_batched)


./serve.sh \
	--model openai/gpt-oss-120b \
	--port 8011 \
	--gpu-memory-utilization 0.60 \
	"${GPT_COMMON[@]}" &
pids+=($!)

./serve.sh \
	--model openai/gpt-oss-120b \
	--port 8012 \
	--gpu-memory-utilization 0.25 \
	"${GPT_COMMON[@]}" &
pids+=$(!)

env LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true OPENAI_API_KEY=local-no-auth \
    uv run litellm --host $HOST --port 8010 --config $YAML_CONFIG
