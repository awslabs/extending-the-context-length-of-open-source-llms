# pip intall vllm
export NUM_SHARDS=1
export VLLM_PORT=8000 # change the client port accordingly

python -u -m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 --port $VLLM_PORT \
--model amazon/MistralLite \
--tensor-parallel-size $NUM_SHARDS \
--dtype bfloat16