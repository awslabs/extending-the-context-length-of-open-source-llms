docker run -d --gpus all \
     -p 8000:8000 \
     ghcr.io/mistralai/mistral-src/vllm:latest \
    --host 0.0.0.0 \
    --model amazon/MistralLite \
    --dtype bfloat16