export model=amazon/FalconLite
export num_shard=4
export volume=/home/ubuntu/model # share a volume with the Docker container to avoid downloading weights every run

export max_input_length=12000 
export max_total_tokens=12001
export docker_image_id=falcon-lctx:0.9.2 # ghcr.io/huggingface/text-generation-inference:0.9.2
export falconlite_tgi_port=443

docker run -d --gpus all --env GPTQ_BITS=4 --env GPTQ_GROUPSIZE=128 --env DNTK_ALPHA_SCALER=0.25 \
                        --shm-size 1g -p $falconlite_tgi_port:80 -v $volume:/data $docker_image_id \
                        --model-id $model --num-shard $num_shard --max-input-length \
                        $max_input_length --max-total-tokens $max_total_tokens \
                        --max-batch-prefill-tokens $max_total_tokens \
                        --max-batch-total-tokens $max_total_tokens \
                        --trust-remote-code \
                        --quantize gptq