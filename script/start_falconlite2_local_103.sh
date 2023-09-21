export model=/data 
export num_shard=4
export volume=/home/ubuntu/model/falcon-lite2 # where model is downloaded to

export max_input_length=12000
export max_total_tokens=12001
export docker_image_id=falcon-lctx:1.0.3 # ghcr.io/huggingface/text-generation-inference:1.0.3
export falconlite_tgi_port=443

docker run -d --gpus all --env GPTQ_BITS=4 --env GPTQ_GROUPSIZE=128 \
                        --shm-size 1g -p $falconlite_tgi_port:80 -v $volume:/data $docker_image_id \
                        --model-id $model --num-shard $num_shard --max-input-length \
                        $max_input_length --max-total-tokens $max_total_tokens \
                        --max-batch-prefill-tokens $max_input_length \
                        --trust-remote-code \
                        --quantize gptq
