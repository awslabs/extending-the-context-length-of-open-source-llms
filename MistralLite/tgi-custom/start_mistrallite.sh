export model=amazon/MistralLite
export num_shard=$(nvidia-smi --list-gpus | wc -l)
export volume=$(pwd)/models # where model is downloaded to

export max_input_length=24000
export max_total_tokens=24576
export docker_image_id=mistrallite:1.1.0 # ghcr.io/huggingface/text-generation-inference:1.1.0
export mistrallite_tgi_port=443

docker run -d --gpus all --shm-size 1g -p $mistrallite_tgi_port:80 -v $volume:/data $docker_image_id \
                        --model-id $model --num-shard $num_shard --max-input-length \
                        $max_input_length --max-total-tokens $max_total_tokens \
                        --max-batch-prefill-tokens $max_input_length \
                        --dtype bfloat16 \
                        --trust-remote-code
