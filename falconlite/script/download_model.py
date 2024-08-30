from huggingface_hub import snapshot_download

snapshot_download(repo_id="amazon/FalconLite", 
                  local_dir="/home/ubuntu/model",
                  local_dir_use_symlinks=False)