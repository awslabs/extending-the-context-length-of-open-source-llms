# This file is adapted from TGI 1.0.3 Dockerfile https://github.com/huggingface/text-generation-inference/blob/v1.0.3/Dockerfile
# Licensed under Hugging Face Optimized Inference License 1.0 https://github.com/huggingface/text-generation-inference/blob/v1.0.3/LICENSE

FROM debian:bullseye-slim as pytorch-install

ARG PYTORCH_VERSION=2.0.1
ARG PYTHON_VERSION=3.9
# Keep in sync with `server/pyproject.toml
ARG CUDA_VERSION=11.8
ARG MAMBA_VERSION=23.1.0-1
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*

# Install conda
# translating Docker's TARGETPLATFORM into mamba arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
         *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/mambaforge.sh -O  "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh"
RUN chmod +x ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

# Install pytorch
# On arm64 we exit with an error code
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  exit 1 ;; \
         *)              /opt/conda/bin/conda update -y conda &&  \
                         /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" pytorch==$PYTORCH_VERSION "pytorch-cuda=$(echo $CUDA_VERSION | cut -d'.' -f 1-2)"  ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

# CUDA kernels builder image
FROM pytorch-install as kernel-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install -c "nvidia/label/cuda-11.8.0"  cuda==11.8 && \
    /opt/conda/bin/conda clean -ya

# Re-Build vllm CUDA kernels
WORKDIR /usr/src
RUN git clone https://github.com/OlivierDehaene/vllm.git
WORKDIR /usr/src/vllm
RUN git fetch
# Obtain the version of vllm as per TGI 1.0.3
RUN git checkout d284b831c17f42a8ea63369a06138325f73c4cf9

# Copy bug fixes for issue https://github.com/vllm-project/vllm/issues/905
COPY vllm/attention_kernels.cu /usr/src/vllm/csrc/attention/attention_kernels.cu
COPY vllm/test_attention.py /usr/src/vllm/tests/kernels/test_attention.py
COPY vllm//utils.py  /usr/src/vllm/vllm/utils.py
COPY vllm/worker.py /usr/src/vllm/vllm/worker/worker.py

# Re-build vllm as per the fix
RUN python setup.py build

FROM ghcr.io/huggingface/text-generation-inference:1.0.3

# Copy builds artifacts from vllm builder
COPY --from=kernel-builder /usr/src/vllm/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages

RUN sed -i '250s/.*/           config=config, dim=self.head_size, base=1000000.0, device=weights.device/' /opt/conda/lib/python3.9/site-packages/text_generation_server/models/custom_modeling/flash_rw_modeling.py
RUN sed -i '57s/.*/            aliases={"lm_head.weight": ["transformer.word_embeddings.weight"], "transformer.word_embeddings.weight": ["lm_head.weight"]},/' /opt/conda/lib/python3.9/site-packages/text_generation_server/models/flash_rw.py