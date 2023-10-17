---
license: apache-2.0
inference: false
---

# MistralLite Model

[MistralLite](https://huggingface.co/amazon/MistralLite) is a fine-tuned language model from [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), with enhanced capblities of processing long context (up to 32K tokens). MistralLite is useful for applications such as long context line and topic retrieval, summarization, question-answering, and etc. This repo demonstrates how to use the model in different frameworks.

> All the Jupyter Notebooks are suggested to run on a `ml.g5.2xlarge` SageMaker Notebook Instance.

- How to Use MistralFlite from [Python HuggingFace transformers](https://huggingface.co/docs/transformers/index): [an example Jupyter notebook](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/huggingface-transformers/example_usage.ipynb) 
- How to Serve MistralFlite on [Text-Generation-Inference 1.1.0](https://github.com/huggingface/text-generation-inference/tree/v1.1.0)
    - Original Container: [an example Jupyter notebook](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/tgi/example_usage.ipynb)
    - Custom Container supporting context length greater than 12K tokens: [an example Jupyter notebook](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/tgi-custom/example_usage.ipynb)
- How to Serve MistralFlite on [HuggingFace Text Generation Inference (TGI) container on SageMaker](https://github.com/awslabs/llm-hosting-container)
    - Built-in Container: [an example Jupyter notebook](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/sagemaker-tgi/example_usage.ipynb)
    - Custom Container supporting context length greater than 12K tokens: [an example Jupyter notebook](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/sagemaker-tgi-custom/example_usage.ipynb)
- How to Serve MistralFlite on [vLLM](https://github.com/vllm-project/vllm): [an example Jupyter notebook](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/tree/main/MistralLite/vllm/example_usage.ipynb)
