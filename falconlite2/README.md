---
license: apache-2.0
inference: false
---

# FalconLite2 Model

FalconLit2 is a fine-tuned and quantized [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b) language model, capable of processing long (up to 24K tokens) input sequences. By utilizing 4-bit [GPTQ quantization](https://github.com/PanQiWei/AutoGPTQ) and adapted RotaryEmbedding, FalconLite2 is able to process 10x longer contexts while consuming 4x less GPU memory than the original model. FalconLite2 is useful for applications such as topic retrieval, summarization, and question-answering. FalconLite2 can be deployed on a single AWS `g5.12x` instance with [TGI 1.0.3](https://github.com/huggingface/text-generation-inference/tree/v1.0.3), making it suitable for applications that require high performance in resource-constrained environments.

FalconLite2 evolves from [FalconLite](https://huggingface.co/amazon/FalconLite), and their similarities and differences are summarized below:
|Model|Fine-tuned on long contexts| Quantization | Max context length| RotaryEmbedding adaptation|
|----------|-------------:|-------------:|------------:|-----------:|
| FalconLite | No | 4-bit GPTQ |12K | [dNTK](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/) |
| FalconLite2 | Yes | 4-bit GPTQ |24K | rope_theta = 1000000 |

## Model Details

- **Developed by:** [AWS Contributors](https://github.com/orgs/aws-samples/teams/aws-prototype-ml-apac)
- **Model type:** [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b)
- **Language:** English
- **Finetuned from weights:** [Falcon 40B SFT OASST-TOP1 model](https://huggingface.co/OpenAssistant/falcon-40b-sft-top1-560)
- **Finetuned on data:** [SLidingEncoder and Decoder (SLED)](https://huggingface.co/datasets/tau/sled) and [(Long) Natural Questions (NQ)](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections#multi-passage-qa-from-natural-questions)
- **Deployed using framework:** [Text-Generation-Inference 1.0.3](https://github.com/huggingface/text-generation-inference/tree/v1.0.3)
- **Model License:** Apache 2.0
- **Contact:** [GitHub issues](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/issues)

## Deploy FalconLite ##
SSH login to an AWS `g5.12x` instance with the [Deep Learning AMI](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-ubuntu-20-04/).

### Start TGI server
```bash
git clone https://github.com/awslabs/extending-the-context-length-of-open-source-llms.git falconlite-dev
cd falconlite-dev/falconlite2
# this may take a while to build updated vLLM CUDA kernels
./docker_build.sh
./start_falconlite.sh
```
### Perform inference
```bash
# after FalconLite has been completely started
pip install -r requirements-client.txt

# test short context
python falconlite_client.py

# test long context (13400 tokens)
python falconlite_client.py -l
```
**Important** - Use the prompt template below for FalconLite2:
```
<|prompter|>What are the main challenges to support a long context for LLM?<|endoftext|><|assistant|>
```

**Important** - When using FalconLite2 for inference for the first time, it may require a brief 'warm-up' period that can take 10s of seconds. However, subsequent inferences should be faster and return results in a more timely manner. This warm-up period is normal and should not affect the overall performance of the system once the initialisation period has been completed.

## Evalution Result ##
We evaluated FalconLite against benchmarks that are specifically designed to assess the capabilities of LLMs in handling longer contexts.

### Accuracy ###
|Eval task|Input length| Input length | Input length| Input length| Input length|
|----------|-------------:|-------------:|------------:|-----------:|-----------:|
|          | 2851| 5568 |8313 | 11044 | 13780 
| [Topic Retrieval](https://lmsys.org/blog/2023-06-29-longchat/)    | 100%        | 100%       | 100%      | 100%     | 90% |

|Eval task|Input length| Input length | Input length| Input length| Input length|Input length|
|----------|-------------:|-------------:|------------:|-----------:|-----------:|-----------:|
|          | 3818| 5661 |7505 | 9354 | 11188 | 12657 
| [Line Retrieval](https://lmsys.org/blog/2023-06-29-longchat/#longeval-results)    | 84%        | 82%       | 66%      | 56%     | 62% | 34% |

|Eval task|Input length| Input length | Input length| Input length|
|----------|-------------:|-------------:|------------:|-----------:|
|          | 3264| 5396 |8329 | 10197 | 
| [Pass key Retrieval](https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py#L101)    | 100%        | 100%       | 100%      | 100%   |


|Eval task| Test set Accuracy | Hard subset Accuracy|
|----------|-------------:|-------------:|
| [Question Answering with Long Input Texts](https://nyu-mll.github.io/quality/) | 53.4% | 45.4% |

## Limitations ##
Before using the FalconLite model, it is important to perform your own independent assessment, and take measures to ensure that your use would comply with your own specific quality control practices and standards, and that your use would comply with the local rules, laws, regulations, licenses and terms that apply to you, and your content.