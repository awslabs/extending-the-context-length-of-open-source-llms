# FalconLite Model

FalconLite is a quantized version of the [Falcon 40B SFT OASST-TOP1 model](https://huggingface.co/OpenAssistant/falcon-40b-sft-top1-560), capable of processing long (i.e. 11K tokens) input sequences while consuming 4x less GPU memory. By utilizing 4-bit [GPTQ quantization](https://github.com/PanQiWei/AutoGPTQ) and adapted [dynamic NTK](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/) RotaryEmbedding, FalconLite achieves a balance between latency, accuracy, and memory efficiency. With the ability to process 5x longer contexts than the original model, FalconLite is useful for applications such as topic retrieval, summarization, and question-answering. FalconLite can be deployed on a single AWS `g5.12x` instance with [TGI 0.9.2](https://github.com/huggingface/text-generation-inference/tree/v0.9.2), making it suitable for applications that require high performance in resource-constrained environments.

## Model Details

- **Developed by:** [AWS Contributors](https://github.com/orgs/aws-samples/teams/aws-prototype-ml-apac)
- **Model type:** [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b)
- **Language:** English
- **Quantized from weights:** [Falcon 40B SFT OASST-TOP1 model](https://huggingface.co/OpenAssistant/falcon-40b-sft-top1-560)
- **Modified from layers:** [Text-Generation-Inference 0.9.2](https://github.com/huggingface/text-generation-inference/tree/v0.9.2)
- **License:** Apache 2.0
- **Contact:** [GitHub issues](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/issues)
- **Blogpost:** [Extend the context length of Falcon40B to 10k](https://medium.com/@chenwuperth/extend-the-context-length-of-falcon40b-to-10k-85d81d32146f)

## Deploy FalconLite ##
SSH login to an AWS `g5.12x` instance with the [Deep Learning AMI](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-ubuntu-20-04/).

### Start LLM server
```bash
git clone https://github.com/awslabs/extending-the-context-length-of-open-source-llms.git falconlite-dev
cd falconlite-dev/script
./docker_build.sh
./start_falconlite.sh
```
### Perform inference
```
# after FalconLite has been completely started
pip install -r requirements-client.txt
python falconlite_client.py
```
**Important** - When using FalconLite for inference for the first time, it may require a brief 'warm-up' period that can take 10s of seconds. However, subsequent inferences should be faster and return results in a more timely manner. This warm-up period is normal and should not affect the overall performance of the system once the initialisation period has been completed.

## Evalution Result ##
We evaluated FalconLite against benchmarks that are specifically designed to assess the capabilities of LLMs in handling longer contexts. All evaluations were conducted without fine-tuning the model.

### Accuracy ###
|Eval task|Input length| Input length | Input length| Input length|
|----------|-------------|-------------|------------|-----------|
|          | 2800 ~ 3800| 5500 ~ 5600 |7500 ~ 8300 | 9300 ~ 11000 |
| [Topic Retrieval](https://lmsys.org/blog/2023-06-29-longchat/)    | 100%        | 100%       | 92%      | 92%     |
| [Line Retrieval](https://lmsys.org/blog/2023-06-29-longchat/#longeval-results)     | 38%         | 12%        |  8%      |  4%     |
| [Pass key Retrieval](https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py#L101) | 100%        | 100%       | 100%      | 100%     |

|Eval task| Test set Accuracy | Hard subset Accuracy|
|----------|-------------|-------------|
| [Question Answering with Long Input Texts](https://nyu-mll.github.io/quality/) | 46.9% | 40.8% |

### Performance ###
**metrics** = the average number of generated tokens per second (TPS) = $\frac{nb-generated-tokens}{end-to-end-response-time} $

The `end-to-end-response-time` = when the last token is generated - when the inference request is received 

|Instance| Input length | Input length| Input length|Input length|
|----------|-------------|-------------|------------|------------|
|      |     20 | 3300 | 5500 |10000 |
| g5.48x  | 22 tps | 12 tps | 12 tps | 12 tps |
| g5.12x  | 18 tps | 11 tps | 11 tps | 10 tps |

## Limitations ##
* Our evaluation shows that FalconLite's capability in `Line Retrieval` is limited, and requires further effort.
* While `g5.12x` is sufficient for FalconLite to handle 10K long contexts, a larger instance with more memory capcacity such as `g5.48x` is recommended for sustained, heavy workloads.
* Before using the FalconLite model, it is important to perform your own independent assessment, and take measures to ensure that your use would comply with your own specific quality control practices and standards, and that your use would comply with the local rules, laws, regulations, licenses and terms that apply to you, and your content.