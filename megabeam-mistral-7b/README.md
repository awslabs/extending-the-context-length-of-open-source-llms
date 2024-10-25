## Deploy [MegaBeam-Mistral-7B-512k](https://huggingface.co/aws-prototyping/MegaBeam-Mistral-7B-512k) on EC2 instances ##
### Memory Requirements

Supporting long context (e.g., half a million tokens in this case) typically requires GPUs with large VRAM capacity (such as AWS `p4d.24xlarge` instances). However, if you're working with limited resources, the open source community has provided quantized model versions (e.g. [here](https://huggingface.co/bartowski/MegaBeam-Mistral-7B-512k-GGUF)) that can process up to 390,000 tokens with just 24GB of VRAM.

### Start the vLLM server

On an instance with sufficient GPU RAM (e.g. `p4d.24xlarge`)
```shell
pip install vllm==0.6.2
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python3 -m vllm.entrypoints.openai.api_server \
        --model aws-prototyping/MegaBeam-Mistral-7B-512k \
        --tensor-parallel-size 8 \
        --enable-prefix-caching
```

### Obtain long context document
Download the entire [Mistral Inference git repo](https://github.com/mistralai/mistral-inference). Use this file as the long context, which has 211,501 tokens. Here we are using the awesome [unithub's OpenAPI](https://uithub.com/openapi.html#/) to download the github content.
```bash
wget -O mistral_repo_content.txt https://uithub.com/mistralai/mistral-inference?accept=text%2Fplain
```

### Build long contexts prompt for the Python client
```python
from openai import OpenAI
import argparse

user_prompt_tpl = """Analyze the git hub repository below to understand its structure, purpose, and functionality. Prepare to answer the questions based on your analysis.

{repo_content}

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=str, 
                        default="What can I do with this git repository?",
                        help="The question to ask about the git repository")
    args = parser.parse_args()
    
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id
    
    
    with open("mistral_repo_content.txt", "r") as fin:
        repo_content = fin.read()
    initial_user_prompt = user_prompt_tpl.format(repo_content=repo_content)

    chat_completion = client.chat.completions.create(
            messages = [
                {"role": "user", "content": initial_user_prompt}, # insert your long context here
                {"role": "assistant", "content": "Got it! Please ask your questions."},
                {"role": "user", "content": args.question} # insert your long context here
            ],
            model=model,
            temperature=0.0,
            timeout=3600,
            max_tokens=1024,
            n=1,
    )

    #print("Chat completion results:")
    outs = [choice.message.content for choice in chat_completion.choices]
    print(outs[0])
```

### Run the Python client

Notice that we included `--enable-prefix-caching` flag when launching the vLLM server. This means when we use the long context to call vLLM for the first time, the `time-to-first-token` can be quite long (~ 1 minute). However, thanks to [vLLM Prefix Caching](https://docs.vllm.ai/en/v0.6.2/automatic_prefix_caching/apc.html), subsequent calls would be much quicker (in the order of 25 seconds).
```bash
python megabeam_client.py 

python megabeam_client.py -q "show me the code that I can make an inference on the 8x22B Instruct model"
```