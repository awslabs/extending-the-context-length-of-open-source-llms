from text_generation import Client
import argparse

SERVER_PORT = 443 # e.g. check `falconlite_tgi_port` in start_falconlite.sh
SERVER_HOST = "localhost"
SERVER_URL = f"{SERVER_HOST}:{SERVER_PORT}"
oa_client = Client(f"http://{SERVER_URL}", timeout=60)

def invoke_falconlite(prompt, 
                      random_seed=1, 
                      max_new_tokens=250, 
                      print_stream=True,
                      assist_role=True):
    if (assist_role):
        prompt = f"<|prompter|>{prompt}<|endoftext|><|assistant|>"
    output = ""
    for response in oa_client.generate_stream(
        prompt,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        typical_p=0.2,
        temperature=None,
        truncate=None,
        seed=random_seed,
    ):
        if hasattr(response, "token"):
            if not response.token.special:
                snippet = response.token.text
                output += snippet
                if (print_stream):
                    print(snippet, end='', flush=True)
    return output

if __name__ == "__main__":
    """
    Replace `prompt` with your long context here!
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--long_context',
                        action='store_true', 
                        default=False)
    args = parser.parse_args()
    if args.long_context:
        # use a context of over 13,400 tokens
        with open("example_long_ctx.txt", "r") as fin:
            prompt = fin.read()
        prompt = prompt.format(
            my_question="please tell me how does pgvector help with Generative AI and give me some examples."
        )
    else:
        prompt = "What are the main challenges to support a long context for LLM?"
    invoke_falconlite(prompt)

