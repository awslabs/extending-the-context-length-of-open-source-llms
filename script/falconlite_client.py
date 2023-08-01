from text_generation import Client

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
                    print(snippet, end='')
    return output

if __name__ == "__main__":
    """
    Replace `prompt` with your long context here!
    """
    prompt = "What are the main challenges to support a long context for LLM?"
    invoke_falconlite(prompt)

