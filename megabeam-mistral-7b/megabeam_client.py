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