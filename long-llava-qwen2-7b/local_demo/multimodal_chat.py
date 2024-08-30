import gradio as gr
import os
import json

import hashlib
import argparse
from PIL import Image

from loguru import logger

from theme_dropdown import create_theme_dropdown  # noqa: F401
from constants import (
    html_header,
)
dropdown, js = create_theme_dropdown()

from long_llava_backend import LongLlava

#longva = LongLlava(pretrained="aws-prototyping/long-llava-qwen2-7b", attn_implementation="flash_attention_2", load_in_4bit=True)
longva = LongLlava(pretrained="aws-prototyping/long-llava-qwen2-7b", load_in_4bit=True)


def generate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:6]


def print_like_dislike(x: gr.LikeData):
    logger.info(x.index, x.value, x.liked)


def add_message(history, message, video_input=None):
    #logger.info(f"{history=}, {video_input}")
    #logger.info(f"{message=}")
    if video_input is not None and video_input != "" and len(message["files"]) == 0:
        # speicial treatment for video input to enable multi-turn chat for the same video.
        if len(history) == 0:
            history.append(((video_input,), None))
        elif len(history)>0 and history[0][0][0] != video_input:
            history.append(((video_input,), None))
    else:
        for x in message["files"]:
            history.append(((x,), None))

    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def http_bot(
    video_input,
    state,
    sample_frames=16,
    temperature=0.2,
    max_new_tokens=8192,
    top_p=1.0,
    do_sample=False,
):
    try:
        visual_count = 0
        conv_count = 0
        prev_conv = []
        last_visual_index = 0
        for idx, x in enumerate(state):
            logger.info(f"{x=}")
            if type(x[0]) == tuple:
                visual_count += 1
                image_path = x[0][0]
                last_visual_index = idx
            elif type(x[0]) == str and type(x[1]) == str:
                conv_count += 1
                prev_conv.append(x)

        visuals = state[last_visual_index][0]
        if visual_count > 1:
            logger.info(f"Visual count: {visual_count}")
            logger.info(f"Resetting state to {last_visual_index}")
            state = state[last_visual_index:]
            prev_conv = []
        elif last_visual_index != 0:
            state = state[last_visual_index:]
            logger.info(f"Resetting state to {last_visual_index}")

        def get_task_type(visuals):
            if visuals[0].split(".")[-1] in ["mp4", "mov", "avi", "mp3", "wav", "mpga", "mpg", "mpeg"]:
                return "video"
            elif visuals[0].split(".")[-1] in ["png", "jpg", "jpeg", "webp", "bmp", "gif"]:
                video_input = None
                return "image"
            else:
                return "text"

        if visual_count == 0:
            image_path = ""
            task_type = "text"                    
        elif get_task_type(visuals) == "video":
            image_path = visuals[0]
            task_type = "video"
        elif get_task_type(visuals) == "image":
            task_type = "image"

        prompt = state[-1][0]

        if task_type != "text" and not os.path.exists(image_path):
            state[-1][1] = "The conversation is not correctly processed. Please try again."
            return state

        if task_type != "text":
            logger.info(f"Processing Visual: {image_path}")
            logger.info(f"Processing Question: {prompt}")
        else:
            logger.info(f"Processing Question (text): {prompt}")

        try:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
            }
            state[-1][1] = ""

            if task_type == "text":
                request = {
                    "prev_conv": prev_conv,
                    "visuals": [],
                    "context": prompt,
                    "task_type": task_type,
                }
                prev = 0
                for x in longva.stream_generate_until(request, gen_kwargs):
                    output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                    print(output[prev:], end="", flush=True)
                    state[-1][1] += output[prev:]
                    prev = len(output)
                    yield state

            elif image_path.split(".")[-1] in ["png", "jpg", "jpeg", "webp", "bmp", "gif"]:
                task_type = "image"
                # stream output
                image = Image.open(image_path).convert("RGB")
                request = {
                    "prev_conv": prev_conv,
                    "visuals": [image],
                    "context": prompt,
                    "task_type": task_type,
                }

                prev = 0
                for x in longva.stream_generate_until(request, gen_kwargs):
                    output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                    print(output[prev:], end="", flush=True)
                    state[-1][1] += output[prev:]
                    prev = len(output)
                    yield state

            elif image_path.split(".")[-1] in ["mp4", "mov", "avi", "mp3", "wav", "mpga", "mpg", "mpeg"]:
                task_type = "video"
                request = {
                    "prev_conv": prev_conv,
                    "visuals": [image_path],
                    "context": prompt,
                    "task_type": task_type,
                }
                gen_kwargs["sample_frames"] = sample_frames

                prev = 0
                for x in longva.stream_generate_until(request, gen_kwargs):
                    output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                    print(output[prev:], end="", flush=True)
                    state[-1][1] += output[prev:]
                    prev = len(output)
                    yield state

            else:
                state[-1][1] = "Image format is not supported. Please upload a valid image file."
                yield state
        except Exception as e:
            raise e

    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="LongLLava-7B", help="Model name")
    parser.add_argument("--temperature", default="0", help="Temperature")
    parser.add_argument("--max_new_tokens", default="8192", help="Max new tokens")
    args = parser.parse_args()

    PARENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
    LOGDIR = f"{PARENT_FOLDER}/logs"
    logger.info(PARENT_FOLDER)
    logger.info(LOGDIR)

    chatbot = gr.Chatbot(
        [],
        label=f"Model: {args.model_name}",
        elem_id="chatbot",
        bubble_full_width=False,
        height=700,
        avatar_images=(
            (
                os.path.join(
                    os.path.dirname(__file__), f"{PARENT_FOLDER}/assets/user_logo.png"
                )
            ),
            (
                os.path.join(
                    os.path.dirname(__file__),
                    f"{PARENT_FOLDER}/assets/assistant_logo.png",
                )
            ),
        ),
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", "video"],
        placeholder="Enter message or upload file...",
        show_label=False,
        max_lines=10000,
    )

    with gr.Blocks(
        theme="finlaymacklon/smooth_slate",
        title="Long Context Multimodal LLM Demo by AWS Prototyping Team",
        css=".message-wrap.svelte-1lcyrx4>div.svelte-1lcyrx4  img {min-width: 50px}",
    ) as demo:
        gr.HTML(html_header)

        models = ["LongLlava"]
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=models,
                    value=models[0] if len(models) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                )

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    sample_frames = gr.Slider(
                        minimum=0,
                        maximum=256,
                        value=16,
                        step=4,
                        interactive=True,
                        label="Sample Frames",
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=8192,
                        value=1024,
                        step=256,
                        interactive=True,
                        label="Max output tokens",
                    )
                    do_sample = gr.Checkbox(label="do_sample", info="Whether to sample during text generation.", interactive=True, value=False)

                video = gr.Video(label="Input Video", visible=False)
                gr.Examples(
                    examples=[
                        [
                            f"{PARENT_FOLDER}/assets/aws_bmw.mp4",
                            {
                                "text": "What is the video about? What cloud provider did BMW use?",
                            },
                        ],
                        [
                            f"{PARENT_FOLDER}/assets/amazon_prime_day.mp4",
                            {
                                "text": "When is the Amazon prime day? What is interesting about this commercial?",
                            },
                        ],
                        [
                            f"{PARENT_FOLDER}/assets/amazon_day_one.mp4",
                            {
                                "text": "What is the culture mentioned by Andy Jassy in the video?",
                            },
                        ],
                    ],
                    inputs=[video, chat_input],
                )
                gr.Examples(
                    examples=[
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/cute_dog.jpg",
                            ],
                            "text": "Why is the dog wearing a hat?",
                        },
                         {
                             "files": [
                                 f"{PARENT_FOLDER}/assets/aws_ec2_example.png",
                             ],
                             "text": "Can you generate the python boto3 code to create the infrastructure in the diagram?",
                         },
                         {
                             "files": [
                                 f"{PARENT_FOLDER}/assets/mona_lisa.jpg",
                             ],
                             "text": "What is the name of this painting? Why is it famous?",
                         },
                    ],
                    inputs=[chat_input],
                )
                with gr.Accordion("More Examples", open=False) as more_examples_row:
                    gr.Examples(
                        examples=[
                            {
                                "files": [
                                    f"{PARENT_FOLDER}/assets/cute_cat.jpg",
                                ],
                                "text": "Why is this cat cute?",
                            },
                            {
                                 "files": [
                                     f"{PARENT_FOLDER}/assets/starry_night.jpg",
                                 ],
                                 "text": "Who drew this? Why is it famous?",
                            },
                        ],
                        inputs=[chat_input],
                    )
            with gr.Column(scale=9):
                chatbot.render()
                chat_input.render()

                chat_msg = chat_input.submit(
                    add_message, [chatbot, chat_input, video], [chatbot, chat_input]
                )
                bot_msg = chat_msg.then(
                    http_bot,
                    inputs=[
                        video,
                        chatbot,
                        sample_frames,
                        temperature,
                        max_output_tokens,
                        top_p,
                        do_sample,
                    ],
                    outputs=[chatbot],
                    api_name="bot_response",
                )
                bot_msg.then(
                    lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
                )
                bot_msg.then(lambda: video, None, [video])

                chatbot.like(print_like_dislike, None, None)

                with gr.Row():
                    clear_btn = gr.ClearButton(chatbot, chat_input, chat_msg, bot_msg)
                    clear_btn.click(
                        lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
                    ).then(
                        lambda: gr.Video(value=None), None, [video]  # Set video to None
                    )
                    
                    submit_btn = gr.Button("Send", chat_msg)
                    submit_btn.click(
                        add_message, [chatbot, chat_input, video], [chatbot, chat_input]
                    ).then(
                        http_bot,
                        inputs=[
                            video,
                            chatbot,
                            sample_frames,
                            temperature,
                            max_output_tokens,
                            top_p,
                            do_sample,
                        ],
                        outputs=[chatbot],
                        api_name="bot_response",
                    ).then(
                        lambda: gr.MultimodalTextbox(interactive=True),
                        None,
                        [chat_input],
                    ).then(
                        lambda: gr.Video(value=None), None, [video]  # Set video to None
                    )

    demo.queue(max_size=128)
    demo.launch(max_threads=8, share=True, inline=False, server_name="0.0.0.0", server_port=6006, favicon_path=f"{PARENT_FOLDER}/assets/assistant_logo.png",)