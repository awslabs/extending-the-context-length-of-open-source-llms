import argparse
import copy
import math
import warnings
import json
import os

from datetime import timedelta
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from packaging import version
from transformers import AutoConfig, BitsAndBytesConfig

from transformers import TextIteratorStreamer
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from threading import Thread

import requests

#torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"


if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


class LongLlava:
    """
    LongLlava Model
    """

    def __init__(
        self,
        pretrained: str = "amazon/long-llava-qwen2-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "auto",
        load_in_4bit: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        truncate_context: Optional[
            bool
        ] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        video_decode_backend: str = "decord",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.max_frames_num = max_frames_num
        self.video_decode_backend = video_decode_backend

        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit)
        else:
            quantization_config = None

        self._model = LlavaForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map=device_map, attn_implementation=attn_implementation, quantization_config=quantization_config)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._processor = AutoProcessor.from_pretrained(pretrained)
        self._image_processor = self._processor.image_processor
        self._max_length = 224000 # to be parameterized

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert (
            self.batch_size_per_gpu == 1
        ), "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_images(self, images, image_processor):
        new_images = []
        for image in images:
            image = self.expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            new_images.append(image)
        return new_images

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        print(f"spare_frames: {spare_frames.shape}")
        return spare_frames  # (frames, height, width, channels)

    def stream_generate_until(self, requests: dict, gen_kwargs: dict) -> List[str]:

        question_input = []

        visuals = requests["visuals"]
        context = requests["context"]
        task_type = requests["task_type"]
        
        print(f"################# requests ######################\n{requests}")
        print(f"################# gen_kwargs ######################\n{gen_kwargs}")

        if task_type == "text":
            image_tensor = None

        # encode, pad, and truncate contexts for this batch
        elif task_type == "image":  # For image task
            image_tensor = self.process_images(visuals, self._image_processor)
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": context},
                        {"type": "image"},
                    ],
                },
            ]

        elif task_type == "video":  # For video task
            max_frames = gen_kwargs.get("sample_frames", self.max_frames_num)
            if "sample_frames" in gen_kwargs:
                gen_kwargs.pop("sample_frames")

            try:
                if self.video_decode_backend == "decord":
                    frames = self.load_video(visuals, max_frames)
                images = [Image.fromarray(frame).convert("RGB") for frame in frames]
                image_tensor = self.process_images(images, self._image_processor)
            except Exception as e:
                eval_logger.error(f"Error {e} in loading video")
                image_tensor = None

            conversation = [
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": context}] + [{"type": "image"}] * len(frames),
                },
            ]
        
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(images=image_tensor, text=prompt, return_tensors='pt').to("cuda", torch.bfloat16)


        # preconfigure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        stop_str = "<|im_end|>"
        

        max_context_length = getattr(self.model.config, "max_position_embeddings", 2048)
        num_image_tokens = (
            prompt.count(DEFAULT_IMAGE_TOKEN)
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
        )

        try:
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    use_cache=self.use_cache,
                    streamer=streamer,
                    **inputs,
                    **gen_kwargs,
                )
            )
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                if generated_text.endswith(stop_str):
                    generated_text = generated_text[: -len(stop_str)]
                yield json.dumps(
                    {"text": generated_text, "error_code": 0}
                ).encode() + b"\0"
        except Exception as e:
            raise e