import os
import re
import tempfile
import gradio as gr
import torch
import cv2
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoConfig, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the OmniLong-Qwen2.5-VL-7B model and processor
model_name = "aws-prototyping/OmniLong-Qwen2.5-VL-7B"

try:
    processor = AutoProcessor.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
        _attn_implementation="flash_attention_2",
    )
    tokenizer = processor.tokenizer
    print(f"Model loaded successfully! {model.config._attn_implementation=}")
except Exception as e:
    print(f"Error loading model: {e}")

# Define a directory where server-side videos are stored
SERVER_VIDEOS_DIR = "long-llava-qwen2-7b/local_demo/assets"  # Change this to your actual videos directory


def list_video_files(directory):
    """List all video files in the given directory and its subdirectories."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return []
    
    video_files = []
    
    # Walk through directory and all subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                file_path = os.path.join(root, filename)
                # Create a display path that's relative to the base directory
                rel_path = os.path.relpath(file_path, directory)
                # Use the relative path for display, but store the full path as the value
                video_files.append((rel_path, file_path))
    
    # Sort by the display name
    video_files.sort(key=lambda x: x[0])
    
    return video_files

def extract_frames(video_path, num_frames=8):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Ensure num_frames doesn't exceed total frames
    num_frames = min(num_frames, total_frames)
    
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    sample_fps = len(frames) / max(total_frames, 1e-6) * video_fps
    return frames, sample_fps

def get_video_title(video_path):
    """Extract a title from the video path."""
    # Get the filename without extension
    return os.path.basename(video_path).rsplit('.', 1)[0]

def process_video(video_path, num_frames, progress=gr.Progress()):
    """Process the selected video and return frames."""
    if not video_path:
        raise gr.Error("Please select a video file first.")
    
    progress(0.5, desc=f"Extracting {num_frames} frames...")
    frames, sample_fps = extract_frames(video_path, num_frames)
    video_title = get_video_title(video_path)
    
    progress(1.0, desc="Processing complete")
    return frames, sample_fps, video_title, video_path

def answer_question(frames, fps, video_title, question, temperature, top_p, do_sample):
    """Answer a question about the video using the LLaVA model."""
    if not frames:
        return "Please select a video file first."
    
    try:
        # Prepare the prompt
        # Apply chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "fps": fps,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        # Process the frames and question with the model
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Generate response with user-specified parameters
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=do_sample,
                temperature=temperature if do_sample else 0.0,
                top_p=top_p if do_sample else 1.0,
            )
            output = output[:, inputs["input_ids"].shape[-1] :]
            
        # Decode and return the response
        response = processor.decode(output[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        return f"Error generating response: {e}"

def display_frames(frames):
    """Convert frames to a gallery for display."""
    return frames if frames else None

def get_video_choices():
    """Get video files as choices for the dropdown."""
    video_files = list_video_files(SERVER_VIDEOS_DIR)
    return video_files

# Define a compact UI layout that fits without scrolling
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Video Question Answering with OmniLong-Qwen2.5-VL-7B")
    
    with gr.Row():
        # Left column - Video selection and controls
        with gr.Column(scale=1):
            video_choices = get_video_choices()
            video_dropdown = gr.Dropdown(
                choices=video_choices,
                label="Select Video",
                type="value"
            )
            
            with gr.Row():
                num_frames_slider = gr.Slider(
                    minimum=8, 
                    maximum=1024, 
                    value=768, 
                    step=8, 
                    label="Frames"
                )
                process_button = gr.Button("Process Video", scale=1)
            
            video_title = gr.Textbox(label="Video Title", visible=True, interactive=False)
            
            # Question and answer section
            with gr.Accordion("Question & Settings", open=True):
                question_input = gr.Textbox(
                    label="Ask a question", 
                    placeholder="What is happening in this video?",
                    lines=2
                )
                
                with gr.Row():
                    do_sample = gr.Checkbox(label="Do Sample", value=False)
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P")
                
                answer_button = gr.Button("Answer Question", variant="primary")
            
            # Answer output
            answer_output = gr.Textbox(label="Answer", interactive=False, lines=6)
        
        # Right column - Video display
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Video"):
                    video_player = gr.Video(label="Selected Video")
                with gr.TabItem("Frames"):
                    frames_gallery = gr.Gallery(label="Extracted Frames", columns=3)
            
    # State variables
    frames_output = gr.State([])
    sample_fps_out = gr.State(None)
    video_path_state = gr.State(None)
    
    # Connect the components
    process_button.click(
        process_video, 
        inputs=[video_dropdown, num_frames_slider], 
        outputs=[frames_output, sample_fps_out, video_title, video_path_state]
    ).then(
        display_frames,
        inputs=[frames_output],
        outputs=[frames_gallery]
    ).then(
        lambda x: x,
        inputs=[video_dropdown],
        outputs=[video_player]
    )
    
    answer_button.click(
        answer_question,
        inputs=[frames_output, sample_fps_out, video_title, question_input, temperature, top_p, do_sample],
        outputs=[answer_output]
    )

demo.launch(share=True)