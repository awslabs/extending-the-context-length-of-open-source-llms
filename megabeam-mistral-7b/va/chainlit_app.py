import io
from openai import OpenAI
from chainlit.element import ElementBased
import chainlit as cl
import logging
from transformers import pipeline
import numpy as np
from pydub import AudioSegment
from melo.api import TTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ASR pipeline
logger.info("Initializing ASR pipeline...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
logger.info("ASR pipeline initialized successfully")

logger.info("Initializing TTS model...")
tts_model = TTS(language='EN', device='cpu')  # or 'cuda' if GPU is needed
speaker_ids = tts_model.hps.data.spk2id
logger.info("TTS model initialized successfully")

# Initialize OpenAI client for local LLM
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
logger.info("OpenAI client initialized")
llm_model = client.models.list().data[0].id

#@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    """Generate speech from text using local TTS model directly in memory"""
    try:
        logger.info("Generating speech with TTS model")
        bio = io.BytesIO()
        tts_model.tts_to_file(
            text=text,
            speaker_id=speaker_ids['EN-AU'],
            output_path=bio,
            speed=1.0,
            quiet=True,
            format='wav'
        )
    
        logger.info("TTS generation completed successfully")
        return "output.wav", bio.getvalue()
        
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise

async def process_audio_with_whisper(audio_file, mime_type):
    """Process audio using local Whisper model"""
    logger.info("Processing audio with local Whisper model")
    try:
        # Convert webm to wav using pydub
        audio_bytes = io.BytesIO(audio_file)
        
        # Load audio with pydub
        logger.info("Converting audio format")
        audio = AudioSegment.from_file(audio_bytes, format=mime_type.split('/')[-1])
        
        # Get audio array directly from AudioSegment
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Get sample rate directly from AudioSegment
        sample_rate = audio.frame_rate
        
        logger.info(f"Audio array shape: {audio_array.shape}")
        logger.info(f"Sample rate: {sample_rate}")
        
        # Transcribe with Whisper
        logger.info("Starting transcription with Whisper")
        result = transcriber({"sampling_rate": sample_rate, "raw": audio_array})
        logger.info(f"Whisper transcription result: {result}")
        
        return result["text"]
            
    except Exception as e:
        logger.error(f"Error in Whisper processing: {str(e)}")
        raise

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome! Press `P` to talk!").send()
    cl.user_session.set("message_history", [
        {
         "role": "user", 
         "content": "Please respond conversationally, as if we're having a natural back-and-forth dialogue. "
         "Limit your response to one paragraph, and keep it casual yet engaging, like you would in a real conversation."
         "Do not add meta-commentary about how you will respond or notes about maintaining a conversational style."
         },
        {"role": "assistant", 
         "content": "Hi there! I'm ready to chat with you in a concise, natural, conversational way. What's on your mind?"}
    ])

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    logger.info(f"Received audio chunk. Is start: {chunk.isStart}")
    if chunk.isStart:
        buffer = io.BytesIO()
        buffer.name = "You said:"#f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
    
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    logger.info("Audio recording ended")
    # Get the audio buffer
    audio_buffer = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_file = audio_buffer.read()
    audio_mime_type = cl.user_session.get("audio_mime_type")
    
    logger.info(f"Audio MIME type: {audio_mime_type}")

    # Show the audio in the UI
    input_audio_el = cl.Audio(
        mime=audio_mime_type, 
        content=audio_file, 
        name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        content="",
        elements=[input_audio_el]
    ).send()

    try:
        # Use our local Whisper model
        transcription = await process_audio_with_whisper(audio_file, audio_mime_type)
        logger.info(f"Transcription from Whisper: {transcription}")
        await cl.Message(content=f"I heard: {transcription}").send()

        # Get message history
        message_history = cl.user_session.get("message_history")
        message_history.append({"role": "user", "content": transcription + "\n(Please limit you answer within a single paragraph.)"})

        # Create streaming message
        msg = cl.Message(content="")
        await msg.send()

        # Query LLM with streaming
        stream = client.chat.completions.create(
            model=llm_model,
            messages=message_history,
            temperature=0.7,
            stream=True,
            stop=["(Note: ", "[", "*Note"]
        )

        # Stream the response
        complete_response = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    complete_response += content
                    await msg.stream_token(content)
                    
        # Generate audio response
        output_name, output_audio = await text_to_speech(complete_response, audio_mime_type)
        
        output_audio_el = cl.Audio(
            name="Assistant said:",
            auto_play=True,
            mime=audio_mime_type,
            content=output_audio,
        )

        msg.content = complete_response  # This sets the text part of the message
        msg.elements = [output_audio_el]  # This adds the audio player to the message
        # Update message and history
        await msg.update()
        message_history.append({"role": "assistant", "content": complete_response})
        cl.user_session.set("message_history", message_history)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        await cl.Message(content=f"Error: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Handle text messages
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    try:
        # Create streaming message
        msg = cl.Message(content="")
        await msg.send()

        # Query LLM with streaming
        stream = client.chat.completions.create(
            model=llm_model,
            messages=message_history,
            temperature=0.7,
            stream=True
        )

        # Stream the response
        complete_response = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    complete_response += content
                    await msg.stream_token(content)

        # Update message and history
        await msg.update()
        message_history.append({"role": "assistant", "content": complete_response})
        cl.user_session.set("message_history", message_history)

    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        await cl.Message(content=f"Error: {str(e)}").send()

if __name__ == "__main__":
    logger.info("Starting application...")
    cl.run()