
import os
import re
import tempfile
import uuid
import shutil
import json
import requests
import threading
import time
import queue
from typing import List, Tuple, Optional
import torch
import numpy as np
import gradio as gr

from num2words import num2words
from pydub import AudioSegment
import torchaudio
from speechbrain.pretrained import HIFIGAN, Tacotron2
import whisper

from TTS.api import TTS

# --- Initial Setup & Configuration ---
os.environ["COQUI_TOS_AGREED"] = "1"

# Constants
SPEECH_TTS_PREFIX = "speech-tts-"
WAV_SUFFIX = ".wav"
MP3_SUFFIX = ".mp3"
VOICE_SAMPLES_DIR = "voice_samples"
STANDARD_VOICE_NAME = "Standard Voice (Non-Cloned)"
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = "http://ollama:11434"  # Default Ollama URL
DEFAULT_MODEL = "qwen:1.8b"  # Default model, can be changed

# Realtime transcription state
realtime_transcription_active = False
realtime_thread = None
audio_queue = queue.Queue()
transcription_results = []

# --- Model Loading ---
print("[Gradio App] Initializing models...")

# Transcription model
try:
    # Check for GPU availability for Whisper
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("base", device=whisper_device)
    print(f"[Gradio App] Whisper model loaded successfully on {whisper_device}.")
except Exception as e:
    print(f"[Gradio App] Error loading Whisper model: {e}")
    whisper_model = None

# Voice Cloning model (Coqui XTTS)
voice_model = None
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voice_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"[Gradio App] Coqui XTTS voice model loaded successfully on {device}.")
except Exception as e:
    print(f"[Gradio App] Error initializing Coqui TTS model: {str(e)}")

# Standard TTS models (SpeechBrain) - loaded on demand
tacotron2 = None
hifi_gan = None

print("[Gradio App] Model initialization complete.")

# --- Core Logic Functions ---

def preprocess_text(text: str) -> str:
    """Converts numbers in the text to words."""
    return re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)

def run_voice_clone_tts_and_save_file(text: str, speaker_id: str) -> str:
    """Generates speech using a cloned voice."""
    if voice_model is None:
        raise gr.Error("Voice cloning model (Coqui TTS) is not available. Please check logs.")

    reference_audio = os.path.join(VOICE_SAMPLES_DIR, speaker_id, "reference.wav")
    if not os.path.exists(reference_audio):
        raise gr.Error(f"Speaker ID '{speaker_id}' not found. Please train the voice first.")

    tmp_path_wav = os.path.join(tempfile.gettempdir(), SPEECH_TTS_PREFIX + str(uuid.uuid4()) + WAV_SUFFIX)

    try:
        voice_model.tts_to_file(
            text=text,
            file_path=tmp_path_wav,
            speaker_wav=reference_audio,
            language="en"
        )
        return tmp_path_wav
    except Exception as e:
        raise gr.Error(f"An error occurred during voice cloning: {str(e)}")

def run_standard_tts_and_save_file(text: str) -> str:
    """Generates speech using the standard, non-cloned voice."""
    global tacotron2, hifi_gan
    if tacotron2 is None or hifi_gan is None:
        print("[Gradio App] Initializing standard TTS models (SpeechBrain)...")
        try:
            tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
            hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
            print("[Gradio App] Standard TTS models loaded.")
        except Exception as e:
            raise gr.Error(f"Could not load standard TTS models: {e}")

    mel_outputs, _, _ = tacotron2.encode_batch([text])
    waveforms = hifi_gan.decode_batch(mel_outputs)

    tmp_path_wav = os.path.join(tempfile.gettempdir(), SPEECH_TTS_PREFIX + str(uuid.uuid4()) + WAV_SUFFIX)
    torchaudio.save(tmp_path_wav, waveforms.squeeze(1), 44100) # Use 22050 sample rate for consistency
    return tmp_path_wav

# --- Ollama Integration Functions ---

def get_ollama_models() -> List[str]:
    """Fetch available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json()
        return [model["name"] for model in models.get("models", [])]
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] Error connecting to Ollama: {e}")
        return [DEFAULT_MODEL]

def chat_with_ollama(message: str, model: str = DEFAULT_MODEL, conversation_history: List = None) -> Tuple[str, List]:
    """Send a message to Ollama and get a response."""
    if conversation_history is None:
        conversation_history = []

    history_copy = conversation_history.copy()
    history_copy.append({"role": "user", "content": message})

    try:
        payload = {
            "model": model,
            "messages": history_copy,
            "stream": False
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        response.raise_for_status()

        response_data = response.json()
        assistant_message = response_data.get("message", {}).get("content", "")
        print(f"[Ollama Response] {assistant_message}")

        history_copy.append({"role": "assistant", "content": assistant_message})
        return assistant_message, history_copy

    except requests.exceptions.RequestException as e:
        error_msg = f"Error communicating with Ollama: {str(e)}"
        print(f"[Ollama] {error_msg}")
        return error_msg, history_copy

# --- Realtime Transcription Functions (Corrected) ---

def realtime_transcription_worker():
    """Improved worker function for realtime transcription."""
    global realtime_transcription_active, audio_queue, transcription_results

    buffer = []
    buffer_duration = 0
    max_buffer_duration = 3.0  # Process every 3 seconds

    while realtime_transcription_active:
        try:
            # Process all available audio in the queue
            while not audio_queue.empty():
                audio_data = audio_queue.get_nowait()
                if audio_data is not None:
                    sample_rate, audio_array = audio_data

                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)

                    audio_array = audio_array.astype(np.float32)
                    if np.max(np.abs(audio_array)) > 1.0:
                        audio_array = audio_array / np.max(np.abs(audio_array))

                    buffer.append(audio_array)
                    buffer_duration += len(audio_array) / sample_rate
                audio_queue.task_done()

            # Process buffer when it reaches the duration threshold
            if buffer_duration >= max_buffer_duration and buffer:
                combined_audio = np.concatenate(buffer)
                transcription = transcribe_audio_chunk(combined_audio)

                if transcription and transcription.strip():
                    print(f"[Realtime] Transcribed: {transcription}")
                    transcription_results.append(transcription)

                    if len(transcription_results) > 10:
                        transcription_results.pop(0)

                buffer = []
                buffer_duration = 0

            time.sleep(0.1)

        except Exception as e:
            print(f"[Realtime Transcription] Worker Error: {e}")
            time.sleep(0.1)

    print("[Realtime Transcription] Worker stopped")

def start_realtime_transcription():
    """Start realtime transcription."""
    global realtime_transcription_active, realtime_thread, transcription_results

    if realtime_transcription_active:
        return "Realtime transcription is already running."

    if whisper_model is None:
        return "Whisper model is not available. Cannot start realtime transcription."

    transcription_results.clear()
    realtime_transcription_active = True
    realtime_thread = threading.Thread(target=realtime_transcription_worker)
    realtime_thread.daemon = True
    realtime_thread.start()

    return "Realtime transcription started. Speak into your microphone."

def stop_realtime_transcription():
    """Stop realtime transcription."""
    global realtime_transcription_active

    realtime_transcription_active = False
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    return "Realtime transcription stopped."

def get_realtime_transcription_results():
    """Get the latest transcription results."""
    if not transcription_results:
        return "No transcription results yet. Make sure realtime transcription is running and you're speaking..."
    return "\n".join(transcription_results[-5:])

def add_audio_to_queue(audio_data):
    """Add audio data to the processing queue."""
    if audio_data is not None and realtime_transcription_active:
        try:
            audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass
    return # Explicitly return None

def transcribe_audio_chunk(audio_data: np.ndarray) -> str:
    """Transcribe a chunk of audio data using Whisper."""
    if whisper_model is None:
        return "Whisper model not available"
    try:
        # Whisper can process the numpy array directly
        result = whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available())
        return result["text"].strip()
    except Exception as e:
        print(f"Transcription chunk error: {str(e)}")
        return ""

# --- Gradio Interface Functions ---

def update_speaker_list() -> gr.Dropdown:
    """Update the speaker dropdown list."""
    voices = [STANDARD_VOICE_NAME]
    if os.path.exists(VOICE_SAMPLES_DIR):
        for speaker_id in os.listdir(VOICE_SAMPLES_DIR):
            path = os.path.join(VOICE_SAMPLES_DIR, speaker_id)
            if os.path.isdir(path):
                voices.append(speaker_id)
    return gr.Dropdown(choices=voices, value=STANDARD_VOICE_NAME)

def update_model_list() -> gr.Dropdown:
    """Get available Ollama models for the dropdown."""
    models = get_ollama_models()
    return gr.Dropdown(choices=models, value=models[0] if models else DEFAULT_MODEL)

def gradio_train_voice(speaker_name: str, audio_file_path: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """Process an uploaded audio file to train a new voice clone."""
    if not speaker_name or not audio_file_path:
        raise gr.Error("Speaker name and audio file are required.")

    sane_name = re.sub(r'[^a-zA-Z0-9_-]', '-', speaker_name.lower())
    speaker_id = f"{sane_name}-{uuid.uuid4().hex[:6]}"
    speaker_dir = os.path.join(VOICE_SAMPLES_DIR, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    reference_path = os.path.join(speaker_dir, "reference.wav")

    try:
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(reference_path, format="wav")

        status_message = f"✅ Voice trained successfully!\nSpeaker ID: **{speaker_id}**"
        print(f"Voice trained. Speaker ID: {speaker_id}")

        updated_dropdown = update_speaker_list()
        return status_message, updated_dropdown, updated_dropdown
    except Exception as e:
        shutil.rmtree(speaker_dir, ignore_errors=True)
        raise gr.Error(f"Voice processing failed: {str(e)}")

def gradio_generate_tts(text: str, speaker_id: str) -> str:
    """Main TTS function called by the Gradio interface."""
    if not text:
        raise gr.Error("Text input cannot be empty.")

    clean_text = preprocess_text(text.strip())
    output_files = []

    try:
        if speaker_id == STANDARD_VOICE_NAME:
            # Split text into sentences for better synthesis with the standard voice
            sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', clean_text) if s.strip()]
            if not sentences:
                sentences = [clean_text]
            for sentence in sentences:
                output_files.append(run_standard_tts_and_save_file(sentence))
        else:
            output_files.append(run_voice_clone_tts_and_save_file(clean_text, speaker_id))

        if not output_files:
            raise gr.Error("TTS generation failed to produce any audio.")

        combined_audio = AudioSegment.empty()
        for f in output_files:
            combined_audio += AudioSegment.from_wav(f)

        tmp_path_mp3 = os.path.join(tempfile.gettempdir(), SPEECH_TTS_PREFIX + str(uuid.uuid4()) + MP3_SUFFIX)
        combined_audio.export(tmp_path_mp3, format="mp3")
        return tmp_path_mp3
    finally:
        for f in output_files:
            if os.path.exists(f):
                os.remove(f)

def gradio_transcribe(audio_filepath: str) -> Tuple[str, str]:
    """Transcribe uploaded audio to text using Whisper."""
    if not audio_filepath:
        raise gr.Error("Please upload an audio file to transcribe.")
    if whisper_model is None:
        raise gr.Error("Transcription model (Whisper) is not available. Please check logs.")

    try:
        audio = whisper.load_audio(audio_filepath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        result = whisper_model.transcribe(audio_filepath, language=detected_language)
        return detected_language, result["text"].strip()
    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")

def gradio_text_chat(message: str, model: str, speaker_voice: str, conversation_history: List) -> Tuple[str, str, List, str]:
    """Text-based chat with Ollama, including TTS generation."""
    if not message.strip():
        # Return gracefully instead of raising an error for empty input
        return "", None, conversation_history, ""

    ollama_response, updated_history = chat_with_ollama(message.strip(), model, conversation_history)
    if not ollama_response or "error" in ollama_response.lower():
        gr.Warning(f"Ollama issue: {ollama_response}")
        return ollama_response, None, updated_history, ""

    try:
        audio_response = gradio_generate_tts(ollama_response, speaker_voice)
    except Exception as e:
        gr.Warning(f"TTS generation failed: {str(e)}")
        audio_response = None # Allow chat to continue without audio

    return ollama_response, audio_response, updated_history, ""

def format_conversation_history(history: List) -> str:
    """Format conversation history for display."""
    if not history:
        return "No conversation yet."
    formatted = []
    for msg in history:
        role = "🗣️ You" if msg["role"] == "user" else "🤖 Assistant"
        formatted.append(f"**{role}:** {msg['content']}")
    return "\n\n".join(formatted)

def clear_conversation() -> Tuple[List, str]:
    """Clear the conversation history."""
    return [], "Conversation cleared."

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="Speech API with Ollama Chat") as demo:
    gr.Markdown("# 🗣️ Speech Synthesis, Recognition & AI Chat")
    gr.Markdown("A comprehensive interface for Text-to-Speech, Speech-to-Text, and chat with Ollama.")

    text_chat_state = gr.State([])

    with gr.Tabs():
        # --- Text Chat Tab ---
        with gr.TabItem("💬 Text Chat with Ollama"):
            gr.Markdown("## Chat with an AI using Text")
            gr.Markdown("Have a text-based conversation with an Ollama model. AI responses will be spoken using your selected voice.")

            with gr.Row():
                with gr.Column(scale=1):
                    text_chat_model_dropdown = gr.Dropdown(
                        label="Select Ollama Model",
                        choices=get_ollama_models(),
                        value=get_ollama_models()[0] if get_ollama_models() else DEFAULT_MODEL,
                        interactive=True
                    )
                    text_chat_speaker_dropdown = gr.Dropdown(
                        label="Select AI Voice",
                        choices=update_speaker_list().choices,
                        value=STANDARD_VOICE_NAME,
                        interactive=True
                    )
                    text_chat_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here and press Enter...",
                        lines=3,
                        interactive=True
                    )
                    with gr.Row():
                        text_chat_button = gr.Button("Send", variant="primary", scale=3)
                        text_chat_clear_button = gr.Button("Clear Chat", variant="secondary", scale=1)

                with gr.Column(scale=2):
                    text_chat_response = gr.Textbox(label="AI Response", lines=6, interactive=False)
                    text_chat_audio_output = gr.Audio(label="AI Voice Response", type="filepath", autoplay=True)

            text_chat_conversation_display = gr.Markdown(label="Conversation History", value="No conversation yet.")

        # --- Realtime Transcription Tab (Corrected) ---
        with gr.TabItem("📡 Realtime Transcription"):
            gr.Markdown("## Live Speech-to-Text")
            gr.Markdown("Click 'Start Realtime' and speak into your microphone. Click 'Refresh' to see the transcribed text.")

            with gr.Row():
                with gr.Column(scale=1):
                    realtime_audio_input = gr.Audio(
                        label="Live Audio Input",
                        sources=["microphone"],
                        type="numpy",
                        streaming=True
                    )
                    with gr.Row():
                        start_realtime_button = gr.Button("🎙️ Start Realtime", variant="primary")
                        stop_realtime_button = gr.Button("⏹️ Stop", variant="secondary")
                        refresh_results_button = gr.Button("🔄 Refresh Results", variant="secondary")

                    realtime_status = gr.Textbox(
                        label="Status",
                        value="Ready to start realtime transcription.",
                        interactive=False
                    )

                with gr.Column(scale=2):
                    realtime_transcription_output = gr.Textbox(
                        label="Live Transcription Results",
                        lines=10,
                        interactive=False,
                        placeholder="Transcribed text will appear here..."
                    )

        # --- TTS Tab ---
        with gr.TabItem("🔊 Text-to-Speech (TTS)"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    tts_text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter your text here...", lines=5)
                    tts_speaker_dropdown = gr.Dropdown(
                        label="Select Speaker",
                        choices=update_speaker_list().choices,
                        value=STANDARD_VOICE_NAME,
                        interactive=True
                    )
                    tts_generate_button = gr.Button("Generate Speech", variant="primary")
                with gr.Column(scale=1):
                    tts_audio_output = gr.Audio(label="Generated Audio", type="filepath")

        # --- Voice Training Tab ---
        with gr.TabItem("🎙️ Voice Training"):
            gr.Markdown("## Train a New Voice for Cloning")
            gr.Markdown("Upload a clear audio sample (10-30 seconds is ideal) of a single speaker. A unique Speaker ID will be generated for use in other tabs.")
            with gr.Row():
                train_speaker_name = gr.Textbox(label="Enter Speaker Name")
                train_audio_input = gr.Audio(label="Upload Reference Audio", sources=["upload"], type="filepath")
            train_button = gr.Button("Train Voice", variant="primary")
            train_status_output = gr.Markdown(label="Training Status")

        # --- Transcription Tab ---
        with gr.TabItem("✍️ Transcription (File)"):
            gr.Markdown("## Transcribe an Audio File to Text")
            gr.Markdown("Upload an audio file and the model will transcribe it into text.")
            transcribe_audio_input = gr.Audio(label="Upload Audio File", sources=["upload"], type="filepath")
            transcribe_button = gr.Button("Transcribe Audio", variant="primary")
            with gr.Row():
                transcribe_lang_output = gr.Textbox(label="Detected Language", interactive=False)
                transcribe_text_output = gr.Textbox(label="Transcription Result", lines=8, interactive=False)

    # --- Event Handlers ---
    def chat_and_update(message, model, speaker, history):
        response, audio, updated_history, new_input = gradio_text_chat(message, model, speaker, history)
        return response, audio, updated_history, new_input, format_conversation_history(updated_history)

    text_chat_button.click(
        fn=chat_and_update,
        inputs=[text_chat_input, text_chat_model_dropdown, text_chat_speaker_dropdown, text_chat_state],
        outputs=[text_chat_response, text_chat_audio_output, text_chat_state, text_chat_input, text_chat_conversation_display],
        api_name="text_chat"
    )
    text_chat_input.submit(
        fn=chat_and_update,
        inputs=[text_chat_input, text_chat_model_dropdown, text_chat_speaker_dropdown, text_chat_state],
        outputs=[text_chat_response, text_chat_audio_output, text_chat_state, text_chat_input, text_chat_conversation_display],
        api_name="text_chat_submit"
    )
    text_chat_clear_button.click(
        fn=clear_conversation,
        outputs=[text_chat_state, text_chat_conversation_display]
    )

    # Realtime Transcription Tab (Corrected)
    start_realtime_button.click(fn=start_realtime_transcription, outputs=[realtime_status])
    stop_realtime_button.click(fn=stop_realtime_transcription, outputs=[realtime_status])
    refresh_results_button.click(fn=get_realtime_transcription_results, outputs=[realtime_transcription_output])
    realtime_audio_input.stream(fn=add_audio_to_queue, inputs=[realtime_audio_input])

    # TTS Tab
    tts_generate_button.click(
        fn=gradio_generate_tts,
        inputs=[tts_text_input, tts_speaker_dropdown],
        outputs=[tts_audio_output],
        api_name="tts"
    )

    # Voice Training Tab
    train_button.click(
        fn=gradio_train_voice,
        inputs=[train_speaker_name, train_audio_input],
        outputs=[train_status_output, tts_speaker_dropdown, text_chat_speaker_dropdown]
    )

    # Transcription Tab
    transcribe_button.click(
        fn=gradio_transcribe,
        inputs=[transcribe_audio_input],
        outputs=[transcribe_lang_output, transcribe_text_output],
        api_name="transcribe"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)
