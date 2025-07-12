# video_app.py

import os, re, tempfile, uuid, shutil, json, requests, threading, time, queue
from typing import List, Tuple
import torch, numpy as np, gradio as gr

from num2words import num2words
from pydub import AudioSegment
import torchaudio
from speechbrain.pretrained import HIFIGAN, Tacotron2
import whisper
from TTS.api import TTS
from moviepy.editor import AudioFileClip, TextClip, CompositeVideoClip, ColorClip

# ─── Setup ───────────────────────────────────────────────────────────────────────
os.environ["COQUI_TOS_AGREED"] = "1"
VOICE_SAMPLES_DIR = "voice_samples"
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
STANDARD_VOICE_NAME = "Standard Voice (Non‑Cloned)"
OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_MODEL = "qwen:1.8b"

realtime_transcription_active = False
audio_queue = queue.Queue()
transcription_results = []

# Load models (Whisper, TTS, SpeechBrain)
...
# (same model init as before)  
# ────────────────────────────────────────────────────────────────────────────────

# ─── All your existing functions: preprocess_text, run_voice_clone_tts_and_save_file, etc. ───
# (Copy/paste all your Chat + TTS + Transcription + Gradio functions here)
# ────────────────────────────────────────────────────────────────────────────────

# ─── 🎞️ Video generation function with MoviePy ─────────────────────────────────────

def create_video_from_audio_and_text(audio_path: str, text: str) -> str:
    """
    Creates a simple video with background + subtitles, overlaying the TTS audio.
    """
    audio_clip = AudioFileClip(audio_path)
    w, h = 720, 480
    duration = audio_clip.duration

    # solid background
    bg = ColorClip(size=(w, h), color=(30, 30, 30)).set_duration(duration)

    # subtitle text clip
    subtitle = (TextClip(
        text, fontsize=30, color='white', align='center', size=(w - 40, None), method='caption')
        .set_duration(duration)
        .set_position(('center', h - 80))
    )

    video = CompositeVideoClip([bg, subtitle]).set_audio(audio_clip)
    output_path = os.path.join(tempfile.gettempdir(), f"ollama_video_{uuid.uuid4().hex}.mp4")
    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', verbose=False, logger=None)
    return output_path

# ─── Gradio + Flask hybrid ────────────────────────────────────────────────────────
from flask import Flask, request, send_file, jsonify
app = Flask(__name__)

@app.route("/generate-video", methods=["POST"])
def api_generate_video():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    speaker_id = data.get("speaker_id", STANDARD_VOICE_NAME)
    model = data.get("model", DEFAULT_MODEL)

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        response_text, _ = chat_with_ollama(prompt, model)
        audio_path = gradio_generate_tts(response_text, speaker_id)
        video_path = create_video_from_audio_and_text(audio_path, response_text)
        return send_file(video_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Gradio App ────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="Speech & Video API with Ollama Chat") as demo:
    gr.Markdown("# 🗣️ Speech & 🎥 Video with Ollama")
    text_chat_state = gr.State([])

    with gr.Tabs():
        # Text‑only chat tab (reuse existing UI)
        with gr.TabItem("💬 Chat"):
            gr.Markdown("Chat & TTS")
            # (Copy existing text chat UI setup here)
            ...
        # Other tabs...
        # Optionally add a “Generate Video” UI inside Gradio:
        with gr.TabItem("🎥 Generate Video"):
            prompt_in = gr.Textbox(label="Enter prompt", lines=2)
            speaker_sel = gr.Dropdown(label="Speaker", choices=update_speaker_list().choices, value=STANDARD_VOICE_NAME)
            model_sel = gr.Dropdown(label="Model", choices=get_ollama_models(), value=DEFAULT_MODEL)
            generate_btn = gr.Button("Generate Video")
            video_out = gr.Video(label="Generated Video")
            def gradio_video_func(prompt, speaker, model):
                txt, _ = chat_with_ollama(prompt, model)
                mp3 = gradio_generate_tts(txt, speaker)
                return create_video_from_audio_and_text(mp3, txt)
            generate_btn.click(gradio_video_func, inputs=[prompt_in, speaker_sel, model_sel], outputs=[video_out])

    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)

if __name__ == "__main__":
    # Run both Flask (for API) and Gradio (same process)
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=7860), daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)
