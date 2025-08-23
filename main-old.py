import os
import re
import uuid
import tempfile
import requests
from datetime import datetime
from typing import List, Tuple, Dict
from num2words import num2words
import gradio as gr
import json

# Audio
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
from TTS.api import TTS

# --- Constants ---
OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_MODEL = "qwen:1.8b"
VOICE_SAMPLES_DIR = "voice_samples"
GENERATED_AUDIO_DIR = "generated_audio"
STANDARD_VOICE_NAME = "standard"
WAV_SUFFIX = ".wav"
VOICE_CONFIG_FILE = "voice_config.json"

# Ensure audio output directory exists
os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

# --- Voice Management ---
class VoiceManager:
    def __init__(self):
        self.voice_config_path = VOICE_CONFIG_FILE
        self.voices = self.load_voice_config()
    
    def load_voice_config(self) -> Dict:
        """Load voice configuration from JSON file"""
        if os.path.exists(self.voice_config_path):
            try:
                with open(self.voice_config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Error loading voice config: {e}")
        
        # Default configuration
        default_config = {
            "standard": {
                "name": "Standard Voice",
                "description": "Default synthetic voice",
                "type": "standard",
                "enabled": True
            }
        }
        self.save_voice_config(default_config)
        return default_config
    
    def save_voice_config(self, config: Dict):
        """Save voice configuration to JSON file"""
        try:
            with open(self.voice_config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Error saving voice config: {e}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice IDs"""
        voices = []
        
        # Add standard voice
        if self.voices.get("standard", {}).get("enabled", True):
            voices.append("standard")
        
        # Scan for cloned voices
        if os.path.exists(VOICE_SAMPLES_DIR):
            for voice_dir in os.listdir(VOICE_SAMPLES_DIR):
                voice_path = os.path.join(VOICE_SAMPLES_DIR, voice_dir)
                reference_file = os.path.join(voice_path, "reference.wav")
                
                if os.path.isdir(voice_path) and os.path.exists(reference_file):
                    voice_info = self.voices.get(voice_dir, {})
                    if voice_info.get("enabled", True):
                        voices.append(voice_dir)
        
        return voices if voices else ["standard"]
    
    def get_voice_display_names(self) -> List[str]:
        """Get list of voice display names"""
        voices = self.get_available_voices()
        display_names = []
        
        for voice_id in voices:
            if voice_id == "standard":
                display_names.append("Standard Voice")
            else:
                voice_info = self.voices.get(voice_id, {})
                name = voice_info.get("name", voice_id)
                display_names.append(f"{name} (Cloned)")
        
        return display_names
    
    def get_voice_id_from_display(self, display_name: str) -> str:
        """Get voice ID from display name"""
        if display_name == "Standard Voice":
            return "standard"
        
        for voice_id in self.get_available_voices():
            if voice_id != "standard":
                voice_info = self.voices.get(voice_id, {})
                name = voice_info.get("name", voice_id)
                if display_name == f"{name} (Cloned)":
                    return voice_id
        
        return "standard"
    
    def add_voice(self, voice_id: str, name: str, description: str = ""):
        """Add a new voice to the configuration"""
        self.voices[voice_id] = {
            "name": name,
            "description": description,
            "type": "cloned",
            "enabled": True,
            "created_at": datetime.now().isoformat()
        }
        self.save_voice_config(self.voices)

# --- Ollama Integration ---
class OllamaChat:
    def __init__(self, base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL):
        self.base_url = base_url
        self.default_model = model
        self.history = []

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        print("🤖 Talking to Ollama...")
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.default_model, "messages": self.history, "stream": False},
                timeout=180
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            self.history.append({"role": "assistant", "content": content})
            return content
        except requests.RequestException as e:
            return f"[Ollama Error] {str(e)}"

# --- TTS ---
class TTSGenerator:
    def __init__(self):
        self.voice_model = None
        self.tacotron2 = None
        self.hifi_gan = None
        self.voice_manager = VoiceManager()
        os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)

    def preprocess_text(self, text: str) -> str:
        # Convert numbers to words
        text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
        # Clean up common problematic characters
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _generate_filename(self, text: str, speaker_id: str, fmt: str = "wav") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = re.sub(r'[^\w\s-]', '', text[:30]).strip().replace(" ", "_") or "audio"
        safe_speaker = re.sub(r'[^\w\s-]', '', speaker_id).replace(' ', '_')
        return os.path.join(GENERATED_AUDIO_DIR, f"{ts}_{safe_speaker}_{safe_text}.{fmt}")

    def run_standard_tts(self, text: str) -> str:
        if self.tacotron2 is None or self.hifi_gan is None:
            print("Loading standard TTS models...")
            self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmp_tts")
            self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmp_vocoder")

        mel_outputs, _, _ = self.tacotron2.encode_batch([text])
        waveform = self.hifi_gan.decode_batch(mel_outputs).squeeze().detach().cpu()

        file_path = self._generate_filename(text, STANDARD_VOICE_NAME)
        torchaudio.save(file_path, waveform.unsqueeze(0), 22050)

        print(f"[DEBUG] ✅ Saved standard audio at: {file_path}")
        return file_path

    def run_voice_clone_tts(self, text: str, speaker_id: str) -> str:
        if self.voice_model is None:
            print("Loading voice cloning model...")
            self.voice_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

        reference_audio = os.path.join(VOICE_SAMPLES_DIR, speaker_id, "reference.wav")
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"❌ Reference audio for '{speaker_id}' not found at {reference_audio}")

        path = self._generate_filename(text, speaker_id)
        self.voice_model.tts_to_file(
            text=text,
            file_path=path,
            speaker_wav=reference_audio,
            language="en",
            temperature=0.9,
            split_sentences=True
        )

        print(f"[DEBUG] ✅ Cloned voice saved at: {path}")
        return path

    def generate_audio(self, text: str, voice_id: str) -> str:
        """Generate audio with the specified voice"""
        clean_text = self.preprocess_text(text)
        
        if not clean_text.strip():
            raise ValueError("❌ No valid text to convert to speech")
        
        try:
            if voice_id == "standard":
                return self.run_standard_tts(clean_text)
            else:
                return self.run_voice_clone_tts(clean_text, voice_id)
        except Exception as e:
            print(f"[ERROR] Audio generation failed: {str(e)}")
            raise

# --- Global instances ---
voice_manager = VoiceManager()
chat_engine = OllamaChat()
tts = TTSGenerator()

# --- Gradio UI Functions ---
def generate_script(topic):
    """Generate script based on topic"""
    if not topic.strip():
        return "Please enter a topic first!"
    
    prompt = f"Generate a fast-paced video script for: {topic}"
    return chat_engine.chat(prompt)

def get_available_voices():
    """Get current available voices"""
    return voice_manager.get_voice_display_names()

def generate_audio_wrapper(text, voice_display_name):
    """Generate audio from text and selected voice"""
    if not text.strip():
        return None, "No text to convert to speech!"
    
    if not voice_display_name:
        return None, "Please select a voice!"
    
    try:
        voice_id = voice_manager.get_voice_id_from_display(voice_display_name)
        audio_path = tts.generate_audio(text, voice_id)
        return audio_path, f"Audio generated successfully with {voice_display_name}!"
    except Exception as e:
        return None, f"Error generating audio: {str(e)}"

def upload_voice_sample(audio_file, voice_name):
    """Upload voice sample for cloning"""
    if not audio_file or not voice_name.strip():
        return "Please provide both audio file and voice name!", gr.Dropdown()
    
    try:
        # Create voice directory
        voice_id = re.sub(r'[^\w\s-]', '', voice_name.lower().replace(' ', '_'))
        voice_dir = os.path.join(VOICE_SAMPLES_DIR, voice_id)
        os.makedirs(voice_dir, exist_ok=True)
        
        # Save reference audio
        reference_path = os.path.join(voice_dir, "reference.wav")
        
        # Copy uploaded file to reference location
        import shutil
        shutil.copy2(audio_file, reference_path)
        
        # Add to voice manager
        voice_manager.add_voice(voice_id, voice_name, "Custom uploaded voice")
        
        # Update dropdown choices
        new_choices = get_available_voices()
        updated_dropdown = gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)
        
        return f"Voice '{voice_name}' uploaded successfully!", updated_dropdown
    
    except Exception as e:
        return f"Error uploading voice: {str(e)}", gr.Dropdown()

def refresh_voices():
    """Refresh voice dropdown"""
    choices = get_available_voices()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

# --- Gradio App ---
with gr.Blocks(title="TTS Script Generator") as demo:
    gr.Markdown("# 🎙️ TTS Script Generator")
    gr.Markdown("Generate scripts with AI and convert them to speech using different voices")
    
    with gr.Tab("Script & Audio Generation"):
        topic_input = gr.Textbox(label="Enter Topic", placeholder="e.g., The future of AI")
        
        generate_btn = gr.Button("Generate Script", variant="primary")
        
        script_output = gr.Textbox(label="Generated Script", lines=8)
        
        voice_dropdown = gr.Dropdown(
            label="Select Voice",
            choices=get_available_voices(),
            value=get_available_voices()[0] if get_available_voices() else None
        )
        
        refresh_btn = gr.Button("Refresh Voices")
        
        audio_btn = gr.Button("Generate Audio", variant="secondary")
        
        audio_output = gr.Audio(label="Generated Audio", type="filepath")
        status_output = gr.Textbox(label="Status", lines=2)
    
    with gr.Tab("Voice Management"):
        gr.Markdown("## Upload Voice Sample")
        gr.Markdown("Upload a clear audio sample (10-30 seconds) for voice cloning. WAV format recommended.")
        
        upload_audio = gr.File(label="Audio Sample", file_types=[".wav", ".mp3", ".flac"])
        upload_name = gr.Textbox(label="Voice Name", placeholder="Enter a name for this voice")
        upload_btn = gr.Button("Upload Voice")
        upload_status = gr.Textbox(label="Upload Status", lines=2)
        
        gr.Markdown("### Tips for best results:")
        gr.Markdown("- Use clear, high-quality audio\n- Single speaker only\n- Minimal background noise\n- Natural speaking pace")
    
    # Event handlers
    generate_btn.click(
        fn=generate_script,
        inputs=topic_input,
        outputs=script_output
    )
    
    audio_btn.click(
        fn=generate_audio_wrapper,
        inputs=[script_output, voice_dropdown],
        outputs=[audio_output, status_output]
    )
    
    refresh_btn.click(
        fn=refresh_voices,
        outputs=voice_dropdown
    )
    
    upload_btn.click(
        fn=upload_voice_sample,
        inputs=[upload_audio, upload_name],
        outputs=[upload_status, voice_dropdown]
    )

if __name__ == "__main__":
    print("🚀 Starting TTS Script Generator...")
    print(f"📁 Voice samples directory: {VOICE_SAMPLES_DIR}")
    print(f"🔊 Generated audio directory: {GENERATED_AUDIO_DIR}")
    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)