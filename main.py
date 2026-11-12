import os
import transformers
# --- Transformers 4.40+ Compatibility Monkeypatch for Coqui TTS ---
if not hasattr(transformers, "BeamSearchScorer"):
    try:
        import transformers.generation.beam_search as bs
        transformers.BeamSearchScorer = bs.BeamSearchScorer
    except ImportError:
        pass
# ------------------------------------------------------------------
import re
import uuid
import tempfile
import requests
import asyncio
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Literal, Optional
from num2words import num2words
import gradio as gr
import json
import shutil
import numpy as np
from numpy import ndarray
from pathlib import Path

# --- Gradio 4 Compatibility Shim ---
if not hasattr(gr, "Sidebar"):
    class SidebarShim(gr.Column):
        def __init__(self, *args, **kwargs):
            kwargs.pop("position", None)
            super().__init__(*args, **kwargs)
    gr.Sidebar = SidebarShim

# Audio imports
import torchaudio
try:
    from speechbrain.inference.TTS import Tacotron2
    from speechbrain.inference.vocoders import HIFIGAN
except ImportError as e:
    logger.error(f"Failed to import SpeechBrain inference modules: {e}")
    logger.error("Please ensure 'speechbrain' is installed correctly and all dependencies like 'torchvision' are available.")
    Tacotron2 = None
    HIFIGAN = None
from TTS.api import TTS
from fastrtc import Stream, ReplyOnPause, WebRTC, AdditionalOutputs, get_stt_model, get_tts_model
import torch



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enhanced Constants ---
class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ai.izdrail.com")
    DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    VOICE_SAMPLES_DIR = Path("voice_samples")
    GENERATED_AUDIO_DIR = Path("generated_audio")
    STANDARD_VOICE_NAME = "standard"
    VOICE_CONFIG_FILE = "voice_config.json"
    MAX_TEXT_LENGTH = 1000
    MAX_VOICE_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

# Ensure directories exist
Config.VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
Config.GENERATED_AUDIO_DIR.mkdir(exist_ok=True)

# --- Enhanced Voice Management ---
class VoiceManager:
    def __init__(self):
        self.voice_config_path = Config.VOICE_CONFIG_FILE
        self.voices = self._load_voice_config()
        self._validate_voice_files()

    def _load_voice_config(self) -> Dict:
        """Load voice configuration from JSON file with better error handling"""
        if Path(self.voice_config_path).exists():
            try:
                with open(self.voice_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded voice config with {len(config)} voices")
                return config
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading voice config: {e}")
                return self._create_default_config()

        return self._create_default_config()

    def _create_default_config(self) -> Dict:
        """Create and save default configuration"""
        default_config = {
            "standard": {
                "name": "Standard Voice",
                "description": "Default synthetic voice",
                "type": "standard",
                "enabled": True,
                "created_at": datetime.now().isoformat(),
                "sample_rate": 22050
            }
        }
        self.save_voice_config(default_config)
        return default_config

    def _validate_voice_files(self):
        """Validate that voice files still exist and disable missing ones"""
        updated = False
        for voice_id, voice_info in self.voices.items():
            if voice_id == "standard":
                continue

            voice_path = Config.VOICE_SAMPLES_DIR / voice_id
            reference_file = voice_path / "reference.wav"

            if not reference_file.exists() and voice_info.get("enabled", True):
                logger.warning(f"Voice '{voice_id}' reference file missing, disabling")
                self.voices[voice_id]["enabled"] = False
                updated = True

        if updated:
            self.save_voice_config(self.voices)

    def save_voice_config(self, config: Dict):
        """Save voice configuration with atomic write"""
        try:
            # Write to temporary file first
            temp_path = f"{self.voice_config_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.rename(temp_path, self.voice_config_path)
            logger.info("Voice configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving voice config: {e}")

    def get_available_voices(self) -> List[str]:
        """Get list of available and enabled voice IDs"""
        voices = []

        # Add standard voice
        if self.voices.get("standard", {}).get("enabled", True):
            voices.append("standard")

        # Add cloned voices
        for voice_dir in Config.VOICE_SAMPLES_DIR.iterdir():
            if not voice_dir.is_dir():
                continue

            voice_id = voice_dir.name
            reference_file = voice_dir / "reference.wav"

            if reference_file.exists():
                voice_info = self.voices.get(voice_id, {})
                if voice_info.get("enabled", True):
                    voices.append(voice_id)

        return voices if voices else ["standard"]

    def get_voice_display_names(self) -> List[str]:
        """Get formatted display names for voices"""
        voices = self.get_available_voices()
        display_names = []

        for voice_id in voices:
            if voice_id == "standard":
                display_names.append("🎤 Standard Voice")
            else:
                voice_info = self.voices.get(voice_id, {})
                name = voice_info.get("name", voice_id)
                display_names.append(f"🎭 {name} (Cloned)")

        return display_names

    def get_voice_id_from_display(self, display_name: str) -> str:
        """Extract voice ID from display name"""
        if display_name.startswith("🎤"):
            return "standard"

        # Remove emoji and extract name
        clean_name = display_name.replace("🎭 ", "").replace(" (Cloned)", "")

        for voice_id in self.get_available_voices():
            if voice_id != "standard":
                voice_info = self.voices.get(voice_id, {})
                if voice_info.get("name", voice_id) == clean_name:
                    return voice_id

        return "standard"

    def add_voice(self, voice_id: str, name: str, description: str = "",
                  metadata: Optional[Dict] = None) -> bool:
        """Add a new voice with enhanced metadata"""
        try:
            self.voices[voice_id] = {
                "name": name,
                "description": description,
                "type": "cloned",
                "enabled": True,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            self.save_voice_config(self.voices)
            logger.info(f"Added new voice: {name} ({voice_id})")
            return True
        except Exception as e:
            logger.error(f"Error adding voice: {e}")
            return False

    def remove_voice(self, voice_id: str) -> bool:
        """Remove a voice and its files"""
        if voice_id == "standard":
            return False

        try:
            # Remove voice directory
            voice_path = Config.VOICE_SAMPLES_DIR / voice_id
            if voice_path.exists():
                shutil.rmtree(voice_path)

            # Remove from config
            if voice_id in self.voices:
                del self.voices[voice_id]
                self.save_voice_config(self.voices)

            logger.info(f"Removed voice: {voice_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing voice {voice_id}: {e}")
            return False

# --- Enhanced Ollama Integration ---
class OllamaChat:
    def __init__(self, base_url=Config.OLLAMA_BASE_URL, model=Config.DEFAULT_MODEL):
        self.base_url = base_url
        self.default_model = model
        self.history = []
        self.session = requests.Session()
        self.session.timeout = 180

    def chat(self, user_input: str) -> str:
        """Enhanced chat with better error handling and validation"""
        if not user_input.strip():
            return "Please provide some input!"

        if len(user_input) > Config.MAX_TEXT_LENGTH:
            return f"Input too long! Maximum {Config.MAX_TEXT_LENGTH} characters allowed."

        self.history.append({"role": "user", "content": user_input})
        logger.info("Sending request to Ollama...")

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.default_model,
                    "messages": self.history,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                }
            )
            response.raise_for_status()

            content = response.json().get("message", {}).get("content", "")
            if not content:
                return "No response received from Ollama"

            self.history.append({"role": "assistant", "content": content})
            logger.info("Received response from Ollama")
            return content

        except requests.RequestException as e:
            error_msg = f"Ollama connection error: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def reset_history(self):
        """Reset conversation history"""
        self.history = []
        logger.info("Chat history reset")

    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            # Add trailing slash to avoid redirection that might change POST to GET
            url = f"{self.base_url}/api/show/"
            print(f"DEBUG_LLM: Performing POST to {url} with json: {{'name': '{self.default_model}'}}", flush=True)
            response = self.session.post(url, json={"name": self.default_model})
            print(f"DEBUG_LLM: Received {response.status_code} for {response.url}", flush=True)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"DEBUG_LLM: Error getting model info: {e}", flush=True)
            return {}

    def get_available_models(self) -> List[str]:
        """Get list of available models from the server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return models
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return [self.default_model]

    def set_model(self, model_name: str):
        """Set the model to use"""
        self.default_model = model_name
        logger.info(f"Chat model set to: {model_name}")

# --- Enhanced TTS Generator ---
class TTSGenerator:
    def __init__(self):
        self.voice_model = None
        self.tacotron2 = None
        self.hifi_gan = None
        self.voice_manager = VoiceManager()
        self._model_cache = {}

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Convert numbers to words (handle ranges)
        text = re.sub(r'\d+-\d+', lambda m: f"{num2words(int(m.group(0).split('-')[0]))} to {num2words(int(m.group(0).split('-')[1]))}", text)
        text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)

        # Handle common abbreviations
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'e.g.': 'for example',
            'i.e.': 'that is'
        }

        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)

        # Clean up problematic characters
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _generate_filename(self, text: str, speaker_id: str, fmt: str = "wav") -> Path:
        """Generate unique filename with better sanitization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_text = re.sub(r'[^\w\s-]', '', text[:30]).strip().replace(" ", "_") or "audio"
        safe_speaker = re.sub(r'[^\w\s-]', '', speaker_id).replace(' ', '_')
        unique_id = str(uuid.uuid4())[:8]

        filename = f"{timestamp}_{safe_speaker}_{safe_text}_{unique_id}.{fmt}"
        return Config.GENERATED_AUDIO_DIR / filename

    def run_standard_tts(self, text: str) -> Path:
        """Enhanced standard TTS with caching"""
        cache_key = "standard_tts"

        if cache_key not in self._model_cache:
            logger.info("Loading standard TTS models...")
            try:
                self._model_cache[cache_key] = {
                    'tacotron2': Tacotron2.from_hparams(
                        source="speechbrain/tts-tacotron2-ljspeech",
                        savedir="tmp_tts"
                    ),
                    'hifi_gan': HIFIGAN.from_hparams(
                        source="speechbrain/tts-hifigan-ljspeech",
                        savedir="tmp_vocoder"
                    )
                }
                logger.info("Standard TTS models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading standard TTS models: {e}")
                raise

        models = self._model_cache[cache_key]

        try:
            mel_outputs, _, _ = models['tacotron2'].encode_batch([text])
            waveform = models['hifi_gan'].decode_batch(mel_outputs).squeeze().detach().cpu()

            file_path = self._generate_filename(text, Config.STANDARD_VOICE_NAME)
            torchaudio.save(str(file_path), waveform.unsqueeze(0), 22050)

            logger.info(f"Standard TTS audio saved: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error in standard TTS generation: {e}")
            raise

    def run_voice_clone_tts(self, text: str, speaker_id: str) -> Path:
        """Enhanced voice cloning with better error handling"""
        if self.voice_model is None:
            logger.info("Loading voice cloning model...")
            try:
                self.voice_model = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                    gpu=False  # Set based on availability
                )
                logger.info("Voice cloning model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading voice cloning model: {e}")
                raise

        reference_audio = Config.VOICE_SAMPLES_DIR / speaker_id / "reference.wav"
        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio for '{speaker_id}' not found")

        file_path = self._generate_filename(text, speaker_id)

        try:
            self.voice_model.tts_to_file(
                text=text,
                file_path=str(file_path),
                speaker_wav=str(reference_audio),
                language="en",
                temperature=0.6,
                split_sentences=True
            )

            logger.info(f"Cloned voice audio saved: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error in voice cloning: {e}")
            raise

    def generate_audio(self, text: str, voice_id: str) -> Path:
        """Generate audio with enhanced validation"""
        if not text or not text.strip():
            raise ValueError("No valid text provided for speech synthesis")

        if len(text) > Config.MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long! Maximum {Config.MAX_TEXT_LENGTH} characters allowed")

        clean_text = self.preprocess_text(text)
        if not clean_text.strip():
            raise ValueError("No valid content after text preprocessing")

        logger.info(f"Generating audio for voice '{voice_id}' with text length {len(clean_text)}")

        try:
            if voice_id == "standard":
                return self.run_standard_tts(clean_text)
            else:
                return self.run_voice_clone_tts(clean_text, voice_id)
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise

# --- Enhanced Voice Chat Handler (Global Instance) ---
# --- Enhanced Helper Functions ---



# --- Enhanced Helper Functions ---
def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """Validate uploaded audio file"""
    try:
        if not file_path:
            return False, "No file provided"

        file_path = Path(file_path)
        if not file_path.exists():
            return False, "File not found"

        # Check file size
        if file_path.stat().st_size > Config.MAX_VOICE_UPLOAD_SIZE:
            return False, f"File too large (max {Config.MAX_VOICE_UPLOAD_SIZE // 1024 // 1024}MB)"

        # Check file extension
        if file_path.suffix.lower() not in Config.SUPPORTED_AUDIO_FORMATS:
            return False, f"Unsupported format. Use: {', '.join(Config.SUPPORTED_AUDIO_FORMATS)}"

        # Try to load audio to validate
        try:
            waveform, sample_rate = torchaudio.load(str(file_path))
            duration = waveform.shape[1] / sample_rate

            if duration < 5:
                return False, "Audio too short (minimum 5 seconds)"
            if duration > 120:
                return False, "Audio too long (maximum 2 minutes)"

            return True, f"Valid audio: {duration:.1f}s, {sample_rate}Hz"
        except Exception as e:
            return False, f"Invalid audio file: {str(e)}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"

# --- UI Helper Functions (Enhanced) ---
# --- Removed Script Generation (Replaced by FastRTC) ---

def generate_audio_wrapper(text, voice_display_name):
    """Enhanced audio generation wrapper"""
    if not text or not text.strip():
        return None, "❌ No text to convert to speech!"

    if not voice_display_name:
        return None, "❌ Please select a voice!"

    try:
        voice_id = voice_manager.get_voice_id_from_display(voice_display_name)
        audio_path = tts.generate_audio(text, voice_id)

        success_msg = f"✅ Audio generated successfully with {voice_display_name}!"
        logger.info(success_msg)
        return str(audio_path), success_msg

    except Exception as e:
        error_msg = f"❌ Error generating audio: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def upload_voice_sample(audio_file, voice_name):
    """Enhanced voice sample upload"""
    if not audio_file or not voice_name or not voice_name.strip():
        return "❌ Please provide both audio file and voice name!", gr.Dropdown()

    # Validate audio file
    is_valid, validation_msg = validate_audio_file(audio_file)
    if not is_valid:
        return f"❌ {validation_msg}", gr.Dropdown()

    try:
        # Create safe voice ID
        voice_id = re.sub(r'[^\w\s-]', '', voice_name.lower().replace(' ', '_'))
        if not voice_id:
            return "❌ Invalid voice name!", gr.Dropdown()

        if voice_id in voice_manager.voices:
            return f"❌ Voice '{voice_name}' already exists!", gr.Dropdown()

        # Create voice directory
        voice_dir = Config.VOICE_SAMPLES_DIR / voice_id
        voice_dir.mkdir(exist_ok=True)

        # Process and save audio
        reference_path = voice_dir / "reference.wav"

        # Load and resample if needed
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != 22050:
            resampler = torchaudio.transforms.Resample(sample_rate, 22050)
            waveform = resampler(waveform)

        # Save as WAV
        torchaudio.save(str(reference_path), waveform, 22050)

        # Add to voice manager with metadata
        metadata = {
            "original_file": Path(audio_file).name,
            "duration": waveform.shape[1] / 22050,
            "sample_rate": 22050,
            "validation_info": validation_msg
        }

        success = voice_manager.add_voice(voice_id, voice_name,
                                        "Custom uploaded voice", metadata)

        if success:
            new_choices = voice_manager.get_voice_display_names()
            updated_dropdown = gr.Dropdown(choices=new_choices,
                                         value=new_choices[0] if new_choices else None)
            return f"✅ Voice '{voice_name}' uploaded successfully! {validation_msg}", updated_dropdown
        else:
            return "❌ Failed to save voice configuration!", gr.Dropdown()

    except Exception as e:
        error_msg = f"❌ Error uploading voice: {str(e)}"
        logger.error(error_msg)
        return error_msg, gr.Dropdown()

def refresh_voices():
    """Refresh voice dropdown with current status"""
    try:
        voice_manager._validate_voice_files()  # Revalidate files
        choices = voice_manager.get_voice_display_names()
        logger.info(f"Refreshed voices: {len(choices)} available")
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        logger.error(f"Error refreshing voices: {e}")
        return gr.Dropdown(choices=["🎤 Standard Voice"], value="🎤 Standard Voice")

def clear_generated_text():
    """Clear the generated text area"""
    return ""

def clear_audio():
    """Clear audio output"""
    return None, ""

def reset_chat():
    """Reset chat history"""
    chat_engine.reset_history()
    return "✅ Chat history reset successfully!"

voice_manager = VoiceManager()
chat_engine = OllamaChat()
tts = TTSGenerator()

# --- FastRTC Voice Chat Handler ---
def voice_chat_response_handler(audio: Tuple[int, np.ndarray], *args):
    """Real-time voice-to-voice handler for FastRTC"""
    try:
        # In inputs=[webrtc_component, chat_voice_dropdown]:
        # args[0] is webrtc_component value
        # args[1] is chat_voice_dropdown value
        voice_display_name = args[1] if len(args) > 1 else "🎤 Standard Voice"
        print(f"DEBUG_RTC: Handler received {len(args)} extra args. voice={voice_display_name}", flush=True)
        
        sample_rate, audio_data = audio
        
        # 1. STT (Speech to Text)
        stt_model = get_stt_model()
        user_text = stt_model.stt(sample_rate, audio_data)
        logger.info(f"FastRTC STT: {user_text}")
        
        if not user_text.strip():
            yield None, AdditionalOutputs(None)
            return

        # 2. LLM Response
        ai_response = chat_engine.chat(user_text)
        logger.info(f"FastRTC LLM: {ai_response}")

        # 3. TTS (Text to Speech)
        voice_id = voice_manager.get_voice_id_from_display(voice_display_name)
        audio_path = tts.generate_audio(ai_response, voice_id)
        
        # Load the generated audio
        waveform, file_sample_rate = torchaudio.load(str(audio_path))
        yield (file_sample_rate, waveform.squeeze().numpy()), AdditionalOutputs(ai_response)
        
    except Exception as e:
        logger.error(f"FastRTC error: {e}")
        yield None, AdditionalOutputs(f"Error: {str(e)}")

# Define additional inputs/outputs for the stream
chat_voice_dropdown = gr.Dropdown(
    label="🎭 Chat Voice",
    choices=voice_manager.get_voice_display_names(),
    value=voice_manager.get_voice_display_names()[0] if voice_manager.get_voice_display_names() else None,
    render=False
)
ai_response_textbox = gr.Textbox(label="Last AI Response", render=False)

# Define the speech handler
voice_chat_handler = ReplyOnPause(voice_chat_response_handler)

# Initialize FastRTC Stream
voice_chat_stream = Stream(
    handler=voice_chat_handler,
    modality="audio",
    mode="send-receive",
    additional_inputs=[chat_voice_dropdown],
    additional_outputs=[ai_response_textbox],
    additional_outputs_handler=lambda x: x
)

# --- Enhanced Gradio Interface ---
def create_interface():
    """Create the complete Gradio interface"""

    with gr.Blocks(
        title="🎙️ Advanced TTS Studio",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .header-text {
            text-align: center;
            color: #2d3748;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 12px;
            margin: 0.5rem;
        }
        .status-success {
            color: #10b981 !important;
            font-weight: bold;
        }
        .status-error {
            color: #ef4444 !important;
            font-weight: bold;
        }
        """,
    ) as interface:

        # Header
        gr.HTML("""
        <div class="header-text">
            <h1>🎙️ Advanced Text-to-Speech Studio</h1>
            <p>Generate professional voiceovers with AI-powered script generation and voice cloning</p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # --- Main TTS Tab ---
            with gr.Tab("🎯 Quick TTS", elem_id="main-tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Text Input")
                        text_input = gr.Textbox(
                            label="Enter text to convert to speech",
                            placeholder="Type or paste your text here...",
                            lines=5,
                            max_lines=10
                        )

                        with gr.Row():
                            voice_dropdown = gr.Dropdown(
                                label="🎭 Select Voice",
                                choices=voice_manager.get_voice_display_names(),
                                value=voice_manager.get_voice_display_names()[0] if voice_manager.get_voice_display_names() else None,
                                allow_custom_value=False
                            )
                            refresh_btn = gr.Button("🔄 Refresh Voices", size="sm")

                        with gr.Row():
                            generate_audio_btn = gr.Button(
                                "🎵 Generate Audio",
                                variant="primary",
                                size="lg"
                            )
                            clear_text_btn = gr.Button("🗑️ Clear Text", size="sm")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🔊 Audio Output")
                        audio_output = gr.Audio(
                            label="Generated Audio",
                            type="filepath",
                            interactive=False
                        )
                        status_output = gr.Markdown("")
                        clear_audio_btn = gr.Button("🗑️ Clear Audio", size="sm")

            # --- Live Voice Chat Tab ---
            with gr.Tab("🎙️ Live Voice Chat", elem_id="voice-chat-tab"):
                gr.Markdown("### 🔊 Real-Time Conversation")
                gr.Markdown("Speak and the AI will respond in real-time. Turn-taking is handled automatically.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        webrtc_component = WebRTC(
                            label="Conversation Control",
                            rtc_configuration=voice_chat_stream.rtc_configuration,
                            mode="send-receive",
                            modality="audio"
                        )
                    with gr.Column(scale=1):
                        chat_voice_dropdown.render()
                        ai_response_textbox.render()

                webrtc_component.stream(
                    fn=voice_chat_handler,
                    inputs=[webrtc_component, chat_voice_dropdown],
                    outputs=[webrtc_component, ai_response_textbox],
                    time_limit=90
                )

            # --- Voice Management Tab ---
            with gr.Tab("🎭 Voice Cloning", elem_id="voice-tab"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📤 Upload Voice Sample")
                        gr.Markdown("""
                        Upload a clear audio sample (5-120 seconds) to create a cloned voice.
                        **Tips for best results:**
                        - Use high-quality, clear recordings
                        - Avoid background noise
                        - Include varied intonation
                        - Minimum 5 seconds, maximum 2 minutes
                        """)

                        voice_upload = gr.Audio(
                            label="Voice Sample",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )

                        voice_name_input = gr.Textbox(
                            label="Voice Name",
                            placeholder="e.g., 'John's Voice' or 'Professional Narrator'",
                            max_lines=1
                        )

                        upload_voice_btn = gr.Button(
                            "📤 Upload Voice",
                            variant="primary",
                            size="lg"
                        )

                        upload_status = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("### 🎭 Manage Voices")

                        # Voice list display
                        voice_list = gr.Dataframe(
                            headers=["Voice Name", "Type", "Created", "Status"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Available Voices"
                        )

                        with gr.Row():
                            refresh_voice_list_btn = gr.Button("🔄 Refresh List", size="sm")
                            delete_voice_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop")



            # --- Settings Tab ---
            with gr.Tab("⚙️ Settings", elem_id="settings-tab"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔧 Application Settings")

                        model_selector = gr.Dropdown(
                            label="AI Model",
                            choices=chat_engine.get_available_models(),
                            value=Config.DEFAULT_MODEL,
                            interactive=True,
                            allow_custom_value=False
                        )

                        ollama_url = gr.Textbox(
                            label="Ollama Server URL",
                            value=Config.OLLAMA_BASE_URL,
                            interactive=False
                        )

                        gr.Markdown("### 🎙️ Audio Settings")

                        default_sample_rate = gr.Dropdown(
                            label="Default Sample Rate",
                            choices=["16000", "22050", "44100", "48000"],
                            value="22050",
                            info="Sample rate for audio generation"
                        )

                        audio_quality = gr.Dropdown(
                            label="Audio Quality",
                            choices=["Low", "Medium", "High", "Ultra"],
                            value="High",
                            info="Quality setting for TTS generation"
                        )

                        test_connection_btn = gr.Button("🔍 Test Connections")
                        connection_status = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("### 📊 Statistics")

                        stats_display = gr.JSON(
                            label="System Statistics",
                            value={}
                        )

                        refresh_stats_btn = gr.Button("🔄 Refresh Stats")

                        gr.Markdown("### 🧹 Maintenance")

                        with gr.Row():
                            clean_cache_btn = gr.Button("🗑️ Clear Cache")
                            export_config_btn = gr.Button("💾 Export Config")

        # --- Event Handlers ---

        # Main TTS functionality
        generate_audio_btn.click(
            fn=generate_audio_wrapper,
            inputs=[text_input, voice_dropdown],
            outputs=[audio_output, status_output]
        )

        refresh_btn.click(
            fn=refresh_voices,
            outputs=[voice_dropdown]
        )

        clear_text_btn.click(
            fn=lambda: "",
            outputs=[text_input]
        )

        clear_audio_btn.click(
            fn=lambda: (None, ""),
            outputs=[audio_output, status_output]
        )

        # Reset chat handler removed (now part of Live Voice Chat stream if needed)

        # Voice cloning
        upload_voice_btn.click(
            fn=upload_voice_sample,
            inputs=[voice_upload, voice_name_input],
            outputs=[upload_status, voice_dropdown]
        )

        # Settings and maintenance functions
        def test_connections():
            """Test connections to external services"""
            results = []

            # Test Ollama connection
            try:
                response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    results.append("✅ Ollama: Connected")
                else:
                    results.append(f"❌ Ollama: Error {response.status_code}")
            except Exception as e:
                results.append(f"❌ Ollama: {str(e)}")

            # Test FastRTC (basic check)
            # try:
            #     results.append("✅ FastRTC: Available")
            # except Exception as e:
            #     results.append(f"❌ FastRTC: {str(e)}")

            return "\n".join(results)

        def get_system_stats():
            """Get system statistics"""
            try:
                stats = {
                    "voices_available": len(voice_manager.get_available_voices()),
                    "generated_files": len(list(Config.GENERATED_AUDIO_DIR.glob("*.wav"))),
                    "voice_samples": len(list(Config.VOICE_SAMPLES_DIR.iterdir())),
                    "chat_history_length": len(chat_engine.history),
                    "model_info": chat_engine.get_model_info().get("details", {}),
                    "audio_streaming": "Enabled"
                }
                return stats
            except Exception as e:
                return {"error": str(e)}

        def update_voice_list():
            """Update the voice list display"""
            try:
                voices_data = []
                for voice_id in voice_manager.get_available_voices():
                    voice_info = voice_manager.voices.get(voice_id, {})
                    voices_data.append([
                        voice_info.get("name", voice_id),
                        voice_info.get("type", "unknown"),
                        voice_info.get("created_at", "unknown")[:10] if voice_info.get("created_at") else "unknown",
                        "✅ Active" if voice_info.get("enabled", True) else "❌ Disabled"
                    ])
                return voices_data
            except Exception as e:
                return [["Error", str(e), "", ""]]

        def clean_cache():
            """Clean generated audio cache"""
            try:
                count = 0
                for file in Config.GENERATED_AUDIO_DIR.glob("*.wav"):
                    if file.stat().st_mtime < (datetime.now().timestamp() - 86400):  # 1 day old
                        file.unlink()
                        count += 1
                return f"✅ Cleaned {count} old files from cache"
            except Exception as e:
                return f"❌ Error cleaning cache: {str(e)}"

        def export_config():
            """Export configuration"""
            try:
                config_data = {
                    "voices": voice_manager.voices,
                    "settings": {
                        "ollama_url": Config.OLLAMA_BASE_URL,
                        "model": Config.DEFAULT_MODEL,
                        # "audio_backend": "FastRTC"
                    },
                    "exported_at": datetime.now().isoformat()
                }

                export_path = Path("tts_config_export.json")
                with open(export_path, 'w') as f:
                    json.dump(config_data, f, indent=2)

                return f"✅ Configuration exported to {export_path}"
            except Exception as e:
                return f"❌ Export failed: {str(e)}"

        def update_chat_voice(voice_display_name):
            """Update the voice used for chat responses"""
            try:
                voice_id = voice_manager.get_voice_id_from_display(voice_display_name)
                # global_voice_chat_handler.set_voice(voice_id)
                return f"✅ Chat voice updated to: {voice_display_name}"
            except Exception as e:
                return f"❌ Error updating chat voice: {str(e)}"



        def update_ai_model(model_name):
            """Update the AI model"""
            try:
                chat_engine.set_model(model_name)
                return f"✅ Model updated to: {model_name}"
            except Exception as e:
                return f"❌ Error updating model: {str(e)}"

        # Settings event handlers
        test_connection_btn.click(
            fn=test_connections,
            outputs=[connection_status]
        )

        refresh_stats_btn.click(
            fn=get_system_stats,
            outputs=[stats_display]
        )

        refresh_voice_list_btn.click(
            fn=update_voice_list,
            outputs=[voice_list]
        )

        clean_cache_btn.click(
            fn=clean_cache,
            outputs=[connection_status]
        )

        export_config_btn.click(
            fn=export_config,
            outputs=[connection_status]
        )



        model_selector.change(
            fn=update_ai_model,
            inputs=[model_selector],
            outputs=[connection_status]
        )

        # Initialize interface with current data
        interface.load(
            fn=lambda: (
                update_voice_list(),
                get_system_stats(),
                gr.Dropdown(choices=chat_engine.get_available_models())
            ),
            outputs=[voice_list, stats_display, model_selector]
        )

    return interface

# --- Remove the duplicate Enhanced Voice Chat Response Handler section ---

# --- Launch Application ---
def main():
    """Main application entry point with FastRTC support"""
    logger.info("Starting Enhanced TTS Studio with FastRTC...")

    # Validate dependencies
    try:
        logger.info("Validating system components...")

        # Check if required directories exist
        if not Config.VOICE_SAMPLES_DIR.exists():
            Config.VOICE_SAMPLES_DIR.mkdir(parents=True)
            logger.info("Created voice samples directory")

        if not Config.GENERATED_AUDIO_DIR.exists():
            Config.GENERATED_AUDIO_DIR.mkdir(parents=True)
            logger.info("Created generated audio directory")

        # Test Ollama connection
        try:
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama connection successful")
            else:
                logger.warning(f"⚠️ Ollama returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama connection failed: {e}")

        # Initialize FastRTC components
        # try:
        #     logger.info("✅ FastRTC components initialized")
        # except Exception as e:
        #     logger.error(f"❌ FastRTC initialization failed: {e}")

        logger.info("System validation complete")

    except Exception as e:
        logger.error(f"System validation failed: {e}")
        raise

    # Create and launch interface
    try:
        interface = create_interface()

        logger.info("🚀 Launching TTS Studio interface with FastRTC support...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=1602,
            share=False,
            debug=False,
            show_error=True,

            max_threads=10
        )

    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        raise

if __name__ == "__main__":
    main()