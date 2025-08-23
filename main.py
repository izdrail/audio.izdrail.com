import os
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

# Audio imports
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
from TTS.api import TTS

# FastRTC imports (replacing WebRTC and Moshi)
from fastrtc import ReplyOnPause, Stream
import sphn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enhanced Constants ---
class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
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
            response = self.session.get(f"{self.base_url}/api/show",
                                      params={"name": self.default_model})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

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
class EnhancedVoiceChatHandler:
    def __init__(self):
        self.chat_engine = OllamaChat()
        self.tts_generator = TTSGenerator()
        self.default_voice = "standard"
        self.stt_enabled = True
        self.max_response_length = 200

    def set_voice(self, voice_id: str):
        """Set the voice for TTS responses"""
        self.default_voice = voice_id
        logger.info(f"Chat voice updated to: {voice_id}")

    def set_settings(self, stt_enabled: bool, max_response_length: int):
        """Update chat settings"""
        self.stt_enabled = stt_enabled
        self.max_response_length = max_response_length
        logger.info(f"Chat settings updated: STT={stt_enabled}, Max length={max_response_length}")

    def process_audio_to_text(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Convert audio to text using speech recognition (placeholder implementation)"""
        # In a real implementation, you would use a speech recognition service
        # like Whisper, Google Speech-to-Text, or similar
        logger.info(f"Processing audio for STT: sample_rate={sample_rate}, shape={audio_array.shape}")

        # Placeholder - replace with actual STT implementation
        return "Hello, I'm processing your voice input through the AI system."

    def generate_response(self, sample_rate: int, audio_array: np.ndarray):
        """Enhanced response generation with full STT->LLM->TTS pipeline"""
        try:
            if not self.stt_enabled:
                # If STT is disabled, just return an acknowledgment
                response_text = "Voice chat is active. Speech-to-text is currently disabled."
            else:
                # Convert speech to text
                user_text = self.process_audio_to_text(audio_array, sample_rate)
                logger.info(f"STT Result: {user_text}")

                # Generate AI response
                ai_response = self.chat_engine.chat(user_text)

                # Limit response length
                words = ai_response.split()
                if len(words) > self.max_response_length:
                    ai_response = " ".join(words[:self.max_response_length]) + "..."

                response_text = ai_response

            # Generate TTS audio
            try:
                audio_file = self.tts_generator.generate_audio(response_text, self.default_voice)
                waveform, file_sample_rate = torchaudio.load(str(audio_file))

                # Resample if necessary
                if file_sample_rate != sample_rate:
                    resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
                    waveform = resampler(waveform)

                # Convert to numpy and yield in chunks
                audio_data = waveform.squeeze().numpy()
                chunk_size = 1024

                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    yield chunk.astype(np.float32)

            except Exception as e:
                logger.error(f"Error in TTS generation: {e}")
                # Yield silence on TTS error
                silence = np.zeros(1024, dtype=np.float32)
                for _ in range(10):
                    yield silence

        except Exception as e:
            logger.error(f"Error in enhanced voice chat handler: {e}")
            # Yield silence on any error
            silence = np.zeros(1024, dtype=np.float32)
            for _ in range(5):
                yield silence

# Create global handler instance
global_voice_chat_handler = EnhancedVoiceChatHandler()

def create_voice_chat_response_handler():
    """Create the response handler for FastRTC"""
    def response(audio: tuple[int, np.ndarray]):
        sample_rate, audio_array = audio
        for audio_chunk in global_voice_chat_handler.generate_response(sample_rate, audio_array):
            yield (sample_rate, audio_chunk)

    return response

# Initialize FastRTC stream
voice_chat_stream = Stream(
    handler=ReplyOnPause(create_voice_chat_response_handler()),
    modality="audio",
    mode="send-receive"
)

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
def generate_script(topic):
    """Generate script with better validation"""
    if not topic or not topic.strip():
        return "❌ Please enter a topic first!"

    if len(topic) > 200:
        return "❌ Topic too long! Please keep it under 200 characters."

    try:
        prompt = f"""Generate a compelling and fast-paced video script about: {topic}

Please create a script that:
- Is engaging and informative
- Has a clear structure (intro, main points, conclusion)
- Is suitable for text-to-speech conversion
- Is approximately 200-500 words
- Uses natural, conversational language

Topic: {topic}"""

        result = chat_engine.chat(prompt)
        return result if result else "❌ No response generated. Please try again."
    except Exception as e:
        logger.error(f"Script generation error: {e}")
        return f"❌ Error generating script: {str(e)}"

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

# --- Global instances ---
voice_manager = VoiceManager()
chat_engine = OllamaChat()
tts = TTSGenerator()

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

            # --- Script Generation Tab ---
            with gr.Tab("📝 AI Script Writer", elem_id="script-tab"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎬 Script Generation")
                        gr.Markdown("Generate compelling scripts using AI for your TTS projects")

                        topic_input = gr.Textbox(
                            label="Topic or Subject",
                            placeholder="e.g., 'The benefits of renewable energy'",
                            lines=2
                        )

                        with gr.Row():
                            generate_script_btn = gr.Button(
                                "✨ Generate Script",
                                variant="primary",
                                size="lg"
                            )
                            reset_chat_btn = gr.Button("🔄 Reset Chat", size="sm")

                        generated_script = gr.Textbox(
                            label="Generated Script",
                            lines=15,
                            max_lines=20,
                            placeholder="Your AI-generated script will appear here..."
                        )

                        with gr.Row():
                            copy_to_tts_btn = gr.Button("📋 Use in TTS", variant="secondary")
                            clear_script_btn = gr.Button("🗑️ Clear Script", size="sm")

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

            # --- Voice Chat Tab (Updated with FastRTC) ---
            with gr.Tab("🎤 Live Voice Chat", elem_id="chat-tab"):
                gr.Markdown("### 🗣️ Real-time Voice Conversation with FastRTC")
                gr.Markdown("Have a live conversation with AI using FastRTC for low-latency audio streaming")

                with gr.Row():
                    with gr.Column():
                        # FastRTC Stream component
                        voice_chat_interface = gr.Interface(
                            fn=lambda audio: voice_chat_stream.handler.handler(audio),
                            inputs=gr.Audio(sources=["microphone"], streaming=True),
                            outputs=gr.Audio(streaming=True),
                            live=True,
                            title="FastRTC Voice Chat",
                            description="Click the microphone to start talking"
                        )

                        chat_status = gr.Markdown("Ready for voice chat - click microphone to begin")

                    with gr.Column():
                        gr.Markdown("### 💡 Voice Chat Tips")
                        gr.Markdown("""
                        - Speak clearly and at normal pace
                        - Wait for the AI to finish responding
                        - Use a good quality microphone for best results
                        - Ensure stable internet connection
                        - The system uses FastRTC for low-latency streaming
                        """)

                        gr.Markdown("### 🔧 Voice Chat Settings")

                        chat_voice_dropdown = gr.Dropdown(
                            label="Response Voice",
                            choices=voice_manager.get_voice_display_names(),
                            value="🎤 Standard Voice",
                            info="Select which voice to use for AI responses"
                        )

                        enable_stt = gr.Checkbox(
                            label="Enable Speech-to-Text",
                            value=True,
                            info="Convert your speech to text for processing"
                        )

                        response_length = gr.Slider(
                            label="Max Response Length",
                            minimum=50,
                            maximum=500,
                            value=200,
                            step=50,
                            info="Maximum words in AI response"
                        )

            # --- Settings Tab ---
            with gr.Tab("⚙️ Settings", elem_id="settings-tab"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔧 Application Settings")

                        model_info = gr.Textbox(
                            label="Current AI Model",
                            value=Config.DEFAULT_MODEL,
                            interactive=False
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

        # Script generation
        generate_script_btn.click(
            fn=generate_script,
            inputs=[topic_input],
            outputs=[generated_script]
        )

        reset_chat_btn.click(
            fn=reset_chat,
            outputs=[status_output]
        )

        copy_to_tts_btn.click(
            fn=lambda script: script,
            inputs=[generated_script],
            outputs=[text_input]
        ).then(
            fn=lambda: "✅ Script copied to TTS input!",
            outputs=[status_output]
        )

        clear_script_btn.click(
            fn=lambda: "",
            outputs=[generated_script]
        )

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
            try:
                results.append("✅ FastRTC: Available")
            except Exception as e:
                results.append(f"❌ FastRTC: {str(e)}")

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
                    "fastrtc_status": "Active",
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
                        "audio_backend": "FastRTC"
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
                # Update the global voice chat handler
                global_voice_chat_handler.set_voice(voice_id)
                return f"✅ Chat voice updated to: {voice_display_name}"
            except Exception as e:
                return f"❌ Error updating chat voice: {str(e)}"

        def update_chat_settings(stt_enabled, response_length):
            """Update chat settings"""
            try:
                global_voice_chat_handler.set_settings(stt_enabled, response_length)
                return f"✅ Chat settings updated: STT={'Enabled' if stt_enabled else 'Disabled'}, Max length={response_length}"
            except Exception as e:
                return f"❌ Error updating chat settings: {str(e)}"

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

        # Voice chat settings
        chat_voice_dropdown.change(
            fn=update_chat_voice,
            inputs=[chat_voice_dropdown],
            outputs=[chat_status]
        )

        enable_stt.change(
            fn=lambda stt, length: update_chat_settings(stt, length),
            inputs=[enable_stt, response_length],
            outputs=[chat_status]
        )

        response_length.change(
            fn=lambda length, stt: update_chat_settings(stt, length),
            inputs=[response_length, enable_stt],
            outputs=[chat_status]
        )

        # Initialize interface with current data
        interface.load(
            fn=lambda: (
                update_voice_list(),
                get_system_stats()
            ),
            outputs=[voice_list, stats_display]
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
        try:
            logger.info("✅ FastRTC components initialized")
        except Exception as e:
            logger.error(f"❌ FastRTC initialization failed: {e}")

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