import os
import re
import tempfile
import uuid
import shutil
from typing import Dict, Any, Optional, List
import torch
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from num2words import num2words
from pydub import AudioSegment
import torchaudio
from speechbrain.pretrained import HIFIGAN, Tacotron2
import whisper
from fastapi.middleware.cors import CORSMiddleware

# Import TTS libraries for voice cloning
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# FastAPI app
app = FastAPI(title="Speech API", description="API for Text-to-Speech with Voice Cloning and Speech-to-Text")
# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
# TODO - Simplify and improve code


# File prefixes and suffixes
speech_tts_prefix = "speech-tts-"
wav_suffix = ".wav"
opus_suffix = ".mp3"
voice_samples_dir = "voice_samples"

# Ensure voice samples directory exists
os.makedirs(voice_samples_dir, exist_ok=True)

# Load transcription model
whisper_model = whisper.load_model("base")

# Define global variable for voice models
voice_model = None
voice_model_speaker_ids = {}

# Define request models
class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None

class TrainVoiceRequest(BaseModel):
    speaker_name: str

# Clean temporary files
def clean_tmp():
    tmp_dir = tempfile.gettempdir()
    for file in os.listdir(tmp_dir):
        if file.startswith(speech_tts_prefix):
            os.remove(os.path.join(tmp_dir, file))
    print("[Speech REST API] Temporary files cleaned!")

# Preprocess text to replace numerals with words
def preprocess_text(text: str) -> str:
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    return text

# Initialize voice cloning model
def initialize_voice_model():
    global voice_model
    
    print("[Speech REST API] Initializing voice model...")
    
    try:
        # Initialize XTTS for voice cloning (zero-shot approach)
        voice_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
        print("[Speech REST API] Voice model initialized successfully!")
    except Exception as e:
        print(f"[Speech REST API] Error initializing voice model: {str(e)}")
        voice_model = None

# Run voice clone TTS with custom voice and save file
def run_voice_clone_tts_and_save_file(text: str, speaker_id: str) -> str:
    global voice_model
    
    # If model not initialized, initialize it
    if voice_model is None:
        initialize_voice_model()
        if voice_model is None:
            raise HTTPException(status_code=500, detail="Voice model initialization failed")
    
    # Get reference audio path
    reference_audio = os.path.join(voice_samples_dir, speaker_id, "reference.wav")
    if not os.path.exists(reference_audio):
        raise HTTPException(status_code=404, detail=f"Speaker ID '{speaker_id}' not found")
    
    # Generate speech with cloned voice
    tmp_dir = tempfile.gettempdir()
    tmp_path_wav = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + wav_suffix)
    
    try:
        # Generate speech with XTTS
        voice_model.tts_to_file(
            text=text,
            file_path=tmp_path_wav,
            speaker_wav=reference_audio,
            language="en"
        )
        return tmp_path_wav
    except Exception as e:
        print(f"[Speech REST API] Error generating cloned voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating speech with cloned voice: {str(e)}")

# Run standard TTS and save file
def run_standard_tts_and_save_file(text: str) -> str:
    # Load standard TTS model if needed
    global tacotron2, hifi_gan
    if 'tacotron2' not in globals():
        tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
    
    # Running the TTS
    mel_outputs, mel_length, alignment = tacotron2.encode_batch([text])
    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_outputs)
    # Get temporary directory
    tmp_dir = tempfile.gettempdir()
    # Save wav to temporary file
    tmp_path_wav = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + wav_suffix)
    torchaudio.save(tmp_path_wav, waveforms.squeeze(1), 22050)
    return tmp_path_wav

# Register and train a new voice
@app.post("/voice/train", description="Register and train a new voice from sample audio")
async def train_voice(speaker_name: str = Body(...), audio_file: UploadFile = File(...)):
    if not audio_file:
        raise HTTPException(status_code=400, detail="Audio file is required")
    
    # Create unique speaker ID
    speaker_id = f"{speaker_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
    
    # Create directory for speaker
    speaker_dir = os.path.join(voice_samples_dir, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    
    # Save reference audio
    reference_path = os.path.join(speaker_dir, "reference.wav")
    
    # Save uploaded audio
    with open(reference_path, "wb") as buffer:
        content = await audio_file.read()
        buffer.write(content)
    
    # Convert to proper format if needed
    try:
        # Load and normalize the audio
        audio = AudioSegment.from_file(reference_path)
        
        # Convert to proper format for TTS (wav, 22050Hz, mono)
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(reference_path, format="wav")
        
        # Initialize voice model if not already initialized
        if voice_model is None:
            initialize_voice_model()
        
        return {
            "success": True,
            "speaker_id": speaker_id,
            "message": f"Voice registered successfully with ID: {speaker_id}"
        }
    except Exception as e:
        # Clean up on failure
        if os.path.exists(speaker_dir):
            shutil.rmtree(speaker_dir)
        raise HTTPException(status_code=500, detail=f"Error processing voice sample: {str(e)}")

# List available voices
@app.get("/voice/list", description="List all available voice models")
async def list_voices():
    voices = []
    
    if os.path.exists(voice_samples_dir):
        for speaker_id in os.listdir(voice_samples_dir):
            speaker_dir = os.path.join(voice_samples_dir, speaker_id)
            if os.path.isdir(speaker_dir) and os.path.exists(os.path.join(speaker_dir, "reference.wav")):
                voices.append({
                    "speaker_id": speaker_id,
                    "name": speaker_id.split('-')[0].replace('-', ' ').title()
                })
    
    return {
        "voices": voices,
        "count": len(voices)
    }

# TTS endpoint with voice cloning support
@app.post("/tts", response_class=FileResponse, description="Generate speech from text with optional voice cloning")
async def generate_tts(request: TTSRequest):
    # Sentences to generate
    text = request.text
    
    # Remove ' and " from text
    text = text.replace("'", "")
    text = text.replace('"', "")
    
    # Preprocess text to replace numerals with words
    text = preprocess_text(text)
    
    # Split text by . ? !
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    
    # Trim sentences
    sentences = [sentence.strip() for sentence in sentences]
    
    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]
    
    # Logging
    print(f"[Speech REST API] Got request: length ({len(text)}), sentences ({len(sentences)})")
    
    # Run TTS for each sentence
    output_files = []
    
    if request.speaker_id:
        # Use voice cloning
        print(f"[Speech REST API] Using cloned voice: {request.speaker_id}")
        
        # For voice cloning, we'll pass the whole text at once since XTTS
        # handles sentence breaks internally better
        tmp_path_wav = run_voice_clone_tts_and_save_file(text, request.speaker_id)
        output_files.append(tmp_path_wav)
    else:
        # Use standard TTS
        for sentence in sentences:
            print(f"[Speech REST API] Generating TTS: {sentence}")
            tmp_path_wav = run_standard_tts_and_save_file(sentence)
            output_files.append(tmp_path_wav)
    
    # Concatenate all files
    audio = AudioSegment.empty()
    for file in output_files:
        audio += AudioSegment.from_wav(file)
    
    # Save audio to file
    tmp_dir = tempfile.gettempdir()
    tmp_path_opus = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + opus_suffix)
    audio.export(tmp_path_opus, format="mp3")
    
    # Delete tmp files
    for file in output_files:
        os.remove(file)
    
    # Send file response
    return FileResponse(
        path=tmp_path_opus, 
        media_type='audio/mpeg; codecs=mp3',
        filename=f"speech-{uuid.uuid4()}.mp3"
    )

# Transcribe endpoint
@app.post("/transcribe", description="Transcribe speech from audio")
async def transcribe(audio: UploadFile = File(...)):
    if not audio:
        raise HTTPException(status_code=400, detail="Invalid input, form-data: audio")
    
    # Save audio file into tmp folder
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, str(uuid.uuid4()))
    
    # Save uploaded file
    with open(tmp_path, "wb") as buffer:
        content = await audio.read()
        buffer.write(content)
    
    # Load audio and pad/trim it to fit 30 seconds
    audio_data = whisper.load_audio(tmp_path)
    audio_data = whisper.pad_or_trim(audio_data)
    
    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio_data).to(whisper_model.device)
    
    # Detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    language = max(probs, key=probs.get)
    
    # Decode the audio
    result = whisper.transcribe(whisper_model, tmp_path)
    text_result = result["text"]
    text_result_trim = text_result.strip()
    
    # Delete tmp file
    os.remove(tmp_path)
    
    return {
        'language': language,
        'text': text_result_trim
    }

# Health endpoint
@app.get("/health", description="Check if the API is running")
async def health():
    return {"status": "ok"}

# Clean endpoint
@app.get("/clean", description="Clean temporary files")
async def clean():
    clean_tmp()
    return {"status": "ok"}

# Initialize voice model on startup
@app.on_event("startup")
async def startup_event():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(initialize_voice_model)

# Entry point
if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 1602))
    # Start server
    print(f"[Speech REST API] Starting server on port {port}")
    uvicorn.run("main:app", host='0.0.0.0', port=port, reload=True)