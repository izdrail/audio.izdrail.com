# Speech API

A FastAPI application providing Text-to-Speech (TTS) with voice cloning capabilities and Speech-to-Text (STT) transcription services.

## About
This is the api running my agent voice.
This was developed for the project [Laravel Mail](https://laravelmail.com/), 
[Laravel GPT](https://laravelgpt.com/)


## Demo

[https://audio.izdrail.com/](https://audio.izdrail.com/)


## Overview

This API leverages state-of-the-art machine learning models to perform:

1.  **Text-to-Speech (TTS):** Converts text input into spoken audio using standard TTS models (SpeechBrain Tacotron2 + HiFiGAN) or a custom voice cloned from an audio sample (Coqui XTTS).
2.  **Voice Cloning:** Allows users to register a new voice by uploading a short audio sample. This voice can then be used for TTS generation.
3.  **Speech-to-Text (STT):** Transcribes spoken audio from an uploaded file into text using OpenAI's Whisper model.

## Features

* **Standard TTS:** Generate speech using pre-trained models.
* **Voice Cloning TTS:** Zero-shot voice cloning using Coqui XTTS. Provide a sample, get a speaker ID, and generate speech in that voice.
* **Voice Registration:** Simple endpoint to upload a reference audio file and register a new speaker ID.
* **List Voices:** Endpoint to retrieve all currently registered speaker IDs for voice cloning.
* **Speech Transcription:** Accurate transcription of audio files with automatic language detection.
* **Text Preprocessing:** Automatically converts numerals within the input text to words (e.g., "123" becomes "one hundred twenty-three") for more natural TTS output.
* **Audio Processing:** Handles audio concatenation for longer texts and converts output to MP3 format.
* **Temporary File Management:** Uses temporary files for processing and provides an endpoint for cleanup.
* **CORS Enabled:** Allows requests from any origin.
* **GPU Acceleration:** Utilizes CUDA if available for faster model inference, with CPU fallback.

## Technologies Used

* **Framework:** FastAPI
* **Server:** Uvicorn
* **Standard TTS:** SpeechBrain (Tacotron2, HiFiGAN)
* **Voice Cloning TTS:** Coqui TTS (XTTS v2)
* **Speech-to-Text:** OpenAI Whisper (base model)
* **Deep Learning:** PyTorch
* **Audio Manipulation:** Pydub
* **Data Validation:** Pydantic
* **Utilities:** Num2words, NumPy

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    You will need to create a `requirements.txt` file based on the imports in `main.py`. Key libraries include: `fastapi`, `uvicorn[standard]`, `torch`, `torchaudio`, `speechbrain`, `TTS`, `openai-whisper`, `pydub`, `num2words`, `numpy`, `python-multipart`.
    ```bash
    pip install fastapi uvicorn[standard] torch torchaudio speechbrain TTS openai-whisper pydub num2words numpy python-multipart
    # Note: Ensure compatible versions, especially for PyTorch/CUDA if using GPU.
    # You might need ffmpeg installed for pydub: sudo apt update && sudo apt install ffmpeg (or equivalent)
    ```

4.  **Model Downloads:** The first time you run the API, the required TTS and STT models will be downloaded automatically. This might take some time and require significant disk space.

5.  **GPU Support (Optional but Recommended):** For significantly faster inference, ensure you have a CUDA-enabled GPU, the correct NVIDIA drivers, and a PyTorch version compiled with CUDA support. The code automatically detects and uses the GPU if available (`.to("cuda" if torch.cuda.is_available() else "cpu")`).

6.  **Directory Creation:** The API will automatically create a `voice_samples` directory in the project root to store reference audio for cloned voices.

## Running the API

Start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 1602 --reload