FROM ubuntu:22.04

# Set working directory
WORKDIR /app

ENV API_URL=http://localhost:1602

# Install all system dependencies in a single layer
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    gcc \
    g++ \
    gnupg \
    make \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    espeak-ng \
    libsndfile1-dev \
    supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 --version

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt && \
    python3 -c "import gradio; print(f'Gradio version: {gradio.__version__}')" && \
    python3 -c "import speechbrain; from speechbrain.inference.TTS import Tacotron2; print('SpeechBrain inference verified')" && \
    python3 -c "from fastrtc import Stream, get_stt_model; print('FastRTC imports verified')"

# Pre-download the XTTS model and Silero VAD model during build
RUN python3 -c "import re; path='/usr/local/lib/python3.10/dist-packages/TTS/tts/layers/xtts/stream_generator.py'; c=open(path).read(); c=re.sub(r'from transformers import \(.*?\)', lambda m: m.group(0).replace('BeamSearchScorer,', '').replace('BeamSearchScorer', ''), c, flags=re.DOTALL); open(path, 'w').write('from transformers.generation import BeamSearchScorer\n' + c)" && \
    python3 -c "import os; os.environ['COQUI_TOS_AGREED'] = '1'; from TTS.api import TTS; import torch; print('Pre-downloading XTTS model...'); tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2'); print('XTTS model downloaded successfully!')" && \
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='freddyaboulton/silero-vad', filename='silero_vad.onnx')"

# Copy Supervisor config
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the port
EXPOSE 1602

# Copy application code
COPY . .

# Run Supervisor as the entry point
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]