# Speech API
This powers my own automation system.
A FastAPI-based REST API for Text-to-Speech (TTS) with voice cloning capabilities and Speech-to-Text (STT) transcription.

## Features

- **Text-to-Speech (TTS)**
    - Standard TTS using Tacotron2 and HiFi-GAN
    - Voice cloning with XTTS v2 for custom voices
    - Automatic text preprocessing (numbers to words)
    - Multi-sentence support with proper concatenation
    - MP3 audio output

- **Voice Cloning**
    - Register custom voices from audio samples
    - Zero-shot voice cloning using XTTS v2
    - Voice sample management and listing
    - Support for multiple speaker voices

- **Speech-to-Text (STT)**
    - Audio transcription using OpenAI Whisper
    - Automatic language detection
    - Support for various audio formats

- **Additional Features**
    - Health check endpoint
    - Temporary file management
    - CORS support for web applications
    - Background model initialization

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- FFmpeg (for audio processing)

### Dependencies

```bash
pip install fastapi uvicorn
pip install torch torchaudio
pip install speechbrain
pip install openai-whisper
pip install TTS
pip install pydub
pip install num2words
pip install python-multipart
```

### Setup

1. Clone the repository and navigate to the project directory
2. Install the required dependencies
3. Create a `voice_samples` directory in the project root
4. Run the application:

```bash
python main.py
```

The API will start on `http://localhost:1602` by default.

## API Endpoints

### Text-to-Speech

#### `POST /tts`

Generate speech from text with optional voice cloning.

**Request Body:**
```json
{
  "text": "Hello, this is a test message.",
  "speaker_id": "john-doe-abc123ef" 
}
```

**Response:** MP3 audio file

**Example:**
```bash
curl -X POST "http://localhost:1602/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "speaker_id": "my-voice-123"}' \
  --output speech.mp3
```

### Voice Management

#### `POST /voice/train`

Register and train a new voice from an audio sample.

**Request:**
- `speaker_name`: Name for the voice (form data)
- `audio_file`: Audio file containing voice sample (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "speaker_id": "john-doe-abc123ef",
  "message": "Voice registered successfully with ID: john-doe-abc123ef"
}
```

**Example:**
```bash
curl -X POST "http://localhost:1602/voice/train" \
  -F "speaker_name=John Doe" \
  -F "audio_file=@voice_sample.wav"
```

#### `GET /voice/list`

List all available voice models.

**Response:**
```json
{
  "voices": [
    {
      "speaker_id": "john-doe-abc123ef",
      "name": "John Doe"
    }
  ],
  "count": 1
}
```

### Speech-to-Text

#### `POST /transcribe`

Transcribe speech from audio file.

**Request:**
- `audio`: Audio file (multipart/form-data)

**Response:**
```json
{
  "language": "en",
  "text": "This is the transcribed text from the audio."
}
```

**Example:**
```bash
curl -X POST "http://localhost:1602/transcribe" \
  -F "audio=@audio_file.wav"
```

### Utility Endpoints

#### `GET /health`

Check if the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

#### `GET /clean`

Clean temporary files.

**Response:**
```json
{
  "status": "ok"
}
```

## Configuration

### Environment Variables

- `PORT`: Server port (default: 1602)
- `COQUI_TOS_AGREED`: Set to "1" to agree to Coqui TTS terms of service

### Audio Requirements

For optimal voice cloning results:
- Audio samples should be clear and noise-free
- Recommended length: 10-30 seconds
- Format: WAV, MP3, or other common audio formats
- Sample rate: 22050 Hz (automatically converted)
- Channels: Mono (automatically converted)

## Models Used

- **Standard TTS**: Tacotron2 + HiFi-GAN from SpeechBrain
- **Voice Cloning**: XTTS v2 from Coqui TTS
- **Speech Recognition**: OpenAI Whisper (base model)

## File Structure

```
project/
├── main.py                 # Main application file
├── voice_samples/          # Directory for voice samples
│   └── {speaker_id}/
│       └── reference.wav   # Reference audio for each speaker
└── README.md              # This file
```

## Performance Notes

- **GPU Acceleration**: The API automatically uses CUDA if available
- **Model Loading**: Models are loaded on startup and kept in memory
- **Temporary Files**: Audio files are temporarily stored and cleaned up automatically
- **Concurrent Requests**: FastAPI handles multiple requests asynchronously

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing or invalid parameters)
- `404`: Speaker ID not found
- `500`: Internal server error (model initialization, processing errors)

## Development

### Running in Development Mode

```bash
python main.py
```

The server will start with auto-reload enabled for development.

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:1602/docs`
- ReDoc: `http://localhost:1602/redoc`

## License

This project uses several open-source libraries with their respective licenses:
- FastAPI: MIT License
- SpeechBrain: Apache 2.0 License
- OpenAI Whisper: MIT License
- Coqui TTS: MPL 2.0 License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU processing
2. **Model loading errors**: Ensure all dependencies are installed correctly
3. **Audio format issues**: Check that FFmpeg is installed and accessible
4. **Voice cloning quality**: Ensure reference audio is clear and of good quality

### Debug Mode

Set logging level to DEBUG for more detailed output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```