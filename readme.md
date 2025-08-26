# Advanced Text-to-Speech Studio

A comprehensive AI-powered Text-to-Speech application with voice cloning, script generation, and real-time voice chat capabilities.

## Features

### Core TTS Functionality
- **Standard TTS**: High-quality speech synthesis using Tacotron2 and HiFiGAN models
- **Voice Cloning**: Create custom voices from audio samples using XTTS v2
- **Multiple Audio Formats**: Support for WAV, MP3, FLAC, M4A, and OGG
- **Text Preprocessing**: Automatic number-to-words conversion and text cleaning

### AI Script Generation
- **Intelligent Script Writing**: Generate compelling scripts using Ollama-powered LLMs
- **Topic-based Generation**: Create content from simple topic descriptions
- **Conversation History**: Maintain chat context for iterative improvements
- **Direct Integration**: Copy generated scripts directly to TTS input

### Voice Management
- **Voice Upload**: Add custom voices with audio samples (5-120 seconds)
- **Voice Validation**: Automatic audio quality and format checking
- **Voice Library**: Manage multiple voices with metadata and descriptions
- **Real-time Refresh**: Dynamic voice list updates

### Live Voice Chat
- **Real-time Conversation**: Live voice chat using FastRTC streaming
- **Speech-to-Text**: Convert voice input to text for AI processing
- **Configurable Responses**: Adjustable response length and voice selection
- **Low-latency Streaming**: Optimized for responsive voice interactions

### Advanced Interface
- **Multi-tab Interface**: Organized workflow with dedicated sections
- **Progress Tracking**: Real-time status updates and error handling
- **Statistics Dashboard**: System performance and usage metrics
- **Configuration Management**: Export/import settings and voice configurations

## Requirements

### System Dependencies
```bash
# Core Python packages
pip install torch torchaudio
pip install speechbrain
pip install TTS
pip install gradio
pip install requests
pip install numpy
pip install pathlib
pip install num2words

# FastRTC for real-time audio
pip install fastrtc
pip install sphn

# Optional dependencies
pip install logging
pip install tempfile
pip install uuid
pip install datetime
pip install typing
```

### External Services
- **Ollama Server**: Required for AI script generation and chat functionality
- **Audio Processing**: torchaudio for audio manipulation and format conversion

## Installation

### Option 1: Docker Compose (Recommended)

The easiest way to run the TTS Studio is using Docker Compose, which automatically sets up all dependencies including Ollama.

1. **Clone the Repository**
```bash
git clone <repository-url>
cd tts-studio
```

2. **Configure Environment Variables**
   Create a `.env` file:
```env
WWWUSER=1000
WWWGROUP=1001
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=deepseek-r1:7b
```

3. **Launch with Docker Compose**
```bash
docker-compose up -d
```

The application will be available at:
- TTS Studio: `http://localhost:1602`
- Ollama API: `http://localhost:11434`

4. **Initialize the AI Model**
```bash
# Wait for Ollama to start, then pull the model
docker-compose exec ollama ollama pull deepseek-r1:7b
```

### Option 2: Manual Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd tts-studio
```

2. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Up Ollama**
```bash
# Install Ollama (visit https://ollama.ai)
ollama pull deepseek-r1:7b  # or your preferred model
```

4. **Configure Environment**
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="deepseek-r1:7b"
```

5. **Create Required Directories**
```bash
mkdir -p voice_samples generated_audio
```

## Usage

### Starting the Application
```bash
python main.py
```

The interface will launch at `http://localhost:1602`

### Quick TTS
1. Enter text in the input field
2. Select a voice from the dropdown
3. Click "Generate Audio" to create speech
4. Download or play the generated audio

### Voice Cloning
1. Navigate to the "Voice Cloning" tab
2. Upload a clear audio sample (5-120 seconds)
3. Provide a name for the voice
4. Click "Upload Voice" to add it to your library
5. The new voice will appear in the voice selection dropdown

### AI Script Generation
1. Go to the "AI Script Writer" tab
2. Enter a topic or subject
3. Click "Generate Script" for AI-powered content
4. Edit the generated script as needed
5. Use "Copy to TTS" to transfer the script for audio generation

### Live Voice Chat
1. Open the "Live Voice Chat" tab
2. Configure response voice and settings
3. Click the microphone to start talking
4. The AI will respond with synthesized speech
5. Adjust STT and response length settings as needed

## Configuration

### Environment Variables
- `OLLAMA_BASE_URL`: Ollama server endpoint (default: http://ollama:11434)
- `OLLAMA_MODEL`: AI model for script generation (default: deepseek-r1:7b)

### Audio Settings
- **Sample Rate**: 22050 Hz (configurable)
- **Max Audio Length**: 120 seconds for voice samples
- **Supported Formats**: WAV, MP3, FLAC, M4A, OGG
- **Max File Size**: 50MB for uploads

### Voice Configuration
Voice settings are stored in `voice_config.json`:
```json
{
  "voice_id": {
    "name": "Voice Display Name",
    "description": "Voice description",
    "type": "cloned",
    "enabled": true,
    "created_at": "2024-01-01T00:00:00",
    "metadata": {}
  }
}
```

## File Structure

```
tts-studio/
├── main.py                 # Main application file
├── voice_config.json       # Voice configuration
├── voice_samples/          # Uploaded voice samples
│   └── voice_id/
│       └── reference.wav
├── generated_audio/        # Generated TTS outputs
├── tmp_tts/               # Temporary TTS model files
└── tmp_vocoder/           # Temporary vocoder files
```

## API Integration

### Ollama Integration
The application connects to Ollama for AI-powered features:
- Script generation using LLM models
- Conversation management
- Real-time chat responses

### TTS Models
- **Standard Voice**: Tacotron2 + HiFiGAN (SpeechBrain)
- **Voice Cloning**: XTTS v2 (Coqui TTS)
- **Audio Processing**: PyTorch Audio

### FastRTC Streaming
Real-time audio streaming for voice chat:
- Low-latency audio transmission
- Bidirectional communication
- Configurable audio parameters

## Troubleshooting

### Common Issues

**Ollama Connection Failed**
- Verify Ollama is running: `ollama list`
- Check the base URL configuration
- Ensure the specified model is available

**Voice Cloning Errors**
- Verify audio sample quality (clear, noise-free)
- Check file format compatibility
- Ensure minimum duration requirements (5 seconds)

**Audio Generation Failed**
- Check text length (max 1000 characters)
- Verify voice file existence
- Review system resources and disk space

**FastRTC Streaming Issues**
- Check microphone permissions
- Verify network connectivity
- Ensure audio device compatibility

### Performance Optimization

**Memory Usage**
- Models are cached after first load
- Generated audio files are automatically cleaned
- Voice validation runs periodically

**Processing Speed**
- GPU acceleration (if available)
- Concurrent audio processing
- Optimized text preprocessing

## Development

### Architecture
- **VoiceManager**: Handles voice configuration and file management
- **OllamaChat**: Manages AI chat and script generation
- **TTSGenerator**: Coordinates TTS model execution
- **EnhancedVoiceChatHandler**: Real-time voice chat processing

### Adding New Features
1. Extend the appropriate class (VoiceManager, TTSGenerator, etc.)
2. Add UI components in `create_interface()`
3. Wire event handlers for new functionality
4. Update configuration schema if needed

### Testing
- Voice sample validation
- Audio format compatibility
- Ollama model availability
- FastRTC streaming functionality

## License

This project uses various open-source components:
- SpeechBrain (Apache 2.0)
- Coqui TTS (MPL 2.0)
- Gradio (Apache 2.0)
- FastRTC (MIT)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Review system logs for detailed error messages
4. Ensure external services (Ollama) are properly configured

## Contributing

Contributions welcome for:
- Additional TTS model support
- New voice processing features
- UI/UX improvements
- Performance optimizations
- Documentation updates