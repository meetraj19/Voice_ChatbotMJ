# 🎤 AI Voice Chatbot System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()

A production-ready AI voice chatbot system that handles phone calls through Twilio, processes speech with Deepgram, generates intelligent responses using OpenAI, and synthesizes natural-sounding speech with ElevenLabs.

## 🌟 Features

- **📞 Real-time Phone Call Handling** - Seamless integration with Twilio Voice API
- **🎯 Advanced Speech-to-Text** - High-accuracy transcription using Deepgram Nova-2
- **🧠 Intelligent Conversations** - Powered by OpenAI GPT-4 for natural dialogue
- **🗣️ Natural Text-to-Speech** - Human-like voice synthesis with ElevenLabs
- **⚡ Low Latency Processing** - Optimized for real-time conversation flow
- **🔄 WebSocket Streaming** - Bidirectional audio streaming for optimal performance
- **📊 Comprehensive Monitoring** - Built-in health checks and logging
- **🛡️ Production Ready** - Error handling, retry logic, and graceful degradation

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Phone Call    │───▶│   Twilio Voice   │───▶│   FastAPI App   │
│   (Incoming)    │    │   (WebSocket)    │    │   (main.py)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                       ▼                                 ▼                                 ▼
            ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
            │   Deepgram      │              │   OpenAI GPT    │              │   ElevenLabs    │
            │   (Nova-2 STT)  │              │   (Chat API)    │              │   (TTS API)     │
            └─────────────────┘              └─────────────────┘              └─────────────────┘
                       │                                 │                                 │
                       ▼                                 ▼                                 ▼
            ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
            │  Speech → Text  │              │  Text → Reply   │              │  Reply → Speech │
            └─────────────────┘              └─────────────────┘              └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (required for Deepgram SDK 4.1.0)
- **Conda** or **virtualenv**
- **ngrok** (for webhook tunneling)
- API keys for:
  - [Twilio](https://www.twilio.com/) (Phone number with voice capabilities)
  - [OpenAI](https://platform.openai.com/) (GPT API access)
  - [Deepgram](https://deepgram.com/) (Speech-to-Text API)
  - [ElevenLabs](https://elevenlabs.io/) (Text-to-Speech API)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voice-chatbot-system.git
   cd voice-chatbot-system
   ```

2. **Create and activate environment**
   ```bash
   # Using Conda (recommended)
   conda create -n voice-chatbot python=3.10 -y
   conda activate voice-chatbot
   
   # Or using venv
   python -m venv voice-chatbot
   source voice-chatbot/bin/activate  # Linux/Mac
   # voice-chatbot\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   # Install system packages first
   conda install -c conda-forge numpy scipy aiohttp -y
   
   # Install Python packages
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see Configuration section)
   ```

5. **Start ngrok tunnel**
   ```bash
   ngrok http 8000
   # Copy the HTTPS URL to your .env file
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4-turbo-preview

# Deepgram Configuration
DEEPGRAM_API_KEY=your_deepgram_api_key

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
WEBHOOK_BASE_URL=https://your-ngrok-url.ngrok.io

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
```

### Twilio Webhook Setup

1. Go to [Twilio Console](https://console.twilio.com/) → Phone Numbers
2. Select your phone number
3. Configure Voice webhook:
   - **URL**: `https://your-ngrok-url.ngrok.io/voice`
   - **HTTP Method**: POST
4. Optional: Configure status webhook:
   - **URL**: `https://your-ngrok-url.ngrok.io/status`
   - **HTTP Method**: POST

## 📚 API Documentation

### Health Check
```bash
GET /health
```
Returns system status and service configuration.

### Test Endpoints

#### Test Text-to-Speech
```bash
POST /test/tts
Content-Type: application/json

{
  "text": "Hello, this is a test message."
}
```

#### Test Language Model
```bash
POST /test/llm
Content-Type: application/json

{
  "prompt": "Say hello and introduce yourself."
}
```

#### Test Speech-to-Text
```bash
POST /test/stt
```

### Voice Webhook (Twilio)
```bash
POST /voice
```
Handles incoming phone calls from Twilio.

### WebSocket Endpoint
```bash
WS /ws/{call_sid}
```
Handles real-time audio streaming for active calls.

## 🔧 Package Versions

| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.110.0 | Web framework and API server |
| Deepgram SDK | 4.1.0 | Speech-to-Text processing |
| OpenAI | 1.12.0 | Language model integration |
| ElevenLabs | 1.0.1 | Text-to-Speech synthesis |
| Twilio | 9.0.4 | Phone call handling |
| Uvicorn | 0.29.0 | ASGI server |

## 🧪 Testing

### Run All Tests
```bash
# Health check
curl http://localhost:8000/health

# Test individual services
curl -X POST http://localhost:8000/test/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Test message"}'

curl -X POST http://localhost:8000/test/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello"}'

curl -X POST http://localhost:8000/test/stt
```

### Phone Call Test
1. Ensure your server is running
2. Verify ngrok tunnel is active
3. Call your Twilio phone number
4. You should hear: *"Hello! This is your AI assistant. How can I help you today?"*
5. Speak and listen for AI responses

## 📊 Monitoring and Logging

The application provides comprehensive logging:

```bash
# Application logs
INFO - Starting Voice Chatbot API...
INFO - Incoming call CAxxxx from +1234567890
INFO - User said: Hello, how are you?
INFO - Generating speech for: I'm doing great! How can I help you today?
```

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARNING**: Potential issues
- **ERROR**: Error conditions

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Ensure you're in the correct environment
conda activate voice-chatbot
pip install -r requirements.txt --force-reinstall
```

#### 2. Deepgram Connection Issues
```bash
# Check API key and credits
curl -H "Authorization: Token YOUR_DEEPGRAM_API_KEY" \
  https://api.deepgram.com/v1/projects
```

#### 3. Audio Quality Issues
- Verify sample rates: 8kHz (Twilio) ↔ 16kHz (Deepgram)
- Check mulaw ↔ linear PCM conversions
- Monitor audio chunk sizes

#### 4. WebSocket Connection Drops
- Ensure ngrok tunnel is stable
- Verify Twilio webhook configuration
- Check firewall settings

### Debug Mode
Enable detailed debugging:
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

## 🏭 Production Deployment

### Environment Configuration
```env
DEBUG=False
LOG_LEVEL=INFO
WEBHOOK_BASE_URL=https://your-production-domain.com
```

### Recommended Hosting Platforms
- **Railway**: Automatic deployments with built-in HTTPS
- **Heroku**: Easy scaling with dynos
- **AWS/GCP**: Full control and enterprise features
- **DigitalOcean**: Cost-effective VPS options

### Performance Optimization
- Use faster OpenAI models (gpt-3.5-turbo) for lower latency
- Implement connection pooling for HTTP requests
- Add Redis for session management at scale
- Use load balancers for multiple instances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install black flake8 pytest pytest-asyncio

# Format code
black main.py

# Lint code
flake8 main.py

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Twilio](https://www.twilio.com/)** - Voice communication platform
- **[Deepgram](https://deepgram.com/)** - Advanced speech recognition
- **[OpenAI](https://openai.com/)** - Intelligent language processing
- **[ElevenLabs](https://elevenlabs.io/)** - Natural voice synthesis
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework

## 📞 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via [GitHub Issues](https://github.com/yourusername/voice-chatbot-system/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/voice-chatbot-system/discussions)

---

**Built with ❤️ by [Your Name](https://github.com/yourusername)**

*Transform conversations into intelligent interactions.*