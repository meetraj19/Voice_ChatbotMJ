#!/usr/bin/env python3
"""
Voice Chatbot System
"""

import asyncio
import json
import logging
import os
import sys
import time
import base64
import audioop
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, AsyncGenerator, List
from dotenv import load_dotenv

import numpy as np
from fastapi import FastAPI, Request, Response, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import uvicorn
from twilio.twiml.voice_response import VoiceResponse, Start, Stream
from twilio.rest import Client
from twilio.request_validator import RequestValidator

# Updated imports for Deepgram SDK 4.1.0
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    PrerecordedOptions
)

import aiohttp
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
import openai

# ==============================================================================
# CONFIGURATION MODULE
# ==============================================================================

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Application configuration"""
    
    # Twilio
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
    
    # Deepgram
    DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
    
    # ElevenLabs
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')
    
    # Server
    SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
    SERVER_PORT = int(os.getenv('SERVER_PORT', 8000))
    WEBHOOK_BASE_URL = os.getenv('WEBHOOK_BASE_URL', 'http://localhost:8000')
    
    # Application
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Audio settings
    SAMPLE_RATE = 8000  # Twilio uses 8kHz
    CHUNK_SIZE = 160  # 20ms chunks at 8kHz
    AUDIO_FORMAT = 'mulaw'  # Twilio audio format
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_vars = [
            'TWILIO_ACCOUNT_SID',
            'TWILIO_AUTH_TOKEN',
            'TWILIO_PHONE_NUMBER',
            'OPENAI_API_KEY',
            'DEEPGRAM_API_KEY',
            'ELEVENLABS_API_KEY'
        ]
        
        missing = []
        for var in required_vars:
            if not getattr(cls, var):
                missing.append(var)
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ==============================================================================
# AUDIO PROCESSOR MODULE
# ==============================================================================

class AudioProcessor:
    """Handle audio format conversions and processing"""
    
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        self.channels = 1  # Mono
        self.sample_width = 2  # 16-bit
        
    def decode_mulaw(self, mulaw_data: bytes) -> bytes:
        """Convert mulaw encoded audio to linear PCM"""
        try:
            # Decode mulaw to linear PCM
            linear_data = audioop.ulaw2lin(mulaw_data, self.sample_width)
            return linear_data
        except Exception as e:
            logger.error(f"Error decoding mulaw: {e}")
            return b""
    
    def encode_mulaw(self, linear_data: bytes) -> bytes:
        """Convert linear PCM to mulaw encoding"""
        try:
            # Encode linear PCM to mulaw
            mulaw_data = audioop.lin2ulaw(linear_data, self.sample_width)
            return mulaw_data
        except Exception as e:
            logger.error(f"Error encoding mulaw: {e}")
            return b""
    
    def base64_to_audio(self, base64_audio: str) -> bytes:
        """Convert base64 encoded audio to bytes"""
        try:
            return base64.b64decode(base64_audio)
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            return b""
    
    def audio_to_base64(self, audio_data: bytes) -> str:
        """Convert audio bytes to base64 encoding"""
        try:
            return base64.b64encode(audio_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding base64: {e}")
            return ""
    
    def resample_audio(self, audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio data to different sample rate"""
        if from_rate == to_rate:
            return audio_data
        
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate new length
            new_length = int(len(audio_array) * to_rate / from_rate)
            
            # Resample using linear interpolation
            resampled = np.interp(
                np.linspace(0, len(audio_array) - 1, new_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
            
            return resampled.tobytes()
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data
    
    def convert_to_deepgram_format(self, twilio_audio: bytes) -> bytes:
        """Convert Twilio mulaw audio to format suitable for Deepgram"""
        # Deepgram accepts various formats, but linear PCM works well
        linear_audio = self.decode_mulaw(twilio_audio)
        
        # Deepgram typically expects 16kHz, so resample if needed
        if self.sample_rate != 16000:
            linear_audio = self.resample_audio(linear_audio, self.sample_rate, 16000)
        
        return linear_audio
    
    def convert_from_elevenlabs_format(self, elevenlabs_audio: bytes, output_format: str = 'mulaw') -> bytes:
        """Convert ElevenLabs audio output to Twilio format"""
        # ElevenLabs typically outputs MP3 or PCM
        # For simplicity, we'll assume PCM input
        
        if output_format == 'mulaw':
            # Resample to 8kHz if needed
            if self.sample_rate != 8000:
                elevenlabs_audio = self.resample_audio(elevenlabs_audio, 16000, 8000)
            
            # Convert to mulaw
            return self.encode_mulaw(elevenlabs_audio)
        
        return elevenlabs_audio
    
    def create_silence(self, duration_ms: int) -> bytes:
        """Create silence audio data"""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        silence = np.zeros(num_samples, dtype=np.int16)
        
        if self.sample_rate == 8000:
            # Convert to mulaw for Twilio
            return self.encode_mulaw(silence.tobytes())
        
        return silence.tobytes()

# ==============================================================================
# STT HANDLER MODULE
# ==============================================================================

class STTHandler:
    """Handle Speech-to-Text using Deepgram SDK 4.1.0"""
    
    def __init__(self, on_transcript: Callable):
        # Initialize Deepgram client with SDK 4.1.0
        self.deepgram = DeepgramClient(Config.DEEPGRAM_API_KEY)
        self.on_transcript = on_transcript
        self.dg_connection = None
        self._running = False
        
    async def start(self):
        """Start the STT service"""
        try:
            logger.info("Starting Deepgram STT connection...")
            
            # Configure options for SDK 4.1.0
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True,
                endpointing=300
            )
            
            # Create connection using SDK 4.1.0 API
            self.dg_connection = self.deepgram.listen.websocket.v("1")
            
            # Set up event handlers
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.dg_connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            # Start the connection (SDK 4.1.0 - not async)
            try:
                start_result = self.dg_connection.start(options)
                if start_result is False:
                    logger.error("Failed to start Deepgram connection")
                    return False
            except Exception as e:
                logger.error(f"Error starting Deepgram connection: {e}")
                return False
                
            self._running = True
            logger.info("Deepgram STT connection started successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start STT: {e}")
            raise
    
    async def stop(self):
        """Stop the STT service"""
        self._running = False
        if self.dg_connection:
            try:
                # SDK 4.1.0 - finish() might be synchronous
                if hasattr(self.dg_connection, 'finish'):
                    if asyncio.iscoroutinefunction(self.dg_connection.finish):
                        await self.dg_connection.finish()
                    else:
                        self.dg_connection.finish()
                logger.info("Deepgram STT connection closed")
            except Exception as e:
                logger.error(f"Error closing STT connection: {e}")
    
    async def process_audio(self, audio_data: bytes):
        """Process audio data for transcription"""
        if not self._running or not self.dg_connection:
            logger.warning("STT not running, skipping audio")
            return
        
        try:
            # Send audio to Deepgram (SDK 4.1.0 - might be synchronous)
            if hasattr(self.dg_connection, 'send'):
                if asyncio.iscoroutinefunction(self.dg_connection.send):
                    await self.dg_connection.send(audio_data)
                else:
                    self.dg_connection.send(audio_data)
        except Exception as e:
            logger.error(f"Error sending audio to STT: {e}")
    
    def _on_message(self, *args, **kwargs):
        """Handle incoming messages from Deepgram"""
        try:
            result = kwargs.get('result')
            if not result:
                return

            is_final = result.get('is_final', False)
            
            # Get transcript from the result (SDK 4.1.0 format)
            channel = result.get('channel', {})
            alternatives = channel.get('alternatives', [])
            if alternatives:
                transcript = alternatives[0].get('transcript', '')
                if transcript:
                    # Create async task to handle transcript
                    asyncio.create_task(self.on_transcript(transcript, is_final))
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
    
    def _on_error(self, *args, **kwargs):
        """Handle errors from Deepgram"""
        error = kwargs.get("error", "Unknown error")
        logger.error(f"Deepgram error: {error}")
    
    def _on_close(self, *args, **kwargs):
        """Handle connection close"""
        logger.info("Deepgram connection closed")
        self._running = False

class TranscriptBuffer:
    """Buffer and manage transcripts"""
    
    def __init__(self):
        self.buffer = []
        self.final_transcript = ""
        self.lock = asyncio.Lock()
    
    async def add_transcript(self, text: str, is_final: bool):
        """Add transcript to buffer"""
        async with self.lock:
            if is_final:
                self.final_transcript += text + " "
                self.buffer.clear()
            else:
                # Update interim buffer
                if self.buffer:
                    self.buffer[-1] = text
                else:
                    self.buffer.append(text)
    
    async def get_current_text(self) -> str:
        """Get the current complete text"""
        async with self.lock:
            interim = " ".join(self.buffer) if self.buffer else ""
            return self.final_transcript + interim
    
    async def clear(self):
        """Clear all buffers"""
        async with self.lock:
            self.buffer.clear()
            self.final_transcript = ""

# ==============================================================================
# TTS HANDLER MODULE
# ==============================================================================

class TTSHandler:
    """Handle Text-to-Speech using ElevenLabs 1.0.1"""
    
    def __init__(self):
        # Initialize ElevenLabs client with version 1.0.1
        self.client = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)
        self.voice_id = Config.ELEVENLABS_VOICE_ID
        self.model = "eleven_monolingual_v1"
        
        # Voice settings for ElevenLabs 1.0.1
        self.voice_settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True
        )
    
    async def generate_speech(self, text: str) -> bytes:
        """Generate speech from text and return complete audio"""
        try:
            logger.info(f"Generating speech for: {text[:50]}...")
            
            # Generate audio using ElevenLabs 1.0.1 API
            audio_data = b""
            
            # Use the synchronous client in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def generate():
                # Updated for ElevenLabs 1.0.1
                return self.client.generate(
                    text=text,
                    voice=Voice(
                        voice_id=self.voice_id,
                        settings=self.voice_settings
                    ),
                    model=self.model
                )
            
            # Run in executor to not block the event loop
            audio_generator = await loop.run_in_executor(None, generate)
            
            # Collect all audio chunks
            for chunk in audio_generator:
                audio_data += chunk
            
            logger.info(f"Generated {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return b""
    
    async def generate_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate speech from text as a stream"""
        try:
            logger.info(f"Streaming speech for: {text[:50]}...")
            
            # Use the ElevenLabs streaming API with 1.0.1
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": Config.ELEVENLABS_API_KEY
            }
            
            data = {
                "text": text,
                "model_id": self.model,
                "voice_settings": {
                    "stability": self.voice_settings.stability,
                    "similarity_boost": self.voice_settings.similarity_boost,
                    "style": self.voice_settings.style,
                    "use_speaker_boost": self.voice_settings.use_speaker_boost
                },
                "optimize_streaming_latency": 3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"ElevenLabs API error: {error_text}")
                        return
                    
                    # Stream the audio chunks
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            yield chunk
            
        except Exception as e:
            logger.error(f"Error in speech stream: {e}")
            yield b""
    
    async def preprocess_text(self, text: str) -> str:
        """Preprocess text for better TTS output"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Add pauses for better speech flow
        text = text.replace("...", "... ")
        text = text.replace(".", ". ")
        text = text.replace("?", "? ")
        text = text.replace("!", "! ")
        
        # Remove any characters that might cause issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text.strip()

class AudioStreamManager:
    """Manage audio streaming to Twilio"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_streaming = False
        
    async def add_audio(self, audio_data: bytes):
        """Add audio data to the streaming queue"""
        await self.queue.put(audio_data)
    
    async def get_audio(self) -> bytes:
        """Get next audio chunk from queue"""
        return await self.queue.get()
    
    def has_audio(self) -> bool:
        """Check if there's audio in the queue"""
        return not self.queue.empty()
    
    async def clear(self):
        """Clear the audio queue"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

# ==============================================================================
# LLM HANDLER MODULE
# ==============================================================================

class SimpleLLMHandler:
    """LLM handler using OpenAI 1.12.0"""
    
    def __init__(self):
        # Initialize OpenAI client with version 1.12.0
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant on a phone call. Keep responses concise and conversational."}
        ]
    
    async def get_response(self, user_input: str) -> str:
        """Get response from OpenAI"""
        try:
            self.messages.append({"role": "user", "content": user_input})
            
            # Make async call to OpenAI using 1.12.0 API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=Config.OPENAI_MODEL,
                messages=self.messages,
                temperature=0.7,
                max_tokens=150
            )
            
            assistant_message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {e}")
            return "I'm sorry, I'm having trouble understanding. Could you please repeat that?"

# ==============================================================================
# TWILIO HANDLER MODULE
# ==============================================================================

class TwilioHandler:
    """Handle Twilio voice calls and WebSocket streams using Twilio 9.0.4"""
    
    def __init__(self):
        self.twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
        self.validator = RequestValidator(Config.TWILIO_AUTH_TOKEN)
        self.active_calls = {}
        
    def create_voice_response(self, call_sid: str) -> str:
        """Create TwiML response for incoming call"""
        response = VoiceResponse()
        
        # Start the stream
        start = Start()
        
        # Updated for Twilio 9.0.4 - ensure proper WebSocket URL format
        ws_url = Config.WEBHOOK_BASE_URL.replace("https://", "").replace("http://", "")
        stream = start.stream(url=f'wss://{ws_url}/ws/{call_sid}')
        
        response.append(start)
        
        # Add a pause to keep the call open
        response.pause(length=3600)  # Keep call open for up to 1 hour
        
        return str(response)
    
    def validate_request(self, url: str, params: Dict[str, str], signature: str) -> bool:
        """Validate Twilio request signature"""
        return self.validator.validate(url, params, signature)
    
    async def handle_incoming_call(self, request_data: Dict[str, Any]) -> str:
        """Handle incoming call webhook"""
        call_sid = request_data.get('CallSid')
        from_number = request_data.get('From')
        to_number = request_data.get('To')
        
        logger.info(f"Incoming call {call_sid} from {from_number} to {to_number}")
        
        # Store call information
        self.active_calls[call_sid] = {
            'from': from_number,
            'to': to_number,
            'status': 'connecting',
            'timestamp': datetime.now().isoformat()
        }
        
        # Return TwiML response
        return self.create_voice_response(call_sid)
    
    async def handle_call_status(self, request_data: Dict[str, Any]):
        """Handle call status updates"""
        call_sid = request_data.get('CallSid')
        call_status = request_data.get('CallStatus')
        
        logger.info(f"Call {call_sid} status: {call_status}")
        
        if call_sid in self.active_calls:
            self.active_calls[call_sid]['status'] = call_status
            
            if call_status in ['completed', 'failed', 'busy', 'no-answer']:
                # Clean up call
                del self.active_calls[call_sid]

class CallSession:
    """Manage a single call session"""
    
    def __init__(self, call_sid: str, websocket: WebSocket):
        self.call_sid = call_sid
        self.websocket = websocket
        self.stream_sid = None
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.transcript_buffer = TranscriptBuffer()
        self.audio_stream_manager = AudioStreamManager()
        
        # Handlers
        self.stt_handler = STTHandler(self.on_transcript)
        self.tts_handler = TTSHandler()
        self.llm_handler = SimpleLLMHandler()
        
        # State
        self.is_active = False
        self.tasks = []
        
        # Audio buffers
        self.inbound_audio_buffer = b""
        self.outbound_audio_buffer = b""
        
    async def start(self):
        """Start the call session"""
        try:
            self.is_active = True
            
            # Start STT handler
            await self.stt_handler.start()
            
            # Start background tasks
            self.tasks.append(asyncio.create_task(self.process_transcripts()))
            self.tasks.append(asyncio.create_task(self.stream_audio()))
            
            logger.info(f"Call session {self.call_sid} started")
            
            # Send initial greeting after a short delay
            await asyncio.sleep(1)
            await self.generate_and_send_speech("Hello! This is your AI assistant. How can I help you today?")
            
        except Exception as e:
            logger.error(f"Error starting call session: {e}")
            raise
    
    async def stop(self):
        """Stop the call session"""
        self.is_active = False
        
        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop STT handler
        await self.stt_handler.stop()
        
        logger.info(f"Call session {self.call_sid} stopped")
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle WebSocket message from Twilio"""
        event = message.get('event')
        
        if event == 'start':
            # Stream started
            self.stream_sid = message['start']['streamSid']
            logger.info(f"Stream started: {self.stream_sid}")
            
        elif event == 'media':
            # Audio data received
            payload = message['media']['payload']
            audio_data = self.audio_processor.base64_to_audio(payload)
            
            # Convert from mulaw to linear PCM for STT
            linear_audio = self.audio_processor.convert_to_deepgram_format(audio_data)
            
            # Send to STT
            await self.stt_handler.process_audio(linear_audio)
            
        elif event == 'stop':
            # Stream stopped
            logger.info(f"Stream stopped: {self.stream_sid}")
            await self.stop()
    
    async def on_transcript(self, text: str, is_final: bool):
        """Handle transcript from STT"""
        await self.transcript_buffer.add_transcript(text, is_final)
        
        if is_final and text.strip():
            logger.info(f"User said: {text}")
    
    async def process_transcripts(self):
        """Process transcripts and generate responses"""
        last_transcript = ""
        silence_duration = 0
        
        while self.is_active:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms
                
                current_transcript = await self.transcript_buffer.get_current_text()
                
                if current_transcript != last_transcript:
                    # New content detected
                    last_transcript = current_transcript
                    silence_duration = 0
                else:
                    # No new content, increment silence duration
                    silence_duration += 0.5
                
                # If we have content and sufficient silence, process it
                if current_transcript and silence_duration >= 1.5:  # 1.5 seconds of silence
                    # Extract the unprocessed part
                    user_input = current_transcript.strip()
                    
                    if user_input:
                        logger.info(f"Processing user input: {user_input}")
                        
                        # Clear the buffer
                        await self.transcript_buffer.clear()
                        last_transcript = ""
                        silence_duration = 0
                        
                        # Get LLM response
                        response = await self.llm_handler.get_response(user_input)
                        
                        # Generate and send speech
                        await self.generate_and_send_speech(response)
                
            except Exception as e:
                logger.error(f"Error processing transcripts: {e}")
                await asyncio.sleep(1)
    
    async def generate_and_send_speech(self, text: str):
        """Generate speech and queue for sending"""
        try:
            logger.info(f"Generating speech for: {text[:50]}...")
            
            # Preprocess text
            processed_text = await self.tts_handler.preprocess_text(text)
            
            # Generate speech
            audio_data = await self.tts_handler.generate_speech(processed_text)
            
            if audio_data:
                # Convert to Twilio format (mulaw)
                mulaw_audio = self.audio_processor.convert_from_elevenlabs_format(
                    audio_data, 
                    output_format='mulaw'
                )
                
                # Add to stream manager
                await self.audio_stream_manager.add_audio(mulaw_audio)
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
    
    async def stream_audio(self):
        """Stream audio to Twilio"""
        while self.is_active:
            try:
                if self.audio_stream_manager.has_audio():
                    # Get audio chunk
                    audio_chunk = await self.audio_stream_manager.get_audio()
                    
                    # Send to Twilio
                    await self.send_audio_to_twilio(audio_chunk)
                else:
                    # No audio, wait a bit
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error streaming audio: {e}")
                await asyncio.sleep(0.1)
    
    async def send_audio_to_twilio(self, audio_data: bytes):
        """Send audio data to Twilio via WebSocket"""
        if not self.stream_sid or not self.websocket:
            return
        
        try:
            # Split audio into chunks (Twilio expects small chunks)
            chunk_size = 8000  # 1 second of audio at 8kHz
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Convert to base64
                audio_base64 = self.audio_processor.audio_to_base64(chunk)
                
                # Create message for Twilio
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                # Send via WebSocket
                await self.websocket.send_json(message)
                
                # Small delay between chunks
                await asyncio.sleep(0.02)
                
        except Exception as e:
            logger.error(f"Error sending audio to Twilio: {e}")

class WebSocketManager:
    """Manage WebSocket connections for Twilio streams"""
    
    def __init__(self):
        self.active_sessions: Dict[str, CallSession] = {}
    
    async def handle_websocket(self, websocket: WebSocket, call_sid: str):
        """Handle WebSocket connection for a call"""
        await websocket.accept()
        
        session = None
        try:
            logger.info(f"WebSocket connected for call {call_sid}")
            
            # Create call session
            session = CallSession(call_sid, websocket)
            self.active_sessions[call_sid] = session
            
            # Start the session
            await session.start()
            
            # Handle messages
            while True:
                try:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process message
                    await session.handle_message(message)
                    
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for call {call_sid}")
                    break
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    
        except Exception as e:
            logger.error(f"WebSocket error for call {call_sid}: {e}")
            
        finally:
            # Clean up
            if session:
                await session.stop()
            
            if call_sid in self.active_sessions:
                del self.active_sessions[call_sid]
            
            try:
                await websocket.close()
            except:
                pass
            
            logger.info(f"WebSocket handler completed for call {call_sid}")

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

# Validate configuration
Config.validate()

# Initialize FastAPI app with updated settings for FastAPI 0.110.0
app = FastAPI(
    title="Voice Chatbot API",
    description="AI Voice Chatbot using Twilio, Deepgram, OpenAI, and ElevenLabs",
    version="1.0.0"
)

# Initialize handlers
twilio_handler = TwilioHandler()
websocket_manager = WebSocketManager()

# Configure logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Chatbot API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "voice_webhook": "/voice",
            "status_webhook": "/status",
            "websocket": "/ws/{call_sid}",
            "health": "/health",
            "calls": "/calls"
        }
    }

@app.post("/voice")
async def voice_webhook(request: Request):
    """Handle incoming voice call webhook from Twilio"""
    try:
        # Get form data
        form_data = await request.form()
        request_data = dict(form_data)
        
        # Get Twilio signature
        signature = request.headers.get('X-Twilio-Signature', '')
        
        # Validate request (skip in debug mode)
        if not Config.DEBUG:
            url = str(request.url)
            if not twilio_handler.validate_request(url, request_data, signature):
                logger.warning("Invalid Twilio signature")
                raise HTTPException(status_code=403, detail="Invalid signature")
        
        # Handle the incoming call
        twiml_response = await twilio_handler.handle_incoming_call(request_data)
        
        # Return TwiML response
        return PlainTextResponse(
            content=twiml_response,
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Error handling voice webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/status")
async def status_webhook(request: Request):
    """Handle call status updates from Twilio"""
    try:
        # Get form data
        form_data = await request.form()
        request_data = dict(form_data)
        
        # Handle status update
        await twilio_handler.handle_call_status(request_data)
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error handling status webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    """WebSocket endpoint for Twilio media streams"""
    await websocket_manager.handle_websocket(websocket, call_sid)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "twilio": "connected",
            "openai": "configured",
            "deepgram": "configured",
            "elevenlabs": "configured"
        },
        "package_versions": {
            "fastapi": "0.110.0",
            "deepgram-sdk": "4.1.0",
            "openai": "1.12.0",
            "elevenlabs": "1.0.1",
            "twilio": "9.0.4"
        }
    }

@app.get("/calls")
async def list_calls():
    """List active calls"""
    return {
        "active_calls": list(twilio_handler.active_calls.keys()),
        "count": len(twilio_handler.active_calls),
        "calls": twilio_handler.active_calls,
        "sessions": list(websocket_manager.active_sessions.keys())
    }

@app.post("/test/tts")
async def test_tts(request: Request):
    """Test TTS endpoint"""
    try:
        data = await request.json()
        text = data.get("text", "Hello, this is a test.")
        
        tts = TTSHandler()
        audio_data = await tts.generate_speech(text)
        
        return {
            "status": "success",
            "text": text,
            "audio_size": len(audio_data),
            "message": "Audio generated successfully"
        }
        
    except Exception as e:
        logger.error(f"TTS test error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/test/llm")
async def test_llm(request: Request):
    """Test LLM endpoint"""
    try:
        data = await request.json()
        prompt = data.get("prompt", "Hello, how are you?")
        
        llm = SimpleLLMHandler()
        response = await llm.get_response(prompt)
        
        return {
            "status": "success",
            "prompt": prompt,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"LLM test error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/test/stt")
async def test_stt():
    """Test STT connection"""
    try:
        async def dummy_transcript(text: str, is_final: bool):
            logger.info(f"Test transcript: {text} (final: {is_final})")
        
        stt = STTHandler(dummy_transcript)
        success = await stt.start()
        await stt.stop()
        
        return {
            "status": "success" if success else "failed",
            "message": "STT connection test completed"
        }
        
    except Exception as e:
        logger.error(f"STT test error: {e}")
        return {"status": "error", "message": str(e)}

def main():
    """Main entry point"""
    logger.info("Starting Voice Chatbot API...")
    logger.info(f"Server URL: {Config.WEBHOOK_BASE_URL}")
    logger.info(f"Twilio Phone: {Config.TWILIO_PHONE_NUMBER}")
    logger.info(f"Debug Mode: {Config.DEBUG}")
    
    # Run the server with uvicorn 0.29.0
    if Config.DEBUG:
        # Use import string for reload functionality
        uvicorn.run(
            "main:app",
            host=Config.SERVER_HOST,
            port=Config.SERVER_PORT,
            log_level=Config.LOG_LEVEL.lower(),
            reload=True,
            access_log=True
        )
    else:
        # Use app object for production
        uvicorn.run(
            app,
            host=Config.SERVER_HOST,
            port=Config.SERVER_PORT,
            log_level=Config.LOG_LEVEL.lower(),
            reload=False,
            access_log=False
        )

if __name__ == "__main__":
    main()