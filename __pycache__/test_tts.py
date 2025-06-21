# test_tts.py - Create this file
import asyncio
from tts_handler import TTSHandler

async def test_tts():
    print("Testing Text-to-Speech...")
    tts = TTSHandler()
    
    # Test text preprocessing
    text = "Hello!   This is a    test..."
    processed = await tts.preprocess_text(text)
    print(f"Original: '{text}'")
    print(f"Processed: '{processed}'")
    
    # Test speech generation
    print("\nGenerating speech...")
    try:
        audio_data = await tts.generate_speech("Hello, this is a test of the text to speech system.")
        print(f"✅ Generated {len(audio_data)} bytes of audio")
        
        # Save to file for manual verification
        with open("test_output.mp3", "wb") as f:
            f.write(audio_data)
        print("✅ Audio saved to test_output.mp3 - you can play this file to verify")
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        print("Check your ELEVENLABS_API_KEY in .env")

asyncio.run(test_tts())