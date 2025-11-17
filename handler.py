"""Kokoro TTS handler - supports both local and RunPod serverless execution."""
import os
from kokoro import KPipeline
import soundfile as sf
import torch
import tempfile
import base64
import numpy as np


# Check if running in RunPod mode via environment variable
USE_RUNPOD = os.getenv("USE_RUNPOD", "false").lower() == "true"

if USE_RUNPOD:
    import runpod

# Initialize pipeline once at startup (outside handler for efficiency)
pipeline = KPipeline(lang_code='a')

def handler(job=None):
    """Handler function that processes TTS jobs and returns audio.
    
    Args:
        job: RunPod job object (only used when USE_RUNPOD=true)
    """
    # Get input based on mode
    if USE_RUNPOD and job:
        job_input = job["input"]
        text = job_input.get("text", "Testing local audio output in basic hardware")
        voice = job_input.get("voice", "af_heart")
        output_format = job_input.get("output_format", "base64")
    else:
        # Local mode defaults
        text = (
            "You are stronger than you think, but you forget it because you let small problems steal your focus. "
            "The Stoics believed that the world cannot break you… only your reaction can. "
            "Every challenge you face today is not an obstacle. It is training. "
            "Marcus Aurelius said, 'The impediment to action advances action.' "
            "So when something blocks your path, don’t stop. Don’t complain. "
            "Use it. "
            "Turn struggle into strength. "
            "Turn pressure into power. "
            "And remember, the person you want to become is built in moments exactly like this."
        )
        voice = "af_heart"
        output_format = "base64"
    
    # Validate input
    if not text:
        return {"error": "No text provided. Please provide 'text' in input."}
    
    try:
        # Generate audio
        generator = pipeline(text, voice=voice)
        
        # Get the first (and typically only) audio output
        audio_segments = []
        sample_rate = 24000

        for i, (gs, ps, audio) in enumerate(generator):
            print(f"Generated segment {i}: {gs} graphemes, {ps} phonemes")
            audio_segments.append(audio)

        if not audio_segments:
            return {"error": "Failed to generate audio"}

        audio_data = np.concatenate(audio_segments)        
        if audio_data is None:
            return {"error": "Failed to generate audio"}
        
        # Create temporary file to save audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)
        
        # Return based on output format
        if output_format == "base64":
            # Read file and encode as base64
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Clean up temp file
            os.unlink(temp_path)
            
            result = {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "format": "wav",
                "voice": voice,
                "text": text
            }
        else:
            # Return file path
            result = {
                "audio_path": temp_path,
                "sample_rate": sample_rate,
                "format": "wav",
                "voice": voice,
                "text": text,
                "note": "Remember to clean up temp file after use"
            }
        
        # In local mode, also save to a permanent file
        if not USE_RUNPOD:
            output_file = "output_audio.wav"
            sf.write(output_file, audio_data, sample_rate)
            result["local_file"] = output_file
            print(f"✓ Audio saved to: {output_file}")
        
        return result
    
    except Exception as e:
        return {"error": f"TTS generation failed: {str(e)}"}

# Main execution
if __name__ == "__main__":
    if USE_RUNPOD:
        print("Starting in RunPod serverless mode...")
        runpod.serverless.start({"handler": handler})
    else:
        print("Running in local mode...")
        result = handler()
        print(f"Result: {result}")
