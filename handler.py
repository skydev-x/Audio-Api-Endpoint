"""Kokoro TTS handler for RunPod serverless."""
import runpod
from kokoro import KPipeline
import soundfile as sf
import torch
import tempfile
import base64
import os

# Initialize pipeline once at startup (outside handler for efficiency)
pipeline = KPipeline(lang_code='a')

def handler(job):
    """Handler function that processes TTS jobs and returns audio."""
    job_input = job["input"]
    
    # Get input parameters
    text = job_input.get("text", "")
    voice = job_input.get("voice", "af_heart")  # Default voice
    output_format = job_input.get("output_format", "base64")  # "base64" or "url"
    
    # Validate input
    if not text:
        return {"error": "No text provided. Please provide 'text' in input."}
    
    try:
        # Generate audio
        generator = pipeline(text, voice=voice)
        
        # Get the first (and typically only) audio output
        audio_data = None
        sample_rate = 24000
        
        for i, (gs, ps, audio) in enumerate(generator):
            print(f"Generated segment {i}: {gs} graphemes, {ps} phonemes")
            audio_data = audio
            break  # Take the first segment
        
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
            
            return {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "format": "wav",
                "voice": voice,
                "text": text
            }
        else:
            # For RunPod, you might want to upload to S3 or return file path
            # This returns the temp file path (ensure it's accessible)
            return {
                "audio_path": temp_path,
                "sample_rate": sample_rate,
                "format": "wav",
                "voice": voice,
                "text": text,
                "note": "Remember to clean up temp file after use"
            }
    
    except Exception as e:
        return {"error": f"TTS generation failed: {str(e)}"}

runpod.serverless.start({"handler": handler})
