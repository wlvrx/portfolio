"""
Real-time Speech-to-Text using faster-whisper
This script captures audio from the microphone and transcribes it in near real-time by processing audio in chunks.
"""

import queue
import threading
import time

import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from typing import Optional


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16" if torch.cuda.is_available() else "float32",
        language: Optional[str] = None,
        chunk_duration: float = 2.0,  # Duration of each audio chunk in seconds
        sample_rate: int = 16000,
        channels: int = 1
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        self.language = language
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Audio parameters
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    
    def _process_audio_chunk(self, audio_data: bytes) -> str:
        """Process a single chunk of audio data"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize audio
        audio_float32 = audio_array.astype(np.float32) / 32768.0
        
        # Transcribe the chunk
        segments, _ = self.model.transcribe(
            audio_float32,
            language=self.language,
            beam_size=5,
            vad_filter=True
        )
        
        # Combine all segments from the chunk
        return " ".join(segment.text for segment in segments)
    
    
    def start_recording(self):
        """Start recording and transcribing audio"""
        self.is_recording = True
        
        # Open the audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        print("Realtime recording started. Press Ctrl+C to stop.")
        
        try:
            while self.is_recording:
                # Get audio chunk from queue
                audio_data = self.audio_queue.get()
                
                # Process the chunk
                transcription = self._process_audio_chunk(audio_data)
                
                if transcription.strip():
                    print(f"Transcription: {transcription}")
                
        except KeyboardInterrupt:
            print("\nStopping recording...")
        finally:
            self.stop_recording()
    
    
    def stop_recording(self):
        """Stop recording and clean up resources"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


def main():
    # Initialize transcriber
    transcriber = AudioTranscriber(
        # Options include tiny, base, small, medium, large. The bigger the model the more accurate, but slower
        # Note: have to download the model locally before it works
        model_size="base",  
        language="en",
        chunk_duration=2.0  # Process 2-second chunks
    )
    
    # Start recording and transcribing
    transcriber.start_recording()

if __name__ == "__main__":
    main()
