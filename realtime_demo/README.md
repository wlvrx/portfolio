# Real-time Speech Transcription Demo

A demonstration of real-time speech-to-text transcription using faster-whisper. This demo captures audio from the microphone and transcribes it in near real-time by processing audio in chunks.

## Features

- Near real-time transcription using faster-whisper
- Configurable chunk size for processing
- Support for different model sizes 
- GPU acceleration support (when available)

## Technical Details

The system uses the following key components:
- faster-whisper for efficient speech recognition
- PyAudio for audio capture
- CUDA acceleration support (optional)
- Chunk-based processing for real-time performance

## Requirements

```bash
pip install faster-whisper pyaudio torch numpy
```

Note: For GPU acceleration, you'll need a CUDA-compatible GPU and the appropriate CUDA toolkit installed.

## Usage

1. Run the demo:
```bash
python realtime.py
```

2. The system will:
   - Initialize the speech recognition model
   - Start capturing audio from your microphone
   - Display transcriptions in real-time
   - Press Ctrl+C to stop recording

## Configuration

You can modify the following parameters in the code:
- `model_size`: Choose from "tiny", "base", "small", "medium", "large"
- `chunk_duration`: Duration of each audio chunk in seconds
- `language`: Set to None for auto-detection or specify language code
- `device`: "cuda" for GPU acceleration or "cpu" for CPU processing

## Note

1. faster-whisper and the chunking workaround was used since this was created prior to the release of more sophisticated real-time transcription technologies like OpenAI's Realtime API. This way it's also cheaper to implement.
2. This is a demonstration version. The full implementation includes additional features and optimizations that are part of a private repository.

## License

This demo is for educational purposes only. The full implementation is proprietary and confidential. 