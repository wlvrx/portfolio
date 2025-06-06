# Speech Emotion Recognition Demo

A simplified demonstration of real-time speech emotion recognition using machine learning. This demo uses audio features and a Random Forest Classifier to predict emotions from speech.

## Features

- Real-time emotion detection from speech input
- Supports 6 basic emotions: happiness, sadness, anger, fear, disgust, and neutral
- Uses MFCCs and other audio features for robust emotion detection
- Simple and efficient implementation

## Technical Details

The system uses the following key components:
- Audio feature extraction using librosa
- Machine learning model (Random Forest Classifier)
- Real-time audio processing capabilities

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Run the demo:
```bash
python ser.py
```

2. The system will:
   - Load the pre-trained model
   - Process audio input
   - Display predicted emotions in real-time

## Note

1. The CREMAD (Crowd-sourced Emotional Multimodal Actors Dataset) was utilized since it is publicly available
2. This is a simplified and sanitized demonstration version. The full implementation includes additional features and optimizations that are part of a private repository.

## License

This demo is for educational purposes only. The full implementation is proprietary and confidential. 