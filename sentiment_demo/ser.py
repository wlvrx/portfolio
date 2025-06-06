import os
import numpy as np
import librosa
import librosa.display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Emotion mapping
EMOTIONS = {
    'SAD': 'sadness',
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral'
}

def extract_features(audio_path):
    """
    Extract audio features using librosa
    """
    # Load audio file
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increased from 13 to 20
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    # Additional features
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Calculate statistics of features
    features = []
    # MFCCs
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    features.extend(np.max(mfccs, axis=1))
    features.extend(np.min(mfccs, axis=1))
    
    # Spectral features
    features.extend(np.mean(spectral_center, axis=1))
    features.extend(np.std(spectral_center, axis=1))
    features.extend(np.mean(spectral_bandwidth, axis=1))
    features.extend(np.std(spectral_bandwidth, axis=1))
    
    # Chroma features
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
    
    # Spectral rolloff
    features.extend(np.mean(spectral_rolloff, axis=1))
    features.extend(np.std(spectral_rolloff, axis=1))
    
    # Zero crossing rate
    features.extend(np.mean(zero_crossing_rate, axis=1))
    features.extend(np.std(zero_crossing_rate, axis=1))
    
    # RMS energy
    features.extend(np.mean(rms, axis=1))
    features.extend(np.std(rms, axis=1))
    
    # Spectral contrast
    features.extend(np.mean(spectral_contrast, axis=1))
    features.extend(np.std(spectral_contrast, axis=1))
    
    return features

def get_emotion_from_filename(filename):
    """
    Extract emotion from CREMA-D filename format
    Example: 1091_TAI_SAD_XX.wav -> SAD
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        emotion_code = parts[2]
        if emotion_code in EMOTIONS:
            return EMOTIONS[emotion_code]
    return None

def prepare_dataset(data_dir):
    """
    Prepare dataset from audio files
    """
    features = []
    labels = []
    
    for audio_file in os.listdir(data_dir):
        if audio_file.endswith('.wav'):
            emotion = get_emotion_from_filename(audio_file)
            if emotion is None:
                continue
                
            audio_path = os.path.join(data_dir, audio_file)
            try:
                feature_vector = extract_features(audio_path)
                features.append(feature_vector)
                labels.append(emotion)
                print(f"Processed {audio_file} - Emotion: {emotion}")
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
    
    if not features:
        raise ValueError("No valid audio files found in the dataset directory")
        
    return np.array(features), np.array(labels)

def train_model(data_dir):
    """
    Train the emotion recognition model
    """
    print("Preparing dataset...")
    X, y = prepare_dataset(data_dir)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with improved parameters
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,  # Increased from 100
        max_depth=20,      # Added max depth
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Added class weight
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Testing accuracy: {test_accuracy:.2f}")
    
    # Save model and scaler in the sentiment_demo directory
    model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'emotion_scaler.joblib')
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    return model, scaler

def predict_emotion(audio_path, model=None, scaler=None):
    """
    Predict emotion from an audio file
    """
    # If model and scaler are not provided, try to load them
    if model is None or scaler is None:
        model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.joblib')
        scaler_path = os.path.join(os.path.dirname(__file__), 'emotion_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print("Loading saved model and scaler...")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        else:
            raise ValueError("Model and scaler files not found. Please train the model first.")
    
    features = extract_features(audio_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return prediction

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(__file__), "Crema")  # Get the correct path to Crema directory
    
    # Train model
    model, scaler = train_model(data_dir)
    
    # Example prediction
    test_audio = os.path.join(data_dir, "1091_TAI_SAD_XX.wav")  # Example test file
    if os.path.exists(test_audio):
        predicted_emotion = predict_emotion(test_audio, model, scaler)
        print(f"Predicted emotion: {predicted_emotion}")
