import numpy as np
import joblib
from utils.audio_utils import extract_features

def get_emotion_prediction(audio_path, model):
    features = extract_features(audio_path)

    if features is None:
        return "Error extracting features."

    # ➕ Convert to mean MFCC (40 features)
    mean_mfcc = np.mean(features, axis=1).reshape(1, -1)

    try:
        prediction = model.predict(mean_mfcc)
        return prediction[0]
    except Exception as e:
        return f"Prediction error: {str(e)}"
