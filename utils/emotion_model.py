import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.audio_utils import extract_features

# RAVDESS emotion labels map
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def parse_emotion(filename):
    return emotion_map[filename.split("-")[2]]

def load_data(dataset_path):
    X, y = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                try:
                    emotion = parse_emotion(file)
                    features = extract_features(os.path.join(root, file))

                    if features is not None:
                        # Use mean of MFCCs (shape: 40,)
                        mean_features = np.mean(features, axis=1)
                        X.append(mean_features)
                        y.append(emotion)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    return np.array(X), np.array(y)



def train_model():
    X, y = load_data("data/ravdess")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    with open("models/emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved!")

def load_model():
    with open("models/emotion_model.pkl", "rb") as f:
        return pickle.load(f)
