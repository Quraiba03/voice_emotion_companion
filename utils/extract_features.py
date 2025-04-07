import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Take the mean across time for each MFCC coefficient to get 1D feature vector
        mfccs_mean = np.mean(mfccs, axis=1)

        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
