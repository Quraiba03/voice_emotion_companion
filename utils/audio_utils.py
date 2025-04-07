import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np

def record_audio(filename, duration=4, fs=22050):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("Recording saved.")

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Pad or trim MFCCs to a consistent shape
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
