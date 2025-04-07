import streamlit as st
from utils.audio_utils import record_audio
from utils.emotion_model import load_model
from gpt_response import get_emotion_prediction

# Page config
st.set_page_config(
    page_title="Voice-Based Emotion Companion",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Session state
if "emotion" not in st.session_state:
    st.session_state.emotion = None

# Emotion gradients (dark → light)
emotion_gradients = {
    "happy": "linear-gradient(135deg, #F4C430, #FFF9C4)",       # Golden yellow to pale yellow
    "sad": "linear-gradient(135deg, #2F4F4F, #AAB2BD)",          # Dark slate gray to light steel blue
    "angry": "linear-gradient(135deg, #8B0000, #FF6B5A)",        # Dark red to light red
    "fearful": "linear-gradient(135deg, #4B0082, #D8BFD8)",      # Indigo to thistle
    "disgust": "linear-gradient(135deg, #006400, #A0E7E5)",      # Dark green to mint
    "surprised": "linear-gradient(135deg, #4682B4, #B3DFFC)",    # Steel blue to baby blue
    "neutral": "linear-gradient(135deg, #808080, #D3D3D3)",      # Gray to light gray
    "calm": "linear-gradient(135deg, #2E8B57, #A8D5BA)"          # Sea green to pastel green
}

# Default background (reset on rerun)
if st.session_state.emotion is None:
    st.markdown("""
        <style>
            body, .stApp {
                background: linear-gradient(135deg, black, #2f2f2f);
                color: white;
                transition: background 1s ease-in-out;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    gradient = emotion_gradients.get(st.session_state.emotion.lower(), "#FFFFFF")
    st.markdown(f"""
        <style>
            body, .stApp {{
                background: {gradient};
                color: black;
                transition: background 1s ease-in-out;
            }}
        </style>
    """, unsafe_allow_html=True)

# Title and instructions
st.title("🎙️ Voice-Based Emotion Companion")
st.markdown("Record your voice and get a comforting response based on your detected emotion.")

# Record audio
if st.button("🎤 Record Voice"):
    record_audio("input.wav")
    st.success("✅ Recording complete and saved as input.wav")

# Analyze and predict emotion
if st.button("🔍 Analyze Emotion"):
    model = load_model()
    emotion = get_emotion_prediction("input.wav", model)
    st.session_state.emotion = emotion
    gradient = emotion_gradients.get(emotion.lower(), "#FFFFFF")

    # Emotion popup with gradient background
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Algerian&display=swap');

        .emotion-popup {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: {gradient};
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }}
        .emotion-popup h1 {{
            font-family: Algerian, sans-serif;
            font-size: 90px;
            color: black;
            text-shadow: 2px 2px 10px white;
        }}
        </style>
        <div class="emotion-popup">
            <h1>{emotion.upper()} {'😊' if emotion == 'happy' else '😢' if emotion == 'sad' else '😐'}</h1>
        </div>
    """, unsafe_allow_html=True)

    if emotion.lower() == "happy":
        st.balloons()
