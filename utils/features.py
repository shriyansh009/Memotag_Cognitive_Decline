import whisper
import librosa
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download('punkt')
model = whisper.load_model("tiny")
FILLERS = [
    "uh", "um", "er", "ah", "like", "you know", "I mean", "sort of",
    "kind of", "actually", "basically", "right", "well", "so", "huh",
    "okay", "you see", "literally", "honestly", "I guess", "hmm", "alright",
    "just", "anyway", "I think", "I suppose", "maybe", "perhaps", "you know what I mean"
]

def compute_features(file_path, return_text=False):
    result = model.transcribe(file_path)
    text = result["text"]
    words = word_tokenize(text.lower())
    total_words = len(words)
    fillers = [w for w in words if w in FILLERS]
    lexical_diversity = len(set(words)) / total_words if total_words > 0 else 0
    pause_count = len(re.findall(r"[.?!]", text))

    y, sr = librosa.load(file_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    try:
        pitch = librosa.yin(y, fmin=50, fmax=300)
        pitch_var = np.nanstd(pitch)
    except:
        pitch_var = 0

    features = {
        "total_words": total_words,
        "fillers_per_100_words": round((len(fillers) / total_words) * 100 if total_words > 0 else 0, 2),
        "lexical_diversity": round(lexical_diversity, 2),
        "pause_estimate": pause_count,
        "duration_sec": round(duration, 2),
        "tempo": round(tempo[0], 2),
        "energy_rms": round(rms, 5),
        "pitch_variability": round(pitch_var, 2),
        "speech_rate": round(total_words / duration, 2) if duration > 0 else 0
    }

    return (features, text) if return_text else features

def summarize_text(text):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:2]) + ("..." if len(sentences) > 2 else "")

def find_hesitations(text):
    words = word_tokenize(text.lower())
    return [(i, word) for i, word in enumerate(words) if word in FILLERS]
