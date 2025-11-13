import numpy as np
import soundfile as sf
import librosa
from typing import Tuple

TARGET_SR = 16000

def load_wav(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    y, s = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        y = np.zeros(1, dtype=np.float32)
    return y.astype(np.float32), s

def save_wav(path: str, y: np.ndarray, sr: int = TARGET_SR) -> None:
    sf.write(path, y, sr)

def normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(y)) if y.size else 1.0
    if m == 0:
        return y
    return (y / m) * peak
