from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseStage, StageContext, StagePayload


class SeparationStage(BaseStage):
    """MossFormer2-inspired speech separation with MFCC cosine fallback."""

    def __init__(self, backend: str = "mossformer2", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("separation", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_mossformer = False
        self._threshold = kwargs.get("similarity_threshold", 0.65)
        self._chunk_seconds = kwargs.get("chunk_seconds", 1.0)

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "mossformer2":
            try:  # pragma: no cover
                import torch  # type: ignore
                from funasr import AutoModel  # type: ignore

                device = context.device if context.device in {"cpu", "cuda"} else "cpu"
                self._model = AutoModel.from_pretrained(
                    "iic/speech_separation_mossformer2_multispk", device=device, trust_remote_code=True
                )
                self._use_mossformer = True
            except Exception:
                self._use_mossformer = False
        self._log_backend_choice(self.backend, self._use_mossformer, extra={"threshold": self._threshold})

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        mixture = payload.mixture
        sr = payload.sample_rate
        target_ref = payload.target_reference
        if target_ref is None:
            raise ValueError("Target reference audio must be supplied to the separation stage.")

        if self._use_mossformer and self._model is not None:  # pragma: no cover
            separated = self._model.generate(mixture, sr)
            if isinstance(separated, list):
                target_audio = separated[0]
                interference = mixture - target_audio
            elif isinstance(separated, dict):
                target_audio = separated.get("target", mixture)
                interference = mixture - target_audio
            else:
                target_audio = separated
                interference = mixture - target_audio
            mask = self._estimate_mask(target_audio, mixture)
        else:
            target_audio, interference, mask = self._mfcc_cosine_separation(mixture, target_ref, sr)

        context.set("target_audio", target_audio)
        context.set("interference_audio", interference)
        context.set("separation_mask", mask)

        payload.mixture = target_audio.astype(np.float32)
        return {
            "target_audio": target_audio,
            "interference_audio": interference,
            "mask": mask,
        }

    def _mfcc_cosine_separation(
        self, mixture: np.ndarray, target_ref: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hop = 256
        win = 1024
        target_mfcc = librosa.feature.mfcc(y=target_ref, sr=sr, n_mfcc=20, hop_length=hop, n_fft=win)
        target_embed = np.mean(target_mfcc, axis=1, keepdims=True).T  # shape (1, n_mfcc)

        mix_stft = librosa.stft(mixture, n_fft=win, hop_length=hop)
        mix_mag, mix_phase = librosa.magphase(mix_stft)

        mix_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(np.abs(mix_stft) ** 2), sr=sr, n_mfcc=20)
        mix_embed = mix_mfcc.T  # frames x features
        sims = cosine_similarity(mix_embed, target_embed).flatten()  # shape (frames,)
        mask = np.clip((sims - self._threshold) / max(1e-5, 1 - self._threshold), 0.0, 1.0)  # (frames,)
        # Broadcast mask over frequency axis: mix_mag is (freq_bins, frames)
        soft_mask = mask[None, :]  # (1, frames)
        target_mag = mix_mag * soft_mask
        interference_mag = mix_mag * (1.0 - soft_mask)

        target_stft = target_mag * mix_phase
        interference_stft = interference_mag * mix_phase

        target_audio = librosa.istft(target_stft, hop_length=hop, length=len(mixture))
        interference = librosa.istft(interference_stft, hop_length=hop, length=len(mixture))
        return target_audio.astype(np.float32), interference.astype(np.float32), mask.squeeze()

    def _estimate_mask(self, target: np.ndarray, mixture: np.ndarray) -> np.ndarray:
        eps = 1e-9
        mask = np.clip(np.abs(target) / (np.abs(mixture) + eps), 0.0, 1.0)
        return mask

