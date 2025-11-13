from __future__ import annotations

from typing import Any, Dict

import numpy as np
import librosa

from .base import BaseStage, StageContext, StagePayload


class RestorationStage(BaseStage):
    """Apollo-inspired audio restoration with de-essing and equalisation fallback."""

    def __init__(self, backend: str = "apollo", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("restoration", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_apollo = False
        self._dry_wet = kwargs.get("dry_wet", 0.7)

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "apollo":
            try:  # pragma: no cover
                from apollo_infer import ApolloRestorer  # type: ignore

                device = context.device if context.device in {"cpu", "cuda"} else "cpu"
                self._model = ApolloRestorer(device=device)
                self._use_apollo = True
            except Exception:
                self._use_apollo = False
        self._log_backend_choice(self.backend, self._use_apollo, extra={"dry_wet": self._dry_wet})

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        target_audio = context.get("target_audio", payload.mixture)
        sr = payload.sample_rate

        if self._use_apollo and self._model is not None:  # pragma: no cover
            enhanced = self._model.enhance(target_audio, sr)
        else:
            enhanced = self._fallback_restore(target_audio, sr)

        blended = (self._dry_wet * enhanced) + ((1.0 - self._dry_wet) * target_audio)
        payload.mixture = blended.astype(np.float32)
        context.set("restored_audio", payload.mixture)
        return {"restored_audio": payload.mixture}

    def _fallback_restore(self, audio: np.ndarray, sr: int) -> np.ndarray:
        bandpass = librosa.effects.preemphasis(audio, coef=0.95)
        de_essed = self._de_ess(bandpass, sr)
        return librosa.util.normalize(de_essed)

    def _de_ess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        S = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude, phase = librosa.magphase(S)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        sibilant_idx = (freqs > 5000) & (freqs < 12000)
        magnitude[sibilant_idx] *= 0.7
        return librosa.istft(magnitude * phase, hop_length=256, length=len(audio))

