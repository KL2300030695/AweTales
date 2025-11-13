from __future__ import annotations

import numpy as np
from typing import Any, Dict

import librosa

from .base import BaseStage, StageContext, StagePayload


class DenoiserStage(BaseStage):
    """UVR-MDX inspired denoiser with Wiener filtering fallback."""

    def __init__(self, backend: str = "uvr_mdx", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("denoiser", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_uvr = False

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "uvr_mdx":
            try:  # pragma: no cover - heavy dependency
                from uvr import MDXNet  # type: ignore

                model_path = self.kwargs.get("model_path")
                self._model = MDXNet(model_path=model_path)
                self._use_uvr = True
            except Exception:
                self._use_uvr = False
        self._log_backend_choice(self.backend, self._use_uvr)

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        audio = payload.mixture
        sr = payload.sample_rate
        if self._use_uvr and self._model is not None:  # pragma: no cover - optional heavy dependency
            denoised = self._model.separate(audio, sr)  # type: ignore[no-untyped-call]
            if isinstance(denoised, dict):
                denoised_audio = denoised.get("vocals", audio)
            else:
                denoised_audio = denoised
        else:
            denoised_audio = self._spectral_gate(audio)

        payload.mixture = denoised_audio.astype(np.float32)
        context.set("denoised_audio", payload.mixture)
        return {"denoised_audio": payload.mixture}

    def _spectral_gate(self, audio: np.ndarray) -> np.ndarray:
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude, phase = librosa.magphase(stft)
        noise_profile = np.median(magnitude[:, :10], axis=1, keepdims=True)
        reduction = np.maximum(magnitude - noise_profile, 0.0)
        denoised = librosa.istft(reduction * phase, hop_length=256)
        if len(denoised) < len(audio):
            padded = np.zeros_like(audio)
            padded[: len(denoised)] = denoised
            denoised = padded
        return denoised

