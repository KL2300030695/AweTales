from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

import librosa

from .base import BaseStage, StageContext, StagePayload


class EndpointDetectionStage(BaseStage):
    """Approximate CAM++ endpoint detection with a lightweight spectral energy fallback."""

    def __init__(self, backend: str = "cam++", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("endpoint_detection", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._use_campp = False
        self._model = None
        self._threshold = kwargs.get("threshold", 40.0)
        self._min_silence = kwargs.get("min_silence_s", 0.3)

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "cam++":
            try:  # pragma: no cover - backend optional
                from camplusplus import CamPlusPlus  # type: ignore

                self._model = CamPlusPlus.from_pretrained("camplusplus")  # type: ignore[attr-defined]
                self._use_campp = True
            except Exception:
                self._use_campp = False
        self._log_backend_choice(self.backend, self._use_campp, extra={"threshold": self._threshold})

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        audio = payload.mixture
        sr = payload.sample_rate

        if self._use_campp and self._model is not None:  # pragma: no cover - dependent on model availability
            boundaries = self._model.detect(audio, sr)  # type: ignore[no-untyped-call]
            speech_regions = [(float(s), float(e)) for s, e in boundaries]
        else:
            speech_regions = self._energy_based_endpoint(audio, sr)

        context.set("speech_regions", speech_regions)
        return {"speech_regions": speech_regions}

    def _energy_based_endpoint(self, audio: np.ndarray, sr: int) -> List[tuple]:
        intervals = librosa.effects.split(audio, top_db=self._threshold, hop_length=256)
        speech_regions: List[tuple] = []
        for start, end in intervals:
            start_t = start / sr
            end_t = end / sr
            if end_t - start_t >= self._min_silence:
                speech_regions.append((start_t, end_t))
        return speech_regions

