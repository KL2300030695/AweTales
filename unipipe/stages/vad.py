from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple

import librosa

from .base import BaseStage, StageContext, StagePayload


class VoiceActivityDetectionStage(BaseStage):
    """FSMN-Monophone VAD approximation using energy-based detection."""

    def __init__(self, backend: str = "fsmn", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("vad", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_fsmn = False
        self._top_db = kwargs.get("top_db", 35.0)
        self._frame_length = kwargs.get("frame_length", 1024)
        self._hop_length = kwargs.get("hop_length", 256)

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "fsmn":
            try:  # pragma: no cover - optional
                from funasr import AutoModel  # type: ignore

                self._model = AutoModel.from_pretrained("fsmn-vad", trust_remote_code=True)
                self._use_fsmn = True
            except Exception:
                self._use_fsmn = False
        self._log_backend_choice(
            self.backend,
            self._use_fsmn,
            extra={"frame_length": self._frame_length, "hop_length": self._hop_length},
        )

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        audio = payload.mixture
        sr = payload.sample_rate
        if self._use_fsmn and self._model is not None:  # pragma: no cover
            vad_segments = self._model.generate(audio, fs=sr)  # type: ignore[no-untyped-call]
            speech_segments = [(float(s), float(e)) for s, e in vad_segments]
        else:
            speech_segments = self._energy_vad(audio, sr)
        context.set("vad_segments", speech_segments)
        return {"vad_segments": speech_segments}

    def _energy_vad(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        intervals = librosa.effects.split(
            audio,
            top_db=self._top_db,
            frame_length=self._frame_length,
            hop_length=self._hop_length,
        )
        segments: List[Tuple[float, float]] = []
        for start, end in intervals:
            segments.append((start / sr, end / sr))
        return segments

