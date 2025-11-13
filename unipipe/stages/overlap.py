from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple

from .base import BaseStage, StageContext, StagePayload


class OverlapDetectionStage(BaseStage):
    """Wrap PyAnnote-style overlap detection with a spectral flatness fallback."""

    def __init__(self, backend: str = "pyannote", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("overlap_detection", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_pyannote = False

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "pyannote":
            try:  # pragma: no cover - optional heavy dependency
                from pyannote.audio import Pipeline  # type: ignore

                token = self.kwargs.get("hf_token")  # huggingface token optional
                pipeline_name = self.kwargs.get("pipeline_name", "pyannote/overlapped-speech-detection")
                self._model = Pipeline.from_pretrained(pipeline_name, use_auth_token=token)  # type: ignore
                self._use_pyannote = True
            except Exception:
                self._use_pyannote = False
        self._log_backend_choice(self.backend, self._use_pyannote)

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        audio = payload.mixture
        sr = payload.sample_rate
        speech_regions = context.get("speech_regions", [])

        if self._use_pyannote and self._model is not None:  # pragma: no cover - optional heavy dependency
            # `Pipeline` expects a dict with waveform and sample_rate
            timeline = self._model({"waveform": audio.reshape(-1, 1), "sample_rate": sr})
            overlaps = [(float(seg.start), float(seg.end)) for seg in timeline.get_timeline().support()]
        else:
            overlaps = self._flatness_overlap(audio, sr, speech_regions)

        context.set("overlap_regions", overlaps)
        return {"overlap_regions": overlaps}

    def _flatness_overlap(
        self, audio: np.ndarray, sr: int, speech_regions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not speech_regions:
            return []

        frame_len = int(0.1 * sr)
        if frame_len <= 0:
            frame_len = 1024

        overlap_candidates: List[Tuple[float, float]] = []
        for start_t, end_t in speech_regions:
            start = int(start_t * sr)
            end = int(end_t * sr)
            segment = audio[start:end]
            if segment.size < frame_len:
                continue
            # Ensure length is a multiple of frame_len to avoid reshape errors
            n_frames = segment.size // frame_len
            if n_frames <= 0:
                continue
            segment = segment[: n_frames * frame_len]
            frames = segment.reshape(n_frames, frame_len)
            flatness = self._spectral_flatness(frames)
            mean_flatness = np.mean(flatness)
            if mean_flatness > 0.4:  # heuristic threshold
                overlap_candidates.append((start_t, end_t))
        return overlap_candidates

    @staticmethod
    def _spectral_flatness(frames: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        spectrum = np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1)
        magnitude = np.abs(spectrum) + eps
        geometric_mean = np.exp(np.mean(np.log(magnitude), axis=1))
        arithmetic_mean = np.mean(magnitude, axis=1)
        return geometric_mean / (arithmetic_mean + eps)

