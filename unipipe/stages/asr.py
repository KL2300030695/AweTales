from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import BaseStage, StageContext, StagePayload


class ASRStage(BaseStage):
    """Paraformer / Whisper wrapper with graceful fallback."""

    def __init__(self, backend: str = "whisper", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("asr", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._use_faster_whisper = False
        self._use_whisper = False
        self._use_paraformer = False
        self._language = kwargs.get("language")

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "whisper":
            try:  # pragma: no cover - heavy dependency
                from faster_whisper import WhisperModel  # type: ignore

                model_size = self.kwargs.get("model_size", "medium")
                device = context.device if context.device in {"cpu", "cuda"} else "cpu"
                compute_type = self.kwargs.get("compute_type", "auto")
                self._model = WhisperModel(model_size, device=device, compute_type=compute_type)  # type: ignore
                self._use_faster_whisper = True
            except Exception:
                try:
                    import whisper  # type: ignore

                    model_size = self.kwargs.get("model_size", "base")
                    self._model = whisper.load_model(model_size)
                    self._use_whisper = True
                except Exception:
                    self._use_whisper = False
        elif self.backend == "paraformer":
            try:  # pragma: no cover
                from funasr import AutoModel  # type: ignore

                self._model = AutoModel.from_pretrained("paraformer-large", trust_remote_code=True)
                self._use_paraformer = True
            except Exception:
                self._use_paraformer = False

        self._log_backend_choice(
            self.backend,
            any([self._use_faster_whisper, self._use_whisper, self._use_paraformer]),
        )

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        sr = payload.sample_rate
        audio = context.get("restored_audio", payload.mixture)
        diar_segments: List[Dict[str, Any]] = context.get("diarization_segments", [])

        transcripts: List[Dict[str, Any]] = []
        for seg in diar_segments:
            start = seg["start"]
            end = seg["end"]
            speaker = seg.get("speaker", "Speaker")
            snippet = self._slice_audio(audio, sr, start, end)
            if snippet.size == 0:
                continue
            text, confidence, lang = await self._transcribe(snippet, sr)
            transcripts.append(
                {
                    "speaker": speaker,
                    "start": float(start),
                    "end": float(end),
                    "text": text,
                    "confidence": float(confidence),
                    "lang": lang,
                }
            )

        context.set("transcripts", transcripts)
        return {"transcripts": transcripts}

    async def _transcribe(self, audio: np.ndarray, sr: int) -> tuple[str, float, str | None]:
        if self._use_faster_whisper and self._model is not None:  # pragma: no cover
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=self.kwargs.get("beam_size", 5),
                vad_filter=True,
                temperature=0.0,
            )
            text_parts = []
            confidences = []
            for segment in segments:
                text_parts.append(segment.text)
                confidences.append(getattr(segment, "avg_logprob", 0.0))
            text = " ".join(text_parts).strip()
            if confidences:
                confidence = float(np.mean([np.exp(c) for c in confidences]))
            else:
                confidence = 0.5
            language = getattr(info, "language", self._language)
            return text, self._sanitize_confidence(confidence), language

        if self._use_whisper and self._model is not None:  # pragma: no cover
            result = self._model.transcribe(audio, language=self._language)
            text = result.get("text", "").strip()
            language = result.get("language")
            confidence = float(result.get("avg_logprob", -0.5))
            return text, self._sanitize_confidence(float(np.exp(confidence))), language

        if self._use_paraformer and self._model is not None:  # pragma: no cover
            res = self._model.generate(audio, sampling_rate=sr)
            text = res[0]["text"]
            confidence = float(res[0].get("confidence", 0.7))
            lang = res[0].get("language", self._language)
            return text, self._sanitize_confidence(confidence), lang

        # Lightweight fallback: return energy-based pseudo transcript
        energy = np.mean(np.abs(audio))
        duration = len(audio) / sr
        if energy < 1e-3 or duration < 0.2:
            text = "[silence]"
            confidence = 0.3
        else:
            syllables = max(int(duration // 0.3), 1)
            text = " ".join(["uh"] * syllables)
            confidence = 0.35 + min(0.3, energy)
        return text, self._sanitize_confidence(confidence), self._language

    def _slice_audio(self, audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
        start_idx = max(int(start * sr), 0)
        end_idx = min(int(end * sr), len(audio))
        return audio[start_idx:end_idx]

    def _sanitize_confidence(self, value: float) -> float:
        if np.isnan(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))

