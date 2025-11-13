from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseStage, StageContext, StagePayload


class SpeakerRecognitionStage(BaseStage):
    """Encapsulates ERes2NetV2-Large embeddings with MFCC fallback."""

    def __init__(self, backend: str = "eres2netv2", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("speaker_recognition", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_eres2net = False
        self._scaler = StandardScaler()

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "eres2netv2":
            try:  # pragma: no cover
                from modelscope.pipelines import pipeline  # type: ignore
                from modelscope.utils.constant import Tasks  # type: ignore

                device = context.device if context.device in {"cpu", "cuda"} else "cpu"
                model_id = self.kwargs.get("model_id", "damo/speech_eres2netv2_sre-en-us-test")
                self._model = pipeline(Tasks.speaker_verification, model=model_id, device=device)  # type: ignore
                self._use_eres2net = True
            except Exception:
                self._use_eres2net = False
        self._log_backend_choice(self.backend, self._use_eres2net)

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        target_ref = payload.target_reference
        if target_ref is None:
            raise ValueError("Target reference audio must be supplied to speaker recognition stage.")
        sr = payload.sample_rate

        target_embedding = self._embed(target_ref, sr)
        context.set("target_embedding", target_embedding)

        vad_segments: List[Tuple[float, float]] = context.get("vad_segments", [])
        mixture = context.get("denoised_audio", payload.mixture)

        segment_scores: List[Dict[str, Any]] = []
        for idx, (start_t, end_t) in enumerate(vad_segments):
            start = int(start_t * sr)
            end = int(end_t * sr)
            snippet = mixture[start:end]
            if snippet.size < sr * 0.1:
                continue
            emb = self._embed(snippet, sr)
            score = float(cosine_similarity(target_embedding.reshape(1, -1), emb.reshape(1, -1))[0, 0])
            segment_scores.append(
                {
                    "segment_index": idx,
                    "start": start_t,
                    "end": end_t,
                    "similarity": score,
                    "embedding": emb.astype(np.float32),
                }
            )
        context.set("speaker_similarity", segment_scores)
        return {"speaker_similarity": segment_scores, "target_embedding": target_embedding}

    def _embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self._use_eres2net and self._model is not None:  # pragma: no cover
            result = self._model(audio=audio, sample_rate=sr)  # type: ignore[no-untyped-call]
            embedding = result.get("embedding")
            if embedding is not None:
                return np.array(embedding, dtype=np.float32)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T
        self._scaler.fit(mfcc)
        normed = self._scaler.transform(mfcc)
        embedding = np.mean(normed, axis=0)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.astype(np.float32)

