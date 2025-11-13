from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .base import BaseStage, StageContext, StagePayload


class DiarizationStage(BaseStage):
    """PyAnnote diarization wrapper with cosine clustering fallback."""

    def __init__(self, backend: str = "clustering", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("diarization", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_pyannote = False
        self._target_threshold = kwargs.get("target_threshold", 0.55)
        self._cluster_threshold = kwargs.get("cluster_threshold", 0.45)

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "pyannote":
            try:  # pragma: no cover - optional heavy dependency
                from pyannote.audio import Pipeline  # type: ignore

                pipeline_name = self.kwargs.get("pipeline_name", "pyannote/speaker-diarization-3.0")
                token = self.kwargs.get("hf_token")
                self._model = Pipeline.from_pretrained(pipeline_name, use_auth_token=token)  # type: ignore
                self._use_pyannote = True
            except Exception:
                self._use_pyannote = False
        self._log_backend_choice(self.backend, self._use_pyannote)

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        sr = payload.sample_rate
        if self._use_pyannote and self._model is not None:  # pragma: no cover
            diarization = self._model({"waveform": payload.mixture.reshape(-1, 1), "sample_rate": sr})
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "confidence": 1.0,
                    }
                )
        else:
            segments = self._cluster_based(context)

        context.set("diarization_segments", segments)
        return {"segments": segments}

    def _cluster_based(self, context: StageContext) -> List[Dict[str, Any]]:
        vad_segments: List[Tuple[float, float]] = context.get("vad_segments", [])
        similarity_info: List[Dict[str, Any]] = context.get("speaker_similarity", [])
        if not vad_segments or not similarity_info:
            return []

        other_embeddings: List[np.ndarray] = []
        other_labels: List[int] = []
        segments: List[Dict[str, Any]] = []
        speaker_counter = 0

        for info in similarity_info:
            start_t = info["start"]
            end_t = info["end"]
            similarity = info["similarity"]
            embedding = info.get("embedding")

            confidence = float(np.clip(similarity, 0.0, 1.0))
            if similarity >= self._target_threshold:
                speaker_label = "Target"
            else:
                speaker_idx = self._assign_cluster(embedding, other_embeddings)
                if speaker_idx >= len(other_embeddings):
                    other_embeddings.append(embedding)
                    other_labels.append(speaker_counter)
                    speaker_counter += 1
                    speaker_idx = len(other_embeddings) - 1
                speaker_label = f"Speaker_{chr(ord('B') + speaker_idx)}"

            segments.append(
                {
                    "speaker": speaker_label,
                    "start": start_t,
                    "end": end_t,
                    "confidence": confidence,
                }
            )
        return segments

    def _assign_cluster(self, embedding: np.ndarray, clusters: List[np.ndarray]) -> int:
        if embedding is None or not clusters:
            return len(clusters)
        best_idx = len(clusters)
        best_score = -1.0
        for idx, centroid in enumerate(clusters):
            sim = float(np.dot(embedding, centroid) / ((np.linalg.norm(embedding) * np.linalg.norm(centroid)) + 1e-8))
            if sim > best_score:
                best_score = sim
                best_idx = idx
        if best_score < self._cluster_threshold:
            return len(clusters)
        return best_idx

