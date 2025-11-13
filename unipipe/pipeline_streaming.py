from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio import load_wav
from .pipeline import UnifiedNeuralPipeline, PipelineResult
from .schemas import Segment

logger = logging.getLogger(__name__)


def decode_base64_audio(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload)
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio


@dataclass
class StreamingState:
    target_reference: Optional[np.ndarray] = None
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    last_emitted: float = 0.0
    sample_rate: int = 16000
    session_id: str = ""


class StreamingDiarizationEngine:
    """Manages streaming inference with sliding window processing."""

    def __init__(self, pipeline: Optional[UnifiedNeuralPipeline] = None) -> None:
        self.pipeline = pipeline or UnifiedNeuralPipeline()
        self.lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        async with self.lock:
            await self.pipeline.setup()

    async def enroll_from_file(self, state: StreamingState, reference_path: str) -> None:
        await self.ensure_ready()
        reference, sr = load_wav(reference_path, sr=self.pipeline.config.sample_rate)
        state.target_reference = reference
        state.sample_rate = sr

    async def enroll_from_chunk(self, state: StreamingState, chunk: str, sample_rate: Optional[int] = None) -> None:
        await self.ensure_ready()
        reference = decode_base64_audio(chunk)
        sr = sample_rate or self.pipeline.config.sample_rate
        state.target_reference = reference
        state.sample_rate = sr

    async def push_chunk(
        self,
        state: StreamingState,
        chunk_payload: str,
        sample_rate: Optional[int] = None,
    ) -> Tuple[List[Segment], Dict[str, object]]:
        await self.ensure_ready()
        if state.target_reference is None:
            raise RuntimeError("target speaker not enrolled")

        chunk = decode_base64_audio(chunk_payload)
        sr = sample_rate or state.sample_rate or self.pipeline.config.sample_rate
        state.sample_rate = sr
        if chunk.size == 0:
            return [], {}

        state.buffer = np.concatenate([state.buffer, chunk]).astype(np.float32)
        window_samples = int(self.pipeline.config.stream_window_s * sr)
        if state.buffer.size < window_samples:
            return [], {}

        window_audio = state.buffer[-window_samples:]
        result: PipelineResult = await self.pipeline.process_audio(window_audio, state.target_reference, sr)

        new_segments = []
        for seg in result.segments:
            if seg.end <= state.last_emitted:
                continue
            new_segments.append(seg)
            state.last_emitted = max(state.last_emitted, seg.end)

        # Keep tail of buffer to maintain context for future windows
        hop_samples = int(self.pipeline.config.stream_hop_s * sr)
        if hop_samples <= 0:
            hop_samples = window_samples // 2
        state.buffer = state.buffer[-hop_samples:]

        return new_segments, result.metadata

    async def flush(self, state: StreamingState) -> Tuple[List[Segment], Dict[str, object]]:
        if state.target_reference is None or state.buffer.size == 0:
            return [], {}

        sr = state.sample_rate
        result: PipelineResult = await self.pipeline.process_audio(state.buffer, state.target_reference, sr)
        state.buffer = np.zeros(0, dtype=np.float32)
        state.last_emitted = 0.0
        return result.segments, result.metadata

