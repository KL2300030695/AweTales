from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StagePayload:
    """Represents audio buffers travelling through the pipeline."""

    mixture: np.ndarray
    sample_rate: int
    target_reference: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StageOutput:
    """Generic stage output container."""

    payload: StagePayload
    data: Dict[str, Any]


@dataclass
class StageContext:
    """Shared context passed to each stage, allowing cross-stage communication."""

    cache: Dict[str, Any]
    device: str
    dtype: str

    def get(self, key: str, default: Any = None) -> Any:
        return self.cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value


class BaseStage(abc.ABC):
    """Base class for all pipeline stages."""

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self._ready_event = asyncio.Event()

    async def setup(self, context: StageContext) -> None:
        if not self.enabled:
            self._ready_event.set()
            return

        try:
            await self._setup_impl(context)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to setup stage %s", self.name)
            raise
        finally:
            self._ready_event.set()

    async def _setup_impl(self, context: StageContext) -> None:  # pragma: no cover - to override
        return None

    async def wait_ready(self) -> None:
        await self._ready_event.wait()

    async def run(self, payload: StagePayload, context: StageContext) -> StageOutput:
        if not self.enabled:
            return StageOutput(payload, data={})

        await self.wait_ready()

        try:
            data = await self._run_impl(payload, context)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Stage %s failed during execution", self.name)
            raise
        return StageOutput(payload, data=data)

    @abc.abstractmethod
    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        ...

    def _log_backend_choice(self, backend: str, ready: bool, extra: Optional[Dict[str, Any]] = None) -> None:
        meta = {"backend": backend, "stage": self.name, **(extra or {})}
        logger.info("Stage %s backend=%s ready=%s", self.name, backend, ready)

    @staticmethod
    def _ensure_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2:
            return np.mean(audio, axis=0)
        raise ValueError("Audio tensor must be mono or stereo")

    @staticmethod
    def _normalize(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        peak = np.max(np.abs(audio)) + eps
        return audio / peak

    @staticmethod
    def _chunk_audio(audio: np.ndarray, sample_rate: int, chunk_duration: float) -> Tuple[np.ndarray, int]:
        samples = int(chunk_duration * sample_rate)
        if samples <= 0:
            raise ValueError("chunk_duration must be positive")
        total = len(audio)
        n_chunks = max(total // samples, 1)
        trimmed = audio[: n_chunks * samples]
        return trimmed.reshape(n_chunks, samples), n_chunks

