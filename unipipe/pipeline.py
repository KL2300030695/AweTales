from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .audio import TARGET_SR, load_wav, save_wav, normalize
from .config import PipelineConfig, StageConfig, default_config
from .schemas import Segment
from .visualization import render_timeline
from .stages import (
    ASRStage,
    BaseStage,
    DenoiserStage,
    DiarizationStage,
    EndpointDetectionStage,
    OverlapDetectionStage,
    PunctuationStage,
    RestorationStage,
    SeparationStage,
    SpeakerRecognitionStage,
    StageContext,
    StagePayload,
    VoiceActivityDetectionStage,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    target_audio: np.ndarray
    sample_rate: int
    segments: List[Segment]
    metadata: Dict[str, object] = field(default_factory=dict)


class UnifiedNeuralPipeline:
    """Coordinates individual stages into a unified diarization + ASR pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or default_config()
        self.device = self._resolve_device(self.config.device)
        self.dtype = self.config.dtype if self.config.dtype != "auto" else "float32"
        self._stages = self._build_stages(self.config.stages or {})
        self._setup_future: Optional[asyncio.Future] = None

    def _build_stages(self, stage_config: Dict[str, StageConfig]) -> List[BaseStage]:
        stage_order = [
            ("endpoint_detection", EndpointDetectionStage),
            ("overlap_detection", OverlapDetectionStage),
            ("denoiser", DenoiserStage),
            ("vad", VoiceActivityDetectionStage),
            ("separation", SeparationStage),
            ("restoration", RestorationStage),
            ("speaker_recognition", SpeakerRecognitionStage),
            ("diarization", DiarizationStage),
            ("asr", ASRStage),
            ("punctuation", PunctuationStage),
        ]

        stages: List[BaseStage] = []
        for name, stage_cls in stage_order:
            cfg = stage_config.get(name, StageConfig())
            params = dict(cfg.params)
            if cfg.backend is not None:
                params["backend"] = cfg.backend
            stage = stage_cls(enabled=cfg.enabled, **params)
            stages.append(stage)
        return stages

    def _resolve_device(self, device_hint: str) -> str:
        if device_hint == "auto":
            try:
                import torch  # type: ignore

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device_hint

    async def setup(self) -> None:
        if self._setup_future is None:
            loop = asyncio.get_running_loop()
            self._setup_future = loop.create_future()
            context = StageContext(cache={}, device=self.device, dtype=self.dtype)
            try:
                for stage in self._stages:
                    await stage.setup(context)
                self._setup_future.set_result(True)
            except Exception as exc:
                self._setup_future.set_exception(exc)
                raise
        if self._setup_future is not None:
            await self._setup_future

    async def process_audio(
        self,
        mixture_audio: np.ndarray,
        target_audio: np.ndarray,
        sample_rate: int,
    ) -> PipelineResult:
        await self.setup()
        context = StageContext(cache={}, device=self.device, dtype=self.dtype)
        payload = StagePayload(
            mixture=mixture_audio.astype(np.float32),
            sample_rate=sample_rate,
            target_reference=target_audio.astype(np.float32),
        )

        for stage in self._stages:
            await stage.run(payload, context)

        target_clean = context.get("restored_audio", context.get("target_audio", payload.mixture))
        transcripts = context.get("transcripts", [])
        segments = [Segment(**seg) if not isinstance(seg, Segment) else seg for seg in transcripts]
        metadata = {
            "speech_regions": context.get("speech_regions"),
            "overlap_regions": context.get("overlap_regions"),
            "speaker_similarity": context.get("speaker_similarity"),
        }
        return PipelineResult(target_audio=target_clean, sample_rate=sample_rate, segments=segments, metadata=metadata)

    async def process_files(self, mixture_path: str, target_path: str) -> PipelineResult:
        mixture, sr = load_wav(mixture_path, sr=self.config.sample_rate or TARGET_SR)
        target, _ = load_wav(target_path, sr=sr)
        return await self.process_audio(mixture, target, sr)

    async def process_to_dir(self, mixture_path: str, target_path: str, out_dir: str) -> PipelineResult:
        result = await self.process_files(mixture_path, target_path)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        target_wav = out_path / "target_speaker.wav"
        save_wav(str(target_wav), normalize(result.target_audio), result.sample_rate)
        diar_path = out_path / "diarization.json"
        with diar_path.open("w", encoding="utf-8") as f:
            json.dump(
                [seg.model_dump(mode="json") for seg in result.segments],
                f,
                indent=2,
            )
        timeline_path = None
        if result.segments:
            timeline_path = render_timeline(result.segments, str(out_path / "timeline.png"))
        return PipelineResult(
            target_audio=result.target_audio,
            sample_rate=result.sample_rate,
            segments=result.segments,
            metadata={
                **result.metadata,
                "target_audio_path": str(target_wav),
                "diarization_path": str(diar_path),
                "timeline_path": timeline_path,
            },
        )

