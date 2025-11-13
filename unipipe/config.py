from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class StageConfig:
    """Configuration for an individual pipeline stage."""

    enabled: bool = True
    backend: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level configuration object for the unified neural pipeline."""

    sample_rate: int = 16000
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    chunk_duration_s: float = 8.0
    stream_hop_s: float = 0.8
    stream_window_s: float = 3.2
    max_active_speakers: int = 4
    diarization_sensitivity: float = 0.6
    asr_language: Optional[str] = None
    stages: Dict[str, StageConfig] = field(default_factory=dict)


def default_config() -> PipelineConfig:
    """Return a PipelineConfig with sane defaults for the reference implementation."""

    return PipelineConfig(
        stages={
            "endpoint_detection": StageConfig(backend="cam++"),
            "overlap_detection": StageConfig(backend="pyannote"),
            "denoiser": StageConfig(backend="uvr_mdx"),
            "vad": StageConfig(backend="fsmn"),
            "diarization": StageConfig(backend="clustering"),
            "separation": StageConfig(backend="mossformer2", params={"num_sources": 2}),
            "restoration": StageConfig(backend="apollo"),
            "speaker_recognition": StageConfig(backend="eres2netv2"),
            "asr": StageConfig(backend="whisper"),
            "punctuation": StageConfig(backend="ct_transformer"),
        }
    )

