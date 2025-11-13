import asyncio
from pathlib import Path
from typing import Optional

from .pipeline import UnifiedNeuralPipeline, PipelineResult

_pipeline: Optional[UnifiedNeuralPipeline] = None


def _get_pipeline() -> UnifiedNeuralPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedNeuralPipeline()
    return _pipeline


async def _process_async(mixture_path: str, target_path: str, out_dir: str) -> PipelineResult:
    pipeline = _get_pipeline()
    return await pipeline.process_to_dir(mixture_path, target_path, out_dir)


async def process_async(mixture_path: str, target_path: str, out_dir: str) -> PipelineResult:
    """Async facade for offline batch processing (for use inside async contexts)."""

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return await _process_async(mixture_path, target_path, out_dir)


def process(mixture_path: str, target_path: str, out_dir: str) -> PipelineResult:
    """Synchronous facade for offline batch processing."""

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_process_async(mixture_path, target_path, out_dir))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
    return result

