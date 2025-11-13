from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseStage, StageContext, StagePayload


class PunctuationStage(BaseStage):
    """CT-Transformer punctuation with heuristic fallback."""

    def __init__(self, backend: str = "ct_transformer", enabled: bool = True, **kwargs: Any) -> None:
        super().__init__("punctuation", enabled=enabled)
        self.backend = backend
        self.kwargs = kwargs
        self._model = None
        self._use_ct_transformer = False

    async def _setup_impl(self, context: StageContext) -> None:
        if self.backend == "ct_transformer":
            try:  # pragma: no cover
                from funasr import AutoModel  # type: ignore

                self._model = AutoModel.from_pretrained("ct-punc", trust_remote_code=True)
                self._use_ct_transformer = True
            except Exception:
                self._use_ct_transformer = False
        self._log_backend_choice(self.backend, self._use_ct_transformer)

    async def _run_impl(self, payload: StagePayload, context: StageContext) -> Dict[str, Any]:
        transcripts: List[Dict[str, Any]] = context.get("transcripts", [])
        if not transcripts:
            return {"transcripts": []}

        processed: List[Dict[str, Any]] = []
        for seg in transcripts:
            text = seg.get("text", "")
            if self._use_ct_transformer and self._model is not None:  # pragma: no cover
                result = self._model.generate(text)
                if isinstance(result, list) and result:
                    text = result[0]
                elif isinstance(result, dict):
                    text = result.get("text", text)
            else:
                text = self._fallback_punctuate(text)
            processed.append({**seg, "text": text})

        context.set("transcripts", processed)
        return {"transcripts": processed}

    def _fallback_punctuate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        if text[-1] not in ".?!":
            text = f"{text.capitalize()}."
        return text

