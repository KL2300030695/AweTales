from pydantic import BaseModel, Field
from typing import List, Optional

class Segment(BaseModel):
    speaker: str = Field(...)
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    text: str = Field("")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    lang: Optional[str] = None

class DiarizationResult(BaseModel):
    segments: List[Segment]

class ProcessResponse(BaseModel):
    diarization: List[Segment]
    target_audio_path: str
