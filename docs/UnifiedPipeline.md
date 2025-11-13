## Unified Neural Pipeline (v1.3)

### Overview
The unified neural pipeline fuses modular speech enhancement, diarization, and ASR stages to isolate and transcribe a target speaker within a multi-speaker environment. Each stage is pluggable and can leverage state-of-the-art back-ends (CAM++, PyAnnote, UVR-MDX-Net, MossFormer2, ERes2NetV2, Paraformer/Whisper, CT-Transformer) while providing lightweight fallbacks to keep the reference implementation runnable without heavyweight dependencies.

- **Batch mode**: `unipipe.pipeline_offline.process` orchestrates offline processing given `mixture_audio.wav` and `target_sample.wav`, producing `target_speaker.wav`, `diarization.json`, and a `timeline.png` visualization.
- **Streaming mode**: `api_ws.py` exposes a WebSocket endpoint that maintains incremental state per session, delivering sub-second diarization/ASR updates from sliding windows.
- **API integrations**: FastAPI endpoints (REST + WebSocket) wrap the pipeline for deployment in services or UI prototypes.

### Stage Graph
1. **Endpoint Detection (CAM++ / Energy)** – detects active speech spans to hint subsequent stages.
2. **Overlap Detection (PyAnnote / Flatness)** – flags overlapping speech segments.
3. **Audio Denoising (UVR-MDX / Spectral Gate)** – reduces noise and reverberation.
4. **Voice Activity Detection (FSMN / Energy)** – yields precise speech segments.
5. **Target Separation (MossFormer2 / MFCC-Sim Masking)** – extracts target speaker audio.
6. **Restoration (Apollo / EQ & De-essing)** – enhances the separated target stream.
7. **Speaker Recognition (ERes2NetV2 / MFCC Embeddings)** – scores segments against the enrolled target.
8. **Diarization (PyAnnote / Cosine Clustering)** – assigns speaker labels per segment.
9. **ASR (Paraformer / Whisper / Placeholder)** – transcribes each diarized turn.
10. **Punctuation (CT-Transformer / Heuristic)** – restores punctuation and casing.

All stage configurations live in `PipelineConfig`, enabling overrides per deployment (e.g., switch to GPU MossFormer2, point to enterprise authentication tokens, disable expensive stages, etc.).

### Configuration
- `unipipe/config.py` defines `PipelineConfig` and per-stage `StageConfig`.
- Customize via `UnifiedNeuralPipeline(PipelineConfig(...))`; pass explicit back-ends, thresholds, or chunk durations.
- Default configuration defaults to SOTA backends but auto-falls back to heuristic implementations when models are unavailable.

### Offline Usage
```bash
python cli.py process --mixture path/to/mixture.wav --target path/to/target.wav --out outputs/
```
Outputs:
- `outputs/target_speaker.wav`
- `outputs/diarization.json`
- `outputs/timeline.png`
- Console dump of transcripts with confidence/lang codes.

Programmatic usage:
```python
from unipipe.pipeline_offline import process
result = process("mixture_audio.wav", "target_sample.wav", "outputs")
print(result.metadata["target_audio_path"])
for segment in result.segments:
    print(segment.model_dump())
```

### REST API
- `POST /process` accepts multipart (`mixture_audio`, `target_sample`) and returns JSON with diarization segments and artifact paths.
- `GET /` offers a minimal HTML harness; full docs at `/docs`.

### WebSocket Streaming
Endpoint: `ws://<host>/ws/session`

Protocol:
1. Client connects → server responds with `{ "type": "ready", "session_id", "sr" }`.
2. Enroll target speaker:
   ```json
   {"type": "enroll", "chunk": "<base64-float32>", "sample_rate": 16000}
   ```
3. Stream audio chunks:
   ```json
   {"type": "audio_chunk", "chunk": "<base64-float32>", "sample_rate": 16000}
   ```
4. Receive partial updates: `{ "type": "partial_result", "segments": [...], "metadata": {...} }`
5. Optional flush/reset commands.

### Visualization
`unipipe/visualization.py` renders a speaker timeline PNG summarizing diarized turns, annotate transcripts inline, and stores it beside other artifacts.

### Dependencies
| Category | Packages (core) | Optional Backends |
|----------|-----------------|-------------------|
| Core | `fastapi`, `librosa`, `numpy`, `soundfile`, `scikit-learn`, `matplotlib` | |
| Separation / ASR | – | `faster-whisper`, `whisper`, `funasr`, `pyannote.audio`, `modelscope`, `camplusplus`, `uvr`, `apollo-infer` |

Install optional packages selectively to activate higher-quality models. Fallbacks keep the pipeline functional for prototyping and unit tests.

### Testing & Validation
- Unit tests should mock stage outputs and verify JSON serialization, streaming incremental state, and CLI interactions.
- Integration tests can operate on short audio fixtures (≤10 s) to validate artifact generation.

### Future Enhancements
- Replace heuristic fallbacks with lightweight neural models (e.g., `silero-vad`, `torchdenoiser`).
- Support GPU batching for concurrent sessions through configurable worker pools.
- Add confidence calibration and multilingual language ID heuristics.

