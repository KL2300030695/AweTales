# Unified Neural Pipeline

End‑to‑end audio pipeline for target‑speaker extraction, diarization, and ASR with a simple REST UI, WebSocket API, and CLI.

## Features
- REST API with a minimal upload UI at `/`
- WebSocket streaming API (prototype)
- CLI for batch processing
- Outputs written to `outputs/`:
  - `target_speaker.wav`
  - `diarization.json`
  - `timeline.png`

## Requirements
- Windows, Python 3.10+ recommended
- FFmpeg is not strictly required for the default path, but some libraries (e.g., librosa) can use it when handling non‑WAV files

## Project structure
```
Awetales/
  api_rest.py                # FastAPI REST app (UI + /process)
  api_ws.py                  # FastAPI WebSocket app (/ws/session)
  cli.py                     # CLI entry point
  requirements.txt           # Base dependencies
  unipipe/                   # Pipeline package
    pipeline.py              # Core pipeline orchestration
    pipeline_offline.py      # Offline facade (sync + async)
    stages/                  # Individual stages (VAD, diar, ASR, ...)
    config.py                # Defaults and knobs
    schemas.py               # Pydantic models
    visualization.py         # Timeline plot
  outputs/                   # Generated results (created at runtime)
```

## Quick start (REST, recommended)
1) Create virtual environment and install deps
```
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2) Optional: install a real ASR backend (better transcripts)
```
.\.venv\Scripts\python -m pip install faster-whisper
```

3) Run the REST server
```
.\.venv\Scripts\python -m uvicorn api_rest:app --host 127.0.0.1 --port 8000 --reload
```
Then open:
- UI: http://127.0.0.1:8000/
- Docs: http://127.0.0.1:8000/docs

4) Use the UI to select two WAV files and click "Run /process"
- Mixture Audio (WAV): audio that contains the target speaker mixed with other sounds
- Target Sample (WAV): a short clean-ish recording of the target speaker

Results will appear on the page and on disk under `outputs/`.

## Alternative: WebSocket server
```
.\.venv\Scripts\python -m uvicorn api_ws:app --host 127.0.0.1 --port 8000 --reload
```
- Connect a WS client to `ws://127.0.0.1:8000/ws/session`
- Messages: `enroll`, `audio_chunk`, `flush`, `reset`

## CLI usage
Run a batch process without the server:
```
.\.venv\Scripts\python cli.py process --mixture "D:\\path\\to\\mixture.wav" --target "D:\\path\\to\\target.wav" --out outputs
```
Outputs are written to the `--out` directory.

## REST API details
- POST `/process`
  - multipart form with fields `mixture_audio` and `target_sample` (both WAV)
  - returns JSON with `diarization`, `target_audio_path`, `diarization_path`, `timeline_path`, and `metadata`

PowerShell tip: `curl` is an alias for `Invoke-WebRequest`. Prefer the UI or use Python for requests. Example Python snippet:
```python
import requests
files = {
  'mixture_audio': open(r'D:\path\to\mixture.wav','rb'),
  'target_sample': open(r'D:\path\to\target.wav','rb')
}
r = requests.post('http://127.0.0.1:8000/process', files=files)
print(r.status_code)
print(r.json())
```

## Outputs
- `outputs/target_speaker.wav` – extracted/cleaned target speaker audio
- `outputs/diarization.json` – list of segments with `speaker`, `start`, `end`, `text`, `confidence`, `lang`
- `outputs/timeline.png` – simple timeline plot (if segments exist)

## Configuration and tuning
- Edit `unipipe/config.py` to tweak:
  - `device`, `dtype`, `sample_rate`
  - Stage enable/disable and backend selection
  - Thresholds for VAD/endpoint/overlap
- Separation stage (`unipipe/stages/separation.py`): `similarity_threshold` controls how strict target matching is (default ~0.65). Increase to reduce leakage.
- ASR stage (`unipipe/stages/asr.py`): if `faster-whisper` is installed, it will be used automatically. Otherwise a lightweight fallback produces placeholder text.

## Troubleshooting
- UI shows `Internal Server Error` or JSON parse error
  - The app now returns JSON errors `{ "error": "processing_failed", "message": "..." }` on failures.
  - Check the server console for stack traces.
- "Object of type ndarray is not JSON serializable"
  - Fixed: responses sanitize numpy values before returning.
- Powershell `curl -F ...` errors (`-F` not recognized)
  - Use the UI, or Python `requests`, or run `curl.exe` explicitly.
- No meaningful transcript ("Uh uh ...")
  - Install `faster-whisper` and re-run.
- No `timeline.png`
  - It only renders if at least one segment exists.

## Notes
- For LAN access, bind to all interfaces:
```
.\.venv\Scripts\python -m uvicorn api_rest:app --host 0.0.0.0 --port 8000 --reload
```
- Do not expose the service publicly without proper security.

## License
- Add your license information here.
