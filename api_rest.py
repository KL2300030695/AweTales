from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import shutil
import traceback
import numpy as np
from unipipe.pipeline_offline import process_async as offline_process_async

app = FastAPI()


def _to_jsonable(obj):
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        from pydantic import BaseModel  # type: ignore

        if isinstance(obj, BaseModel):
            return obj.model_dump()
    except Exception:
        pass
    return obj

@app.get("/", response_class=HTMLResponse)
async def root_page():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>Unified Neural Pipeline</title>
      <style>
        body{font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px}
        .card{border:1px solid #ddd; border-radius:8px; padding:16px}
        .row{display:flex; gap:12px; align-items:center}
        label{min-width:160px}
        pre{background:#111; color:#0f0; padding:12px; border-radius:8px; overflow:auto}
        button{padding:8px 14px}
      </style>
    </head>
    <body>
      <h1>Unified Neural Pipeline</h1>
      <p>Use this page to test the batch API. For interactive API docs, visit <a href=\"/docs\">/docs</a>.</p>
      <div class=\"card\">
        <h3>Batch Process</h3>
        <div class=\"row\">
          <label>Mixture Audio (WAV):</label>
          <input id=\"mix\" type=\"file\" accept=\"audio/wav\" />
        </div>
        <div class=\"row\" style=\"margin-top:8px\">
          <label>Target Sample (WAV):</label>
          <input id=\"tgt\" type=\"file\" accept=\"audio/wav\" />
        </div>
        <div class=\"row\" style=\"margin-top:12px\">
          <button id=\"run\">Run /process</button>
        </div>
        <div style=\"margin-top:12px\">
          <strong>Result:</strong>
          <pre id=\"out\"></pre>
          <audio id=\"player\" controls></audio>
        </div>
      </div>
      <script>
        const btn = document.getElementById('run');
        const out = document.getElementById('out');
        const player = document.getElementById('player');
        btn.onclick = async () => {
          const mix = document.getElementById('mix').files[0];
          const tgt = document.getElementById('tgt').files[0];
          if(!mix || !tgt){ out.textContent = 'Please select both files.'; return; }
          const fd = new FormData();
          fd.append('mixture_audio', mix);
          fd.append('target_sample', tgt);
          out.textContent = 'Uploading...';
          try{
            const r = await fetch('/process', { method: 'POST', body: fd });
            const ct = r.headers.get('content-type') || '';
            if(ct.includes('application/json')){
              const j = await r.json();
              out.textContent = JSON.stringify(j, null, 2);
            }else{
              const t = await r.text();
              out.textContent = t;
            }
            if(r.ok){
              // The API returns a filesystem path; expose via /outputs if needed.
              // For now, best effort: try to load if served statically later.
            }
          }catch(e){ out.textContent = String(e); }
        };
      </script>
    </body>
    </html>
    """

@app.post("/process")
async def process_endpoint(
    mixture_audio: UploadFile = File(...),
    target_sample: UploadFile = File(...),
):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    temp_dir = out_dir / "uploads"
    temp_dir.mkdir(exist_ok=True)

    mix_path = temp_dir / mixture_audio.filename
    tgt_path = temp_dir / target_sample.filename

    with open(mix_path, "wb") as f:
        shutil.copyfileobj(mixture_audio.file, f)
    with open(tgt_path, "wb") as f:
        shutil.copyfileobj(target_sample.file, f)
    try:
        result = await offline_process_async(str(mix_path), str(tgt_path), str(out_dir))
        payload = [seg.model_dump() for seg in result.segments]
        excluded = {"target_audio_path", "diarization_path", "timeline_path"}
        response = {
            "diarization": payload,
            "target_audio_path": result.metadata.get("target_audio_path"),
            "diarization_path": result.metadata.get("diarization_path"),
            "timeline_path": result.metadata.get("timeline_path"),
            "metadata": _to_jsonable({k: v for k, v in result.metadata.items() if k not in excluded}),
        }
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse(
            {
                "error": "processing_failed",
                "message": str(e),
            },
            status_code=500,
        )
