from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

import torch
from google.cloud import storage

from .model_loader import load_cubicasa_model
from .pipeline import estimate_lengths_from_pdf

APP_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = APP_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local path inside container (Cloud Run) for the downloaded weights
LOCAL_WEIGHTS_PATH = Path(os.getenv("WEIGHTS_PATH", "/tmp/model_best_val_loss_var.pkl"))

# GCS settings (you will set these in Cloud Run env vars)
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "")
MODEL_OBJECT = os.getenv("MODEL_OBJECT", "")

app = FastAPI()
model = None


def _download_weights_if_missing() -> None:
    LOCAL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_WEIGHTS_PATH.exists() and LOCAL_WEIGHTS_PATH.stat().st_size > 0:
        return

    
    if not MODEL_BUCKET or not MODEL_OBJECT:
    # Let the container start so Cloud Run can deploy revisions.
    # The endpoint will return a clear error until env vars are fixed.
        return


    client = storage.Client()
    blob = client.bucket(MODEL_BUCKET).blob(MODEL_OBJECT)
    blob.download_to_filename(str(LOCAL_WEIGHTS_PATH))


@app.on_event("startup")
def _startup() -> None:
    global model
    _download_weights_if_missing()
    if not LOCAL_WEIGHTS_PATH.exists():
        print("Weights not downloaded yet (missing MODEL_BUCKET/MODEL_OBJECT).")
        return
    model = load_cubicasa_model(str(LOCAL_WEIGHTS_PATH), device=device)


@app.post("/estimate")
async def estimate(
    pdf: UploadFile = File(...),
    page_index: int = Form(0),
    scale_inch_per_foot: str = Form("3/16"),
):
    if not pdf.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    file_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{file_id}.pdf"
    out_path.write_bytes(await pdf.read())

    debug_dir = str(UPLOAD_DIR / file_id)
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    result = estimate_lengths_from_pdf(
        pdf_path=str(out_path),
        model=model,
        device=device,
        page_index=page_index,
        scale_inch_per_foot=scale_inch_per_foot,
        debug_outputs_dir=debug_dir,
    )
    return result
