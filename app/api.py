from __future__ import annotations

import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from google.cloud import storage
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_413_REQUEST_ENTITY_TOO_LARGE

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

# Limits
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(10 * 1024 * 1024)))  # default 10 MB
MAX_PAGES = int(os.getenv("MAX_PDF_PAGES", "3"))

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


async def _save_upload_with_limit(upload: UploadFile, dst_path: Path, max_bytes: int) -> int:
    """
    Streams the upload to disk and enforces a max byte limit.
    Returns the number of bytes written.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    try:
        with open(dst_path, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("too_large")
                f.write(chunk)
        return total
    except Exception:
        try:
            if dst_path.exists():
                dst_path.unlink()
        except OSError:
            pass
        raise
    finally:
        try:
            await upload.close()
        except Exception:
            pass


def _validate_pdf_pages(pdf_path: Path, max_pages: int) -> int:
    """
    Opens the saved PDF and validates page count.
    Returns the page count.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Invalid PDF: {e}") from e

    try:
        page_count = len(doc)
        if page_count < 1:
            raise ValueError("PDF has no pages.")
        if page_count > max_pages:
            raise ValueError(f"Too many pages. Max allowed is {max_pages}.")
        return page_count
    finally:
        doc.close()


@app.post("/estimate")
async def estimate(
    pdf: UploadFile = File(...),
    page_index: int = Form(0),
    scale_inch_per_foot: str = Form("3/16"),
):
    if not pdf.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported"}, status_code=HTTP_400_BAD_REQUEST)

    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    file_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{file_id}.pdf"

    # 1) Enforce upload size limit while streaming to disk
    try:
        await _save_upload_with_limit(pdf, out_path, MAX_PDF_BYTES)
    except ValueError as e:
        if str(e) == "too_large":
            return JSONResponse(
                {"error": f"PDF too large. Max allowed is {MAX_PDF_BYTES // (1024 * 1024)} MB."},
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        return JSONResponse({"error": str(e)}, status_code=HTTP_400_BAD_REQUEST)

    # 2) Enforce max pages = 3
    try:
        page_count = _validate_pdf_pages(out_path, MAX_PAGES)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=HTTP_400_BAD_REQUEST)

    # 3) Validate page_index against page_count
    if page_index < 0 or page_index >= page_count:
        return JSONResponse(
            {"error": f"Invalid page_index={page_index}. PDF has {page_count} pages."},
            status_code=HTTP_400_BAD_REQUEST,
        )

    debug_dir = UPLOAD_DIR / file_id
    debug_dir.mkdir(parents=True, exist_ok=True)

    result = estimate_lengths_from_pdf(
        pdf_path=str(out_path),
        model=model,
        device=device,
        page_index=page_index,
        scale_inch_per_foot=scale_inch_per_foot,
        debug_outputs_dir=str(debug_dir),
    )
    return result
