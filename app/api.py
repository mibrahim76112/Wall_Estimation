from __future__ import annotations

from pathlib import Path
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

import torch

from .model_loader import load_cubicasa_model
from .pipeline import estimate_lengths_from_pdf

APP_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = APP_DIR / "weights" / "model_best_val_loss_var.pkl"
UPLOAD_DIR = APP_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_cubicasa_model(str(WEIGHTS_PATH), device=device)

app = FastAPI()


@app.post("/estimate")
async def estimate(
    pdf: UploadFile = File(...),
    page_index: int = Form(0),
    scale_inch_per_foot: str = Form("3/16"),
):
    if not pdf.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

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
