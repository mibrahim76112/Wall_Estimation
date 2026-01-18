Wall Length Estimator – Backend API

This repository contains the backend service for estimating wall lengths from architectural floor plan PDFs.

⚠️ IMPORTANT
The pretrained model weight file is NOT included in this GitHub repository.
It must be downloaded manually and placed correctly, otherwise the backend will not run.

This backend is designed to be consumed by a website frontend (React, Next.js, etc.).

What This Backend Does

Accepts a floor plan PDF

Converts the selected PDF page into an image

Runs a pretrained CubiCasa segmentation model (wall label 23 only)

Detects wall geometry

Computes:

Total wall length

Outer perimeter length

Inner wall length = total − outer

Returns results as JSON

Saves debug images for visual verification

Tech Stack

FastAPI (REST API)

PyTorch (model inference)

PyMuPDF (PDF rendering, no Poppler required)

OpenCV + scikit-image (geometry processing)

Uvicorn (development server)

Repository Structure
Wall_length_estimater/
│
├── app/
│   ├── api.py              # FastAPI endpoint
│   ├── pipeline.py         # Core wall-length logic
│   ├── model_loader.py     # Loads pretrained model
│   ├── pdf_render.py       # PDF → image (PyMuPDF)
│   ├── preprocess.py
│   ├── wall_lines.py
│   ├── outer_contour.py
│   ├── visualize.py        # Debug overlays
│   └── units.py
│
├── vendor/
│   └── floortrans/         # CubiCasa model architecture code
│
├── weights/                # NOT included in GitHub
│   └── model_best_val_loss_var.pkl
│
├── outputs/                # Auto-generated per request
│   └── <uuid>/
│
├── requirements.txt
└── README.md

REQUIRED MANUAL STEP (VERY IMPORTANT)
1. Download Model Weights (NOT in GitHub)

This project WILL NOT RUN without the pretrained model weights.

You must download the file:

model_best_val_loss_var.pkl


Download link (Google Drive):
https://drive.google.com/file/d/13Mi2SkrQTqUYUcMlTlMVEOuQMDzyZb_R/view?usp=sharing

After downloading, place the file exactly at:

weights/model_best_val_loss_var.pkl


If this file is missing or named incorrectly, the server will crash on startup.

2. Required Vendor Code Fix (Already Applied)

The original CubiCasa code attempts to load this file:

floortrans/models/model_1427.pth


That file is NOT required and NOT included.

Ensure the following line is disabled in:

vendor/floortrans/models/init.py

# model.init_weights()


If CubiCasa is re-downloaded or updated, this fix must be reapplied.

Installation (Windows)
1. Clone the Repository
git clone <your-github-repo-url>
cd Wall_length_estimater

2. Create Virtual Environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1


If activation is blocked:

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

3. Install Dependencies
pip install -r requirements.txt


No Poppler installation is required.
PDF rendering uses PyMuPDF.

Running the Backend
uvicorn app.api:app --reload --port 8000


API URL:
http://127.0.0.1:8000

Swagger UI:
http://127.0.0.1:8000/docs

API Endpoint (For Website Integration)
POST /estimate

Form Data Parameters

Field	Type	Required	Description
pdf	File	Yes	Floor plan PDF
page_index	Integer	No	Page number (default: 0)
scale_inch_per_foot	String	No	Example: 3/16
Example API Response
{
  "total_ft": 312.45,
  "outer_ft": 148.20,
  "inner_ft": 164.25,
  "total_arch": "312'-6\"",
  "outer_arch": "148'-2\"",
  "inner_arch": "164'-4\"",
  "lines_overlay_path": "outputs/uuid/lines_overlay.png",
  "outer_overlay_path": "outputs/uuid/outer_overlay.png"
}

Debug Images (Manual Verification)

For every request, the backend saves:

outputs/<uuid>/lines_overlay.png
outputs/<uuid>/outer_overlay.png


These images show:

The exact wall lines used for calculation

The detected outer building boundary

They are useful for QA and validation and can optionally be displayed in the frontend.

How the Website Should Use This API

User uploads a PDF

User enters drawing scale (example: 3/16)

Frontend sends POST request to /estimate

Backend returns JSON

Frontend displays:

Total wall length

Outer perimeter length

Inner wall length

Optional debug overlay images