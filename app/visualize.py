from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def save_lines_overlay(
    page_rgb: np.ndarray,
    lines: list,
    out_path: str,
    pdf_alpha: float = 0.45,
    line_thickness: int = 3,
):
    """
    Draws detected lines on the page and saves as PNG (or PDF if you pass .pdf).
    """
    base = (page_rgb.astype(np.float32) * pdf_alpha).astype(np.uint8)
    overlay = base.copy()

    for l in lines:
        x1, y1, x2, y2 = map(float, l)
        X1, Y1, X2, Y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.line(overlay, (X1, Y1), (X2, Y2), (255, 0, 0), line_thickness)

    out_path = str(out_path)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    if out_path.lower().endswith(".pdf"):
        Image.fromarray(overlay).save(out_path, "PDF")
    else:
        Image.fromarray(overlay).save(out_path)
    return out_path


def save_outer_contour_overlay(
    page_rgb: np.ndarray,
    contour,
    out_path: str,
    pdf_alpha: float = 0.45,
    border_thickness: int = 6,
):
    """
    Draws the outer contour on the page and saves as PNG (or PDF).
    """
    base = (page_rgb.astype(np.float32) * pdf_alpha).astype(np.uint8)
    overlay = base.copy()

    if contour is not None:
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, [contour], -1, (0, 255, 255), border_thickness)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    out_path = str(out_path)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    if out_path.lower().endswith(".pdf"):
        Image.fromarray(overlay).save(out_path, "PDF")
    else:
        Image.fromarray(overlay).save(out_path)
    return out_path
