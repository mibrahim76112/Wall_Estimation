import numpy as np
import fitz  # PyMuPDF

def render_pdf_page(pdf_path: str, dpi: int, page_index: int) -> np.ndarray:
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        raise ValueError(f"Invalid page_index={page_index}. PDF has {len(doc)} pages.")

    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return img
