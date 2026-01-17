from __future__ import annotations

from pathlib import Path
import math
import torch
import cv2

from .preprocess import preprocess_image_rgb, pick_seg_tensor
from .pdf_render import render_pdf_page
from .units import parse_inches_per_foot, feet_per_pixel_from_scale, feet_to_arch
from .wall_lines import wall_mask_from_pred, remove_sheet_margins, extract_wall_lines
from .outer_contour import remove_border_touching_components, get_building_outer_contour
from .visualize import save_lines_overlay, save_outer_contour_overlay


# Fixed settings (no user control)
WALL_LABEL = 23
DPI = 300


def estimate_lengths_from_pdf(
    pdf_path: str,
    model: torch.nn.Module,
    device: str,
    page_index: int = 0,
    scale_inch_per_foot: str = "3/16",
    debug_outputs_dir: str | None = None,
):
    # units
    inches_per_foot = parse_inches_per_foot(scale_inch_per_foot)
    fpp = feet_per_pixel_from_scale(DPI, inches_per_foot)

    # render PDF page
    page_rgb = render_pdf_page(pdf_path, dpi=DPI, page_index=page_index)
    h, w = page_rgb.shape[:2]

    # segmentation
    _orig, _pad, x, (nh, nw) = preprocess_image_rgb(page_rgb, target_long_side=1024)
    x = x.to(device)

    with torch.no_grad():
        out = model(x)

    seg = pick_seg_tensor(out)
    logits = seg[0].detach().cpu()
    pred = torch.argmax(logits, dim=0).numpy()
    pred = pred[:nh, :nw]

    # wall mask in page resolution (ONLY label 23)
    wall = wall_mask_from_pred(pred, wall_label=WALL_LABEL, out_w=w, out_h=h)
    wall = remove_sheet_margins(wall, remove_left_titleblock=True)

    # TOTAL wall length from detected wall line segments
    lines = extract_wall_lines(wall)
    total_ft = 0.0
    line_items = []
    for i, l in enumerate(lines, start=1):
        x1, y1, x2, y2 = map(float, l)
        length_px = math.hypot(x2 - x1, y2 - y1)
        length_ft = length_px * fpp
        total_ft += length_ft
        line_items.append(
            {
                "id": i,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length_ft_decimal": length_ft,
                "length_arch": feet_to_arch(length_ft),
            }
        )

    # OUTER perimeter length
    wall_nb = remove_border_touching_components(wall)
    contour, _blob = get_building_outer_contour(wall_nb, close_k=121, close_iter=2)

    if contour is None:
        outer_ft = 0.0
    else:
        perim_px = cv2.arcLength(contour, True)
        outer_ft = perim_px * fpp

    # INNER = TOTAL - OUTER
    inner_ft = total_ft - outer_ft
    if inner_ft < 0:
        inner_ft = 0.0

    # Optional debug overlays
    lines_overlay_path = None
    outer_overlay_path = None
    if debug_outputs_dir:
        debug_dir = Path(debug_outputs_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        lines_overlay_path = save_lines_overlay(
            page_rgb=page_rgb,
            lines=lines,
            out_path=str(debug_dir / "lines_overlay.png"),
        )

        outer_overlay_path = save_outer_contour_overlay(
            page_rgb=page_rgb,
            contour=contour,
            out_path=str(debug_dir / "outer_overlay.png"),
        )

    return {
        "page_index": page_index,
        "scale_inch_per_foot": scale_inch_per_foot,
        "dpi_fixed": DPI,
        "wall_label_fixed": WALL_LABEL,
        "feet_per_pixel": fpp,
        "total_ft": total_ft,
        "outer_ft": outer_ft,
        "inner_ft": inner_ft,
        "total_arch": feet_to_arch(total_ft),
        "outer_arch": feet_to_arch(outer_ft),
        "inner_arch": feet_to_arch(inner_ft),
        "lines": line_items,
        "lines_overlay_path": lines_overlay_path,
        "outer_overlay_path": outer_overlay_path,
    }
