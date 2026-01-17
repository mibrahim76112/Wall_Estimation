from __future__ import annotations
import cv2
import numpy as np


def remove_border_touching_components(bin255: np.ndarray) -> np.ndarray:
    bin01 = (bin255 > 0).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    h, w = bin01.shape
    out = np.zeros_like(bin01)

    for cid in range(1, num):
        x, y, ww, hh, _area = stats[cid]
        touches = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
        if not touches:
            out[lab == cid] = 1

    return (out * 255).astype(np.uint8)


def fill_holes(bin255: np.ndarray) -> np.ndarray:
    h, w = bin255.shape
    flood = bin255.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(bin255, holes)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def get_building_outer_contour(
    wall255: np.ndarray,
    close_k: int | None = None,
    close_iter: int = 2,
    pre_dilate_k: int | None = None,
    pre_dilate_iter: int = 2,
):
    """
    Builds a solid building blob from wall mask and returns largest external contour.

    The pre-dilation helps merge "double-line" exterior walls into one solid region.
    Kernel sizes are chosen adaptively if not provided.
    """
    h, w = wall255.shape[:2]
    m = min(h, w)

    # adaptive defaults
    if pre_dilate_k is None:
        pre_dilate_k = _odd(max(9, int(0.006 * m)))   # ~0.6% of min dimension
    if close_k is None:
        close_k = _odd(max(61, int(0.03 * m)))        # ~3% of min dimension

    img = wall255.copy()

    # 1) dilate to merge parallel wall lines
    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (pre_dilate_k, pre_dilate_k))
    img = cv2.dilate(img, kd, iterations=pre_dilate_iter)

    # 2) strong close to seal gaps
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kc, iterations=close_iter)

    # 3) fill interior to make a solid building region
    img = fill_holes((img > 0).astype(np.uint8) * 255)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, img
    c = max(contours, key=cv2.contourArea)
    return c, img
