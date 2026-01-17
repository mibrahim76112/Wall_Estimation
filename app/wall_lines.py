from __future__ import annotations

import math
import cv2
import numpy as np
from skimage.morphology import skeletonize


def wall_mask_from_pred(pred_small: np.ndarray, wall_label: int, out_w: int, out_h: int) -> np.ndarray:
    wall_small = ((pred_small == wall_label).astype(np.uint8) * 255)
    wall = cv2.resize(wall_small, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    wall = cv2.medianBlur(wall, 3)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall = cv2.dilate(wall, k_dil, iterations=1)
    return wall


def remove_sheet_margins(wall: np.ndarray, remove_left_titleblock: bool = True) -> np.ndarray:
    h, w = wall.shape[:2]
    mx = int(0.06 * w)
    my = int(0.06 * h)
    wall2 = wall.copy()
    wall2[:my, :] = 0
    wall2[-my:, :] = 0
    wall2[:, :mx] = 0
    wall2[:, -mx:] = 0
    if remove_left_titleblock:
        wall2[:, :int(0.18 * w)] = 0
    return wall2


def snap_hv(line, angle_tol=20):
    x1, y1, x2, y2 = map(float, line)
    ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180

    if min(ang, 180 - ang) < angle_tol:
        y = (y1 + y2) / 2
        return [x1, y, x2, y]

    if abs(ang - 90) < angle_tol:
        x = (x1 + x2) / 2
        return [x, y1, x, y2]

    return None


def normalize_line(l):
    x1, y1, x2, y2 = map(float, l)
    if abs(y2 - y1) < abs(x2 - x1):
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
    else:
        if y2 < y1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def is_horizontal(l, tol=2.5):
    x1, y1, x2, y2 = l
    return abs(y2 - y1) <= tol and abs(x2 - x1) > 20


def is_vertical(l, tol=2.5):
    x1, y1, x2, y2 = l
    return abs(x2 - x1) <= tol and abs(y2 - y1) > 20


def merge_1d_intervals(items, gap=70):
    if not items:
        return []
    items = sorted(items, key=lambda t: t[1])
    out = []
    fc, a, b = items[0]
    for fc2, a2, b2 in items[1:]:
        if a2 <= b + gap:
            b = max(b, b2)
            fc = (fc + fc2) / 2.0
        else:
            out.append((fc, a, b))
            fc, a, b = fc2, a2, b2
    out.append((fc, a, b))
    return out


def merge_axis_aligned(lines_in, band=22, gap=70):
    lines_in = [normalize_line(l) for l in lines_in]
    hs = [l for l in lines_in if is_horizontal(l)]
    vs = [l for l in lines_in if is_vertical(l)]

    merged = []

    hs = sorted(hs, key=lambda l: l[1])
    used = [False] * len(hs)
    for i in range(len(hs)):
        if used[i]:
            continue
        y_ref = hs[i][1]
        cluster = []
        for j in range(i, len(hs)):
            if used[j]:
                continue
            if abs(hs[j][1] - y_ref) <= band:
                x1, y1, x2, y2 = hs[j]
                cluster.append((y1, min(x1, x2), max(x1, x2)))
                used[j] = True
        for y, x1, x2 in merge_1d_intervals(cluster, gap=gap):
            merged.append([x1, y, x2, y])

    vs = sorted(vs, key=lambda l: l[0])
    used = [False] * len(vs)
    for i in range(len(vs)):
        if used[i]:
            continue
        x_ref = vs[i][0]
        cluster = []
        for j in range(i, len(vs)):
            if used[j]:
                continue
            if abs(vs[j][0] - x_ref) <= band:
                x1, y1, x2, y2 = vs[j]
                cluster.append((x1, min(y1, y2), max(y1, y2)))
                used[j] = True
        for x, y1, y2 in merge_1d_intervals(cluster, gap=gap):
            merged.append([x, y1, x, y2])

    return merged


def filter_lines_on_wall(lines_in, wall_mask_255, dist_tol=7.0, keep_ratio=0.55, samples=90):
    inv = (255 - wall_mask_255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

    kept = []
    h, w = wall_mask_255.shape[:2]
    for l in lines_in:
        x1, y1, x2, y2 = map(float, l)
        ok = 0
        for t in np.linspace(0, 1, samples):
            x = int(round(x1 + (x2 - x1) * t))
            y = int(round(y1 + (y2 - y1) * t))
            if 0 <= x < w and 0 <= y < h and dist[y, x] <= dist_tol:
                ok += 1

        length_px = math.hypot(x2 - x1, y2 - y1)
        kr = keep_ratio if length_px > 120 else 0.45
        if ok / samples >= kr:
            kept.append(l)

    return kept


def dedup_overlapping_lines(lines_in, band=22, overlap_gap=25):
    def is_h(l):
        x1, y1, x2, y2 = l
        return abs(y2 - y1) <= 2.5

    def is_v(l):
        x1, y1, x2, y2 = l
        return abs(x2 - x1) <= 2.5

    def length(l):
        x1, y1, x2, y2 = l
        return math.hypot(x2 - x1, y2 - y1)

    norm = []
    for l in lines_in:
        x1, y1, x2, y2 = map(float, l)
        if is_h([x1, y1, x2, y2]) and x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        if is_v([x1, y1, x2, y2]) and y2 < y1:
            y1, y2 = y2, y1
            x1, x2 = x2, x1
        norm.append([x1, y1, x2, y2])

    used = [False] * len(norm)
    kept = []

    for i in range(len(norm)):
        if used[i]:
            continue
        li = norm[i]
        cluster = [li]
        used[i] = True

        for j in range(i + 1, len(norm)):
            if used[j]:
                continue
            lj = norm[j]

            if is_h(li) and is_h(lj):
                yi = (li[1] + li[3]) / 2
                yj = (lj[1] + lj[3]) / 2
                if abs(yi - yj) > band:
                    continue
                a1, a2 = li[0], li[2]
                b1, b2 = lj[0], lj[2]
                if not (b1 > a2 + overlap_gap or a1 > b2 + overlap_gap):
                    cluster.append(lj)
                    used[j] = True

            elif is_v(li) and is_v(lj):
                xi = (li[0] + li[2]) / 2
                xj = (lj[0] + lj[2]) / 2
                if abs(xi - xj) > band:
                    continue
                a1, a2 = li[1], li[3]
                b1, b2 = lj[1], lj[3]
                if not (b1 > a2 + overlap_gap or a1 > b2 + overlap_gap):
                    cluster.append(lj)
                    used[j] = True

        kept.append(max(cluster, key=length))

    return kept


def extract_wall_lines(wall_255: np.ndarray):
    skel = (skeletonize(wall_255 > 0).astype(np.uint8) * 255)

    lines = cv2.HoughLinesP(
        skel,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=60,
        maxLineGap=140,
    )
    if lines is None:
        return []

    lines = lines.reshape(-1, 4)
    snapped = [s for s in (snap_hv(l, angle_tol=20) for l in lines) if s is not None]
    merged = merge_axis_aligned(snapped, band=22, gap=70)

    wall_for_check = cv2.dilate(wall_255, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)
    final_lines = filter_lines_on_wall(merged, wall_for_check, dist_tol=7.0, keep_ratio=0.55, samples=90)
    final_lines = dedup_overlapping_lines(final_lines, band=22, overlap_gap=25)
    return final_lines
