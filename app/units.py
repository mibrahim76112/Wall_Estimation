from __future__ import annotations
import math


def parse_inches_per_foot(s: str) -> float:
    s = s.strip().lower().replace('"', "")
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def feet_per_pixel_from_scale(dpi: int, inches_per_foot: float) -> float:
    # pixel = 1/dpi inch on paper
    # inches_per_foot = inches on paper per 1 foot real
    # feet_per_pixel = (1/dpi) / inches_per_foot
    return (1.0 / dpi) / inches_per_foot


def feet_to_arch(ft: float) -> str:
    total_inches = ft * 12.0
    f = int(total_inches // 12)
    inches = total_inches - f * 12.0

    inches = round(inches * 2) / 2  # nearest 1/2 inch

    if inches >= 12:
        f += 1
        inches = 0

    if abs(inches - round(inches)) < 1e-9:
        return f"{f}'-{int(round(inches))}\""

    whole = int(math.floor(inches))
    frac = inches - whole
    if abs(frac - 0.5) < 1e-9:
        return f"{f}'-{whole} 1/2\""
    return f"{f}'-{inches:.2f}\""
