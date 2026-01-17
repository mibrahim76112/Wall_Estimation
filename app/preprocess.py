from __future__ import annotations

from PIL import Image
import numpy as np
import torch


def preprocess_image_rgb(rgb: np.ndarray, target_long_side: int = 1024, pad_value: int = 0):
    """
    Input: rgb image as numpy uint8 [H,W,3]
    Returns:
      orig_rgb, padded_rgb, tensor x [1,3,S,S], (nh, nw) resized shape before padding
    """
    orig_rgb = rgb
    h, w = orig_rgb.shape[:2]
    scale = target_long_side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    resized = np.array(Image.fromarray(orig_rgb).resize((nw, nh), Image.BILINEAR))

    side = max(nh, nw)
    img_pad = np.full((side, side, 3), pad_value, dtype=resized.dtype)
    img_pad[:nh, :nw] = resized

    x = torch.from_numpy(img_pad).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return orig_rgb, img_pad, x, (nh, nw)


def pick_seg_tensor(out):
    import torch

    if torch.is_tensor(out):
        return out

    if hasattr(out, "logits") and torch.is_tensor(out.logits):
        return out.logits

    if isinstance(out, dict):
        for k in ["seg", "seg_logits", "logits", "out", "pred", "y", "mask_logits"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v

    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v
            if hasattr(v, "logits") and torch.is_tensor(v.logits):
                return v.logits
            if isinstance(v, dict):
                for vv in v.values():
                    if torch.is_tensor(vv):
                        return vv

    raise TypeError(f"Could not find a segmentation tensor in output type: {type(out)}")
