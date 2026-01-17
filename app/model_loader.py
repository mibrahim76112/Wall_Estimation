from __future__ import annotations

import sys
from pathlib import Path
import torch

# Add vendor/ to Python path so we can import floortrans
ROOT = Path(__file__).resolve().parents[1]  # E:\Sherjeel Project\Wall_length_estimater
sys.path.insert(0, str(ROOT / "vendor"))

from floortrans.models import get_model


def load_cubicasa_model(weights_path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(weights_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("Expected checkpoint dict with key 'model_state'")

    state = ckpt["model_state"]

    if "conv4_.weight" not in state:
        raise ValueError("Missing conv4_.weight in model_state, cannot infer n_classes")

    n_classes = state["conv4_.weight"].shape[0]
    model = get_model("hg_furukawa_original", n_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
