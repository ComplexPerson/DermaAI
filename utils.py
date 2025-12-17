"""Small utility helpers used by training/prediction scripts.

This module keeps minimal, dependency-free helpers so other files
don't import an empty module.
"""

import os
import torch
import numpy as np


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: dict, path: str) -> None:
    """Save a checkpoint to `path`, creating parent directories if needed."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Load a checkpoint saved with :func:`save_checkpoint`. Returns the raw object from torch.load."""
    return torch.load(path, map_location=map_location)


__all__ = [
    "ensure_dir",
    "save_checkpoint",
    "load_checkpoint",
    "build_model",
    "load_model_state",
    "find_latest_checkpoint",
    "mixup_data",
    "cutmix_data",
    "DISEASE_MAP",
]


DISEASE_MAP = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}



def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        # keep the same return signature as the normal case
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        # keep the same return signature as the normal case
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, C, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    rx = np.random.randint(W)
    ry = np.random.randint(H)
    rw = int(W * np.sqrt(1 - lam))
    rh = int(H * np.sqrt(1 - lam))
    x1 = int(np.clip(rx - rw // 2, 0, W))
    y1 = int(np.clip(ry - rh // 2, 0, H))
    x2 = int(np.clip(rx + rw // 2, 0, W))
    y2 = int(np.clip(ry + rh // 2, 0, H))
    # perform cutmix by copying a patch from another sample
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def find_latest_checkpoint(checkpoints_dir="checkpoints"):
    """Find the latest modified checkpoint file in a directory."""
    if not os.path.exists(checkpoints_dir):
        return None
    files = [
        os.path.join(checkpoints_dir, f)
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".pt") or f.endswith(".pth")
    ]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


def build_model(num_classes, backbone="resnet34", pretrained=True):
    """Build a model given `backbone` identifier.

    Supported backbones:
      - 'resnet34', 'resnet50' (torchvision)
      - 'timm:<name>' (requires timm)
    Returns an unwrapped torch.nn.Module ready for .to(device).
    """
    import torch.nn as nn
    from torchvision import models

    if isinstance(backbone, str) and backbone.startswith("timm:"):
        try:
            import timm
        except Exception as e:
            raise RuntimeError(
                "timm not installed. Install with: pip install timm"
            ) from e
        model = timm.create_model(
            backbone.split(":", 1)[1], pretrained=pretrained, num_classes=num_classes
        )
        return model

    if backbone == "resnet50":
        m = models.resnet50(pretrained=pretrained)
    else:
        # default to resnet34
        m = models.resnet34(pretrained=pretrained)

    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m


def load_model_state(model, ckpt_path, map_location="cpu"):
    """Load checkpoint state dict into model. Supports checkpoints that wrap state under 'model_state'."""
    import torch

    ck = torch.load(ckpt_path, map_location=map_location)
    state = ck.get("model_state", ck)
    model.load_state_dict(state)
    return model
