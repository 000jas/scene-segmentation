"""
segmentation.py — Model inference for the Streamlit frontend.

Priority order for loaded model:
  1. DeepLabV3+ ResNet-101  (deeplabv3_best.pth)   — best accuracy
  2. DINOv2 + SegmentationHead (segmentation_head.pth) — fallback

Call load_models() once (cache with @st.cache_resource in app.py),
then predict(models, pil_image) -> np.ndarray of class IDs (H x W).
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ── Resolve project root regardless of cwd ──────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Constants ────────────────────────────────────────────────────────
NUM_CLASSES = 10
W_DINO      = int(((960 / 2) // 14) * 14)   # 476
H_DINO      = int(((540 / 2) // 14) * 14)   # 266
IMG_SIZE_DL = 512

CLASS_NAMES = [
    "Background", "Trees", "Bushes", "Dry Grass", "Rocks",
    "Logs", "Terrain", "Sky", "Landscape", "Other",
]

# Distinct, perceptually separated palette (RGB)
PALETTE = np.array([
    [20,  20,  20 ],   # 0  background  — near-black
    [34,  139, 34 ],   # 1  trees       — forest green
    [0,   200, 80 ],   # 2  bushes      — bright green
    [210, 180, 140],   # 3  dry grass   — sandy tan
    [112, 112, 112],   # 4  rocks       — mid gray
    [101, 67,  33 ],   # 5  logs        — dark brown
    [160, 100, 40 ],   # 6  terrain     — warm brown
    [135, 206, 235],   # 7  sky         — sky blue
    [128, 128, 0  ],   # 8  landscape   — olive
    [148, 0,   211],   # 9  other       — purple
], dtype=np.uint8)

# Traversability for path planning
# 0 = clear, 1 = costly (soft obstacle), 2 = hard obstacle
TRAVERSAL_COST = {
    0: 0,   # background  — clear
    1: 2,   # trees       — hard obstacle
    2: 1,   # bushes      — costly
    3: 0,   # dry grass   — clear
    4: 2,   # rocks       — hard obstacle
    5: 2,   # logs        — hard obstacle
    6: 0,   # terrain     — clear (main traversable surface)
    7: 2,   # sky         — hard obstacle (logical)
    8: 0,   # landscape   — clear
    9: 1,   # other       — treat cautiously
}

# ── DINOv2 SegmentationHead (must match train_segmentation.py) ───────

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, out_channels, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.decoder(x)

# ── Transforms ───────────────────────────────────────────────────────

_dino_tf = T.Compose([
    T.Resize((H_DINO, W_DINO)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_dl_tf = T.Compose([
    T.Resize((IMG_SIZE_DL, IMG_SIZE_DL)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model loading ────────────────────────────────────────────────────

def load_models():
    """
    Returns a dict describing what was loaded. Keys:
      "device", "mode" ("deeplabv3" | "dino"),
      and the model(s) themselves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"device": device}

    dl_path   = os.path.join(ROOT, "deeplabv3_best.pth")
    dino_path = os.path.join(ROOT, "segmentation_head.pth")

    # ── Try DeepLabV3+ first ─────────────────────────────────────────
    if os.path.exists(dl_path):
        try:
            import segmentation_models_pytorch as smp
            model = smp.DeepLabV3Plus(
                encoder_name="resnet101",
                encoder_weights=None,
                in_channels=3,
                classes=NUM_CLASSES,
            ).to(device)
            model.load_state_dict(
                torch.load(dl_path, map_location=device, weights_only=True)
            )
            model.eval()
            models["mode"]    = "deeplabv3"
            models["deeplabv3"] = model
            print(f"[segmentation] Loaded DeepLabV3+ from {dl_path}")
            return models
        except Exception as e:
            print(f"[segmentation] DeepLabV3+ load failed: {e}  — falling back to DINOv2")

    # ── Fall back to DINOv2 head ─────────────────────────────────────
    if not os.path.exists(dino_path):
        raise FileNotFoundError(
            "No segmentation model found. "
            "Train one first:\n"
            "  python train_deeplabv3.py   (recommended)\n"
            "  python train_segmentation.py"
        )

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                               verbose=False)
    backbone.eval().to(device)

    dummy = torch.zeros(1, 3, H_DINO, W_DINO).to(device)
    with torch.no_grad():
        embed_dim = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]

    head = SegmentationHead(embed_dim, NUM_CLASSES, H_DINO // 14, W_DINO // 14).to(device)
    head.load_state_dict(torch.load(dino_path, map_location=device, weights_only=True))
    head.eval()

    models["mode"]     = "dino"
    models["backbone"] = backbone
    models["head"]     = head
    print(f"[segmentation] Loaded DINOv2 head from {dino_path}")
    return models

# ── Inference ────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_dino(models, tensor):
    """tensor: (1,3,H,W) on correct device → (1, C, h, w) logits"""
    tokens = models["backbone"].forward_features(tensor)["x_norm_patchtokens"]
    return models["head"](tokens)

@torch.no_grad()
def _forward_dl(models, tensor):
    return models["deeplabv3"](tensor)

def _tta(forward_fn, tensor, out_hw):
    """3-pass TTA: original + hflip + vflip. Returns averaged probs (1,C,H,W)."""
    def run(x):
        p = F.softmax(forward_fn(x), dim=1)
        return F.interpolate(p, size=out_hw, mode="bilinear", align_corners=False)

    p0 = run(tensor)
    ph = torch.flip(run(torch.flip(tensor, dims=[3])), dims=[3])
    pv = torch.flip(run(torch.flip(tensor, dims=[2])), dims=[2])
    return (p0 + ph + pv) / 3.0

def predict(models, pil_image: Image.Image) -> np.ndarray:
    """
    Run segmentation on a PIL image.
    Returns a (H, W) uint8 numpy array of class IDs 0-9.
    """
    device  = models["device"]
    orig_hw = (pil_image.height, pil_image.width)

    if models["mode"] == "deeplabv3":
        tensor = _dl_tf(pil_image).unsqueeze(0).to(device)
        probs  = _tta(lambda x: _forward_dl(models, x), tensor, orig_hw)
    else:
        tensor = _dino_tf(pil_image).unsqueeze(0).to(device)
        probs  = _tta(lambda x: _forward_dino(models, x), tensor, orig_hw)

    return torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def colorize(seg_map: np.ndarray) -> np.ndarray:
    """(H,W) class IDs → (H,W,3) uint8 RGB image."""
    rgb = PALETTE[seg_map.ravel()].reshape(seg_map.shape[0], seg_map.shape[1], 3)
    return rgb


def build_cost_grid(seg_map: np.ndarray) -> np.ndarray:
    """
    (H,W) class IDs → (H,W) uint8 cost grid.
    0 = clear, 1 = costly, 2 = hard obstacle.
    """
    grid = np.full(seg_map.shape, 2, dtype=np.uint8)
    for cls, cost in TRAVERSAL_COST.items():
        grid[seg_map == cls] = cost
    return grid