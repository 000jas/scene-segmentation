import torch
import numpy as np
import os
import random
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler


############################################
# DEVICE
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: Running on CPU (very slow)")


############################################
# CONFIG
############################################

value_map = {
    0:0, 100:1, 200:2, 300:3, 500:4,
    550:5, 700:6, 800:7, 7100:8, 10000:9
}

NUM_CLASSES = len(value_map)

W         = int(((960/2)//14)*14)   # 476
H         = int(((540/2)//14)*14)   # 266
CACHE_DIR = "feature_cache"         # DINOv2 tokens saved here (written once)


############################################
# MASK UTILITY
############################################

def convert_mask(mask_img):
    arr     = np.array(mask_img)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for k, v in value_map.items():
        new_arr[arr == k] = v
    return Image.fromarray(new_arr)


############################################
# FEATURE EXTRACTION (runs once)
############################################

_extract_tf = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class _RawImageDS(Dataset):
    """Returns (image_tensor, save_path_str) — used only during extraction."""

    def __init__(self, entries):   # entries: list of (img_path, dst_path)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, dst = self.entries[idx]
        return _extract_tf(Image.open(img_path).convert("RGB")), dst


def ensure_features_cached(backbone, splits, cache_dir):
    """
    For each (img_dir, prefix) in splits, run the frozen DINOv2 backbone once
    and save patch-token grids (Hp, Wp, C) as float16 .pt files.
    Already-present files are skipped so re-runs are instant.
    """
    os.makedirs(cache_dir, exist_ok=True)

    todo = []
    for img_dir, prefix in splits:
        for name in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(name)[0]
            dst  = os.path.join(cache_dir, f"{prefix}_{stem}.pt")
            if not os.path.exists(dst):
                todo.append((os.path.join(img_dir, name), dst))

    if not todo:
        print("Feature cache already complete — skipping extraction.")
        return

    print(f"Extracting backbone features for {len(todo)} images → {cache_dir}")

    Hp, Wp = H // 14, W // 14
    loader = DataLoader(_RawImageDS(todo), batch_size=8, num_workers=4)

    backbone.eval()
    with torch.no_grad():
        for imgs, save_paths in tqdm(loader, desc="  Extracting"):
            imgs   = imgs.to(device)
            tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]  # (B, N, C)
            grids  = tokens.view(tokens.shape[0], Hp, Wp, tokens.shape[2])  # (B, Hp, Wp, C)
            for b, dst in enumerate(save_paths):
                torch.save(grids[b].cpu().half(), dst)   # save as float16

    print("Extraction complete.")


############################################
# CACHED DATASET
# Loads pre-extracted token grids from disk.
# Applies hflip / vflip on the spatial token grid and mask together.
############################################

class CachedDataset(Dataset):

    def __init__(self, img_dir, mask_dir, cache_dir, prefix, augment=False):
        self.mask_dir  = mask_dir
        self.cache_dir = cache_dir
        self.prefix    = prefix
        self.augment   = augment
        self.names     = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        stem = os.path.splitext(name)[0]

        # Load token grid (Hp, Wp, C) float16
        grid = torch.load(
            os.path.join(self.cache_dir, f"{self.prefix}_{stem}.pt"),
            map_location="cpu", weights_only=True
        )

        # Load + convert mask, resize to (W, H) = (476, 266)
        mask = convert_mask(Image.open(os.path.join(self.mask_dir, name)))
        mask = mask.resize((W, H), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                grid = grid.flip(1)       # flip spatial width  (Wp dim)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                grid = grid.flip(0)       # flip spatial height (Hp dim)
                mask = TF.vflip(mask)

        # Flatten to (N, C) float32 for SegmentationHead
        Hp, Wp, C = grid.shape
        tokens = grid.reshape(Hp * Wp, C).float()
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return tokens, mask_t


############################################
# MODEL
############################################

class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.decoder(x)


############################################
# DICE LOSS
############################################

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred          = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        intersection  = (pred * target_onehot).sum(dim=(2, 3))
        union         = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice          = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()


############################################
# CLASS WEIGHTS
############################################

def compute_class_weights(dataset, num_classes=NUM_CLASSES, max_samples=500):
    counts  = np.zeros(num_classes, dtype=np.float64)
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))

    for idx in tqdm(indices, desc="Computing class weights"):
        _, mask = dataset[idx]
        for c in range(num_classes):
            counts[c] += (mask == c).sum().item()

    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


############################################
# TRAINING
# Runtime breakdown (RTX 4050, first run):
#   Feature extraction  ~4 min  (runs once; skipped on re-runs)
#   Head-only training  ~8 min  (35 epochs × 9300 imgs, batch 8)
#   Total               ~12 min / ~8 min on re-run
############################################

def train():

    batch_size = 8     # larger batch is fine — head is tiny
    epochs     = 35

    ############################################
    # LOAD BACKBONE — for extraction only
    ############################################

    print("Loading DINOv2 backbone…")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)

    ############################################
    # CACHE FEATURES (idempotent — safe to re-run)
    ############################################

    ensure_features_cached(
        backbone,
        splits=[
            ("Offroad_Segmentation_Training_Dataset/train/Color_Images", "train"),
            ("Offroad_Segmentation_Training_Dataset/val/Color_Images",   "val"),
        ],
        cache_dir=CACHE_DIR
    )

    # Free GPU VRAM — backbone is no longer needed
    backbone.cpu()
    del backbone
    torch.cuda.empty_cache()

    ############################################
    # DATASETS
    ############################################

    trainset = CachedDataset(
        "Offroad_Segmentation_Training_Dataset/train/Color_Images",
        "Offroad_Segmentation_Training_Dataset/train/Segmentation",
        CACHE_DIR, prefix="train", augment=True
    )
    valset = CachedDataset(
        "Offroad_Segmentation_Training_Dataset/val/Color_Images",
        "Offroad_Segmentation_Training_Dataset/val/Segmentation",
        CACHE_DIR, prefix="val", augment=True
    )
    combined = ConcatDataset([trainset, valset])

    train_loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    ############################################
    # HEAD
    ############################################

    embed_dim = trainset[0][0].shape[1]   # C from (N, C)
    model = SegmentationHead(embed_dim, NUM_CLASSES, H//14, W//14).to(device)

    ############################################
    # CLASS-WEIGHTED LOSS
    ############################################

    print("Computing class weights…")
    weights = compute_class_weights(trainset).to(device)
    print("Class weights:", np.round(weights.cpu().numpy(), 3))

    ce   = nn.CrossEntropyLoss(weight=weights)
    dice = DiceLoss()

    ############################################
    # OPTIMIZER + SCHEDULER
    ############################################

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler()

    ############################################
    # TRAINING LOOP
    ############################################

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        loop = tqdm(train_loader)

        for tokens, mask in loop:

            tokens = tokens.to(device, non_blocking=True)
            mask   = mask.long().to(device, non_blocking=True)

            with autocast():
                logits = model(tokens)
                logits = F.interpolate(
                    logits, size=(H, W),
                    mode="bilinear", align_corners=False
                )
                loss = ce(logits, mask) + dice(logits, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}  avg_loss={total_loss/len(train_loader):.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}")

    ############################################
    # SAVE
    ############################################

    torch.save(model.state_dict(), "segmentation_head.pth")
    print("Model saved: segmentation_head.pth")


if __name__ == "__main__":
    train()