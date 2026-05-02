import torch
import numpy as np
import os
import random
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# CONFIG  — must match train_segmentation.py
############################################

value_map = {
    0:0, 100:1, 200:2, 300:3, 500:4,
    550:5, 700:6, 800:7, 7100:8, 10000:9
}

NUM_CLASSES = len(value_map)

W       = int(((960/2)//14)*14)   # 476
H       = int(((540/2)//14)*14)   # 266
IMG_W   = W
IMG_H   = H

CACHE_DIR = "feature_cache"

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
# CACHED DATASET
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

        grid = torch.load(
            os.path.join(self.cache_dir, f"{self.prefix}_{stem}.pt"),
            map_location="cpu", weights_only=True
        )

        mask = convert_mask(Image.open(os.path.join(self.mask_dir, name)))
        mask = mask.resize((W, H), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                grid = grid.flip(1)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                grid = grid.flip(0)
                mask = TF.vflip(mask)

        Hp, Wp, C = grid.shape
        tokens = grid.reshape(Hp * Wp, C).float()
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return tokens, mask_t

############################################
# ASPP + FPN DECODER
#
# Wider/deeper than the flat SegmentationHead so the two models
# learn different representations and ensemble well.
#
# Layout:
#   tokens (B, Hp*Wp, 384)
#     -> reshape (B, 384, Hp, Wp)
#     -> proj 384->512
#     -> ASPP (multi-scale context)   -> 256
#     -> 2x up + refine               -> 128
#     -> 2x up + refine               ->  64
#     -> head 64->NUM_CLASSES
#     -> bilinear up to (H, W)
############################################

class ASPPModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels), nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels), nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8),
            nn.BatchNorm2d(out_channels), nn.GELU()
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        p = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), p], dim=1))


def _refine(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout), nn.GELU(),
    )


class FPNDecoder(nn.Module):

    def __init__(self, in_channels, num_classes, Hp, Wp):
        super().__init__()
        self.Hp   = Hp
        self.Wp   = Wp
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1),
            nn.BatchNorm2d(512), nn.GELU()
        )
        self.aspp = ASPPModule(512, 256)
        self.up2  = _refine(256, 128)
        self.up3  = _refine(128,  64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.Hp, self.Wp, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        x = self.aspp(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        return self.head(x)


# Alias kept for any code that imports SegFormerB0
SegFormerB0 = FPNDecoder

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
############################################

def train():

    batch_size = 8
    epochs     = 35

    if not os.path.isdir(CACHE_DIR) or len(os.listdir(CACHE_DIR)) == 0:
        raise RuntimeError(
            f"Feature cache '{CACHE_DIR}' is empty.\n"
            "Run train_segmentation.py first to build the cache."
        )

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
        combined, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    embed_dim = trainset[0][0].shape[1]
    model = FPNDecoder(embed_dim, NUM_CLASSES, H//14, W//14).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"FPNDecoder  embed_dim={embed_dim}  grid={H//14}x{W//14}  params={nparams:.1f}M")

    print("Computing class weights...")
    weights = compute_class_weights(trainset).to(device)
    print("Class weights:", np.round(weights.cpu().numpy(), 3))

    ce        = nn.CrossEntropyLoss(weight=weights)
    dice      = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader)

        for tokens, mask in loop:
            tokens = tokens.to(device, non_blocking=True)
            mask   = mask.long().to(device, non_blocking=True)

            with autocast():
                logits = model(tokens)
                logits = F.interpolate(logits, size=(H, W),
                                       mode="bilinear", align_corners=False)
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

    torch.save(model.state_dict(), "segformer_b0.pth")
    print("Model saved: segformer_b0.pth")


if __name__ == "__main__":
    train()