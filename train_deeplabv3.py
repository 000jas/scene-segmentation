import os, random, numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

############################################
# CONFIG
############################################

value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
NUM_CLASSES = 10
IMG_SIZE    = 512   # larger crops → better spatial detail
BATCH_SIZE  = 8     # resnet101 + 512px fits in 6GB with AMP
EPOCHS      = 1     # quick test run
LR          = 2e-4  # OneCycleLR max_lr
SAVE_PATH   = "deeplabv3_best.pth"
EVAL_EVERY  = 1     # validate every epoch
CLASS_NAMES = ["background","trees","bushes","dry_grass","rocks",
               "logs","terrain","sky","landscape","other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: Running on CPU — training will be very slow")

############################################
# MASK REMAP
############################################

def remap_mask(arr):
    out = np.zeros_like(arr, dtype=np.uint8)
    for k, v in value_map.items():
        out[arr == k] = v
    return out

############################################
# AUGMENTATIONS
############################################

train_aug = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.4, 1.0), ratio=(0.75, 1.33)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5,
                       border_mode=0),
    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ColorJitter(p=0.4),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32),
                    hole_width_range=(16, 32), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

############################################
# CLASS WEIGHTS (inverse-frequency, sampled from training masks)
############################################

def compute_class_weights(mask_dir, n_samples=400):
    """Estimate per-class frequency on a subset of masks and return inverse-freq weights."""
    print("Computing class weights...")
    counts  = np.zeros(NUM_CLASSES, dtype=np.float64)
    files   = sorted(os.listdir(mask_dir))
    sample  = files[:min(n_samples, len(files))]
    for f in sample:
        arr = np.array(Image.open(os.path.join(mask_dir, f)))
        arr = remap_mask(arr)
        for c in range(NUM_CLASSES):
            counts[c] += (arr == c).sum()
    freq    = (counts + 1.0) / (counts.sum() + float(NUM_CLASSES))
    weights = 1.0 / (freq ** 0.5)       # square-root smoothing
    weights = weights / weights.mean()   # normalise so average weight == 1
    print("  class weights:", np.round(weights, 2))
    return torch.tensor(weights, dtype=torch.float32)

############################################
# DATASET
############################################

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.files     = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = np.array(Image.open(os.path.join(self.img_dir,  name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, name)))
        mask = remap_mask(mask)
        if self.transform:
            res  = self.transform(image=img, mask=mask)
            img  = res["image"]
            mask = res["mask"].long()
        return img, mask

############################################
# METRICS
############################################

def compute_per_class_iou(pred, gt):
    ious = []
    p = pred.cpu().numpy().ravel()
    g = gt.cpu().numpy().ravel()
    for c in range(NUM_CLASSES):
        inter = ((p == c) & (g == c)).sum()
        union = ((p == c) | (g == c)).sum()
        ious.append(float(inter) / float(union) if union > 0 else float("nan"))
    return ious

############################################
# VALIDATION
############################################

def validate(model, loader):
    model.eval()
    cls_buckets = [[] for _ in range(NUM_CLASSES)]
    with torch.no_grad():
        for img, mask in tqdm(loader, desc="  Val", leave=False):
            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast():
                pred = torch.argmax(model(img), dim=1)
            for c, iou in enumerate(compute_per_class_iou(pred, mask)):
                if iou == iou:   # not nan
                    cls_buckets[c].append(iou)
    per_cls = [float(np.mean(v)) if v else float("nan") for v in cls_buckets]
    valid   = [v for v in per_cls if v == v]
    return per_cls, (float(np.mean(valid)) if valid else 0.0)

############################################
# TRAINING
############################################

def train():
    train_ds = SegDataset(
        "Offroad_Segmentation_Training_Dataset/train/Color_Images",
        "Offroad_Segmentation_Training_Dataset/train/Segmentation",
        transform=train_aug,
    )
    val_ds = SegDataset(
        "Offroad_Segmentation_Training_Dataset/val/Color_Images",
        "Offroad_Segmentation_Training_Dataset/val/Segmentation",
        transform=val_aug,
    )
    # Train on combined dataset, evaluate only on val split
    combined = ConcatDataset([train_ds, val_ds])

    train_loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DeepLabV3+  ResNet-101  params={nparams:.1f}M")

    class_weights = compute_class_weights(
        "Offroad_Segmentation_Training_Dataset/train/Segmentation"
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer  = optim.AdamW(model.parameters(), lr=LR / 25, weight_decay=1e-4)
    # OneCycleLR: fast warmup + cosine decay — ideal for few epochs
    steps_per_epoch = len(train_loader)
    scheduler  = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4,
    )
    scaler    = GradScaler()
    best_miou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for img, mask in loop:
            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast():
                logits = model(img)
                loss   = 0.5 * dice_loss(logits, mask) + 0.5 * ce_loss(logits, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()   # OneCycleLR steps every batch
            total_loss += loss.item()
            loop.set_postfix(loss=round(loss.item(), 4),
                             lr=round(scheduler.get_last_lr()[0], 6))

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS}  train_loss={avg_loss:.4f}")

        # Validate every EVAL_EVERY epochs and on the last epoch
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            per_cls, miou = validate(model, val_loader)
            print(f"  val_mIoU={miou:.4f}")
            for name, iou in zip(CLASS_NAMES, per_cls):
                tag = "  n/a" if iou != iou else f"{iou:.3f}"
                print(f"  {name:<12}: {tag}")

            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"  -> Saved best model  mIoU={best_miou:.4f}")
        else:
            # Save a checkpoint every epoch regardless
            torch.save(model.state_dict(), SAVE_PATH.replace(".pth", "_latest.pth"))

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}  -> {SAVE_PATH}")


if __name__ == "__main__":
    train()