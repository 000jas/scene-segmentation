import torch
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import both model architectures from the training scripts
from train_segmentation import SegmentationHead, W as W_DINO, H as H_DINO
from train_segformer     import SegFormerB0,     IMG_W as W_SEG, IMG_H as H_SEG


############################################
# DEVICE
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


############################################
# CONFIG
############################################

value_map = {
    0:0, 100:1, 200:2, 300:3, 500:4,
    550:5, 700:6, 800:7, 7100:8, 10000:9
}

NUM_CLASSES = len(value_map)

# Ensemble weights (DINOv2 head : FPNDecoder)
# DINOv2 head is known-good at 0.44; favour it until FPN is validated
W_DINO_ENS = 0.65
W_SEG_ENS  = 0.35


############################################
# COLOR PALETTE
############################################

colors = np.array([
    [0,   0,   0  ],   # background
    [34,  139, 34 ],   # trees
    [0,   255, 0  ],   # bushes
    [210, 180, 140],   # dry grass
    [139, 90,  43 ],   # rocks
    [128, 128, 0  ],   # logs
    [139, 69,  19 ],   # terrain
    [128, 128, 128],   # sky
    [160, 82,  45 ],   # landscape
    [135, 206, 235],   # other
], dtype=np.uint8)


############################################
# HELPERS
############################################

def convert_mask(mask_img):
    arr     = np.array(mask_img)
    new_arr = np.zeros_like(arr)
    for k, v in value_map.items():
        new_arr[arr == k] = v
    return new_arr


def colorize(mask):
    h, w = mask.shape
    img  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        img[mask == c] = colors[c]
    return img


def compute_iou(pred, gt, num_classes=NUM_CLASSES):
    ious = []
    for cls in range(num_classes):
        inter = np.logical_and(pred == cls, gt == cls).sum()
        union = np.logical_or(pred == cls, gt == cls).sum()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


############################################
# DINO SEGMENTATION HEAD
############################################
# TEST-TIME AUGMENTATION
############################################

def tta_infer(forward_fn, img_tensor, target_size):
    """
    3-pass TTA: original + h-flip + v-flip.
    forward_fn: callable(tensor 1×3×H×W) → softmax probs (1, C, h, w)
    Returns averaged probability map upsampled to target_size (H, W).
    """

    def run(x):
        with torch.no_grad():
            p = forward_fn(x)
        return F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)

    probs = run(img_tensor)

    # horizontal flip — flip probs back after inference
    ph = run(torch.flip(img_tensor, dims=[3]))
    ph = torch.flip(ph, dims=[3])

    # vertical flip
    pv = run(torch.flip(img_tensor, dims=[2]))
    pv = torch.flip(pv, dims=[2])

    return (probs + ph + pv) / 3.0


############################################
# MAIN TEST
############################################

def test():

    ############################################
    # LOAD DINOv2 BACKBONE + HEAD
    ############################################

    print("Loading DINOv2 backbone…")

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    dummy = torch.randn(1, 3, H_DINO, W_DINO).to(device)
    with torch.no_grad():
        embed_dim = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]

    dino_head = SegmentationHead(embed_dim, NUM_CLASSES, H_DINO//14, W_DINO//14).to(device)
    dino_head.load_state_dict(torch.load("segmentation_head.pth", map_location=device, weights_only=True))
    dino_head.eval()

    print("DINOv2 head loaded.")

    ############################################
    # LOAD SEGFORMER (optional — skip if not trained yet)
    ############################################

    use_segformer = os.path.exists("segformer_b0.pth")

    if use_segformer:
        print("Loading SegFormer-B0…")
        segformer = SegFormerB0(embed_dim, NUM_CLASSES, H_DINO//14, W_DINO//14).to(device)
        segformer.load_state_dict(torch.load("segformer_b0.pth", map_location=device, weights_only=True))
        segformer.eval()
        print("SegFormer-B0 loaded.")
    else:
        print("segformer_b0.pth not found — running DINOv2 only (no ensemble).")
        print("Run train_segformer.py first to enable ensemble mode.")

    ############################################
    # TRANSFORMS
    ############################################

    dino_transform = transforms.Compose([
        transforms.Resize((H_DINO, W_DINO)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    seg_transform = transforms.Compose([
        transforms.Resize((H_SEG, W_SEG)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ############################################
    # DATA PATHS
    ############################################

    image_dir  = "Offroad_Segmentation_testImages/Color_Images"
    mask_dir   = "Offroad_Segmentation_testImages/Segmentation"
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    files  = os.listdir(image_dir)
    has_gt = os.path.isdir(mask_dir) and bool(os.listdir(mask_dir))

    ############################################
    # INFERENCE LOOP
    ############################################

    total_iou = 0.0
    count_gt  = 0

    for name in tqdm(files):

        img      = Image.open(os.path.join(image_dir, name)).convert("RGB")
        orig_h_w = (img.height, img.width)

        # --- DINOv2 forward function (used by TTA) ---
        x_dino = dino_transform(img).unsqueeze(0).to(device)

        def dino_forward(x):
            tokens = backbone.forward_features(x)["x_norm_patchtokens"]
            logits = dino_head(tokens)
            return F.softmax(logits, dim=1)

        dino_probs = tta_infer(dino_forward, x_dino, orig_h_w)  # (1, C, H, W)

        # --- SegFormer forward + ensemble (if available) ---
        if use_segformer:
            def seg_forward(x):
                tokens = backbone.forward_features(x)["x_norm_patchtokens"]
                logits = segformer(tokens)
                return F.softmax(logits, dim=1)

            seg_probs = tta_infer(seg_forward, x_dino, orig_h_w)
            probs     = W_DINO_ENS * dino_probs + W_SEG_ENS * seg_probs
        else:
            probs = dino_probs

        pred = torch.argmax(probs, dim=1).cpu().numpy()[0]

        # --- IoU (if ground-truth exists) ---
        gt_path = os.path.join(mask_dir, name)
        if has_gt and os.path.exists(gt_path):
            gt = convert_mask(Image.open(gt_path))
            gt = cv2.resize(
                gt.astype(np.uint8),
                (pred.shape[1], pred.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            total_iou += compute_iou(pred, gt)
            count_gt  += 1

        # --- Save outputs ---
        cv2.imwrite(os.path.join(output_dir, name), pred.astype(np.uint8))

        color_mask = colorize(pred)
        cv2.imwrite(
            os.path.join(output_dir, "color_" + name),
            cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        )

    ############################################
    # FINAL SCORE
    ############################################

    if count_gt > 0:
        print(f"\nMean IoU: {total_iou / count_gt:.4f}  (over {count_gt} images)")
    else:
        print("\nNo ground-truth masks found — predictions saved to 'predictions/'.")


############################################
# RUN
############################################

if __name__ == "__main__":
    test()
