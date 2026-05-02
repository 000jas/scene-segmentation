import torch
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms



value_map = {
    0:0,
    100:1,
    200:2,
    300:3,
    500:4,
    550:5,
    700:6,
    800:7,
    7100:8,
    10000:9
}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr)

    for k,v in value_map.items():
        new_arr[arr==k] = v

    return new_arr


############################################
# DEVICE
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


############################################
# CLASS MAP
############################################

value_map = {
    0:0,100:1,200:2,300:3,500:4,
    550:5,700:6,800:7,7100:8,10000:9
}

NUM_CLASSES = len(value_map)


############################################
# COLOR PALETTE
############################################

colors = np.array([
[0,0,0],
[34,139,34],
[0,255,0],
[210,180,140],
[139,90,43],
[128,128,0],
[139,69,19],
[128,128,128],
[160,82,45],
[135,206,235]
],dtype=np.uint8)


def colorize(mask):

    h,w = mask.shape

    img = np.zeros((h,w,3),dtype=np.uint8)

    for c in range(NUM_CLASSES):
        img[mask==c] = colors[c]

    return img


############################################
# IOU FUNCTION
############################################

def compute_iou(pred, gt, num_classes=10):

    ious = []

    for cls in range(num_classes):

        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious)


############################################
# MODEL
############################################

class SegmentationHead(nn.Module):

    def __init__(self,in_channels,out_channels,H,W):

        super().__init__()

        self.H = H
        self.W = W

        self.decoder = nn.Sequential(

            nn.Conv2d(in_channels,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256,out_channels,1)

        )


    def forward(self,x):

        B,N,C = x.shape

        x = x.reshape(B,self.H,self.W,C).permute(0,3,1,2)

        return self.decoder(x)


############################################
# TEST
############################################

def test():

    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)


    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])


    ############################################
    # LOAD BACKBONE
    ############################################

    print("Loading DINOv2 backbone...")

    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    )

    backbone.eval().to(device)


    dummy = torch.randn(1,3,h,w).to(device)

    with torch.no_grad():
        out = backbone.forward_features(dummy)["x_norm_patchtokens"]

    embed_dim = out.shape[2]


    ############################################
    # LOAD MODEL
    ############################################

    model = SegmentationHead(
        embed_dim,
        NUM_CLASSES,
        h//14,
        w//14
    ).to(device)

    model.load_state_dict(torch.load("segmentation_head.pth"))
    model.eval()

    print("Model loaded successfully")


    ############################################
    # DATA PATHS
    ############################################

    image_dir = "Offroad_Segmentation_Training_Dataset/val/Color_Images"
    mask_dir = "Offroad_Segmentation_Training_Dataset/val/Segmentation"

    output_dir = "predictions"
    os.makedirs(output_dir,exist_ok=True)


    files = os.listdir(image_dir)


    ############################################
    # INFERENCE
    ############################################

    total_iou = 0

    for name in tqdm(files):

        img_path = os.path.join(image_dir,name)

        img = Image.open(img_path).convert("RGB")

        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():

            tokens = backbone.forward_features(x)["x_norm_patchtokens"]

            logits = model(tokens)

            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False
            )

            pred = torch.argmax(logits,1).cpu().numpy()[0]


        gt_raw = Image.open(os.path.join(mask_dir,name))

        gt = convert_mask(gt_raw)

        gt = cv2.resize(gt,(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_NEAREST)

        ############################################
        # IOU
        ############################################

        iou = compute_iou(pred, gt)

        total_iou += iou


        ############################################
        # SAVE PREDICTIONS
        ############################################

        cv2.imwrite(
            os.path.join(output_dir,name),
            pred.astype(np.uint8)
        )


        color_mask = colorize(pred)

        cv2.imwrite(
            os.path.join(output_dir,"color_"+name),
            cv2.cvtColor(color_mask,cv2.COLOR_RGB2BGR)
        )


    ############################################
    # FINAL SCORE
    ############################################

    mean_iou = total_iou / len(files)

    print("\nMean IoU:", mean_iou)


############################################
# RUN
############################################

if __name__ == "__main__":
    test()