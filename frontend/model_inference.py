import os
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    from gemini_api import GeminiTerrainAPI
except ImportError:
    from .gemini_api import GeminiTerrainAPI

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Configuration
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

NUM_CLASSES = len(value_map)

# Model dimensions
W = int(((960 / 2) // 14) * 14)  # 476
H = int(((540 / 2) // 14) * 14)  # 266

# Color palette for visualization
colors = np.array([
    [0, 0, 0],         # Class 0: Background/Unknown - Black
    [34, 139, 34],     # Class 1: Forest Green
    [0, 255, 0],       # Class 2: Bright Green
    [210, 180, 140],   # Class 3: Tan (Sand/Terrain)
    [139, 90, 43],     # Class 4: Brown
    [128, 128, 0],     # Class 5: Olive
    [139, 69, 19],     # Class 6: Saddle Brown (Rock)
    [128, 128, 128],   # Class 7: Gray (Rock)
    [160, 82, 45],     # Class 8: Sienna
    [135, 206, 235]    # Class 9: Sky Blue
], dtype=np.uint8)

# Obstacle mapping (1 = obstacle, 0 = navigable)
OBSTACLE_MAP = {
    0: 1,  # Unknown - treat as obstacle
    1: 1,  # Forest Green - tree/vegetation
    2: 1,  # Bright Green - tree/vegetation
    3: 0,  # Tan - sand/terrain (navigable)
    4: 0,  # Brown - terrain (navigable)
    5: 1,  # Olive - vegetation
    6: 1,  # Saddle Brown - rock (obstacle)
    7: 1,  # Gray - rock (obstacle)
    8: 0,  # Sienna - terrain (navigable)
    9: 0   # Sky Blue - sky (treat as navigable for simplicity)
}


class SegmentationHead(nn.Module):
    """Lightweight segmentation decoder head"""
    
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


class TerrainSegmentationModel:
    """Main model class for terrain segmentation"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "segmentation_head.pth")
        self.device = device
        self.model_path = model_path
        self.backbone = None
        self.model = None
        self.mode = "dino"
        self.gemini = GeminiTerrainAPI()
        self.transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        self._load_models()
    
    def _load_models(self):
        """Load DINOv2 backbone and segmentation head"""
        print(f"Loading models on {self.device}...")

        if not os.path.exists(self.model_path):
            print(f"Segmentation weights not found at {self.model_path}.")
            print("Using static Gemini API fallback.")
            self.mode = "gemini-static"
            return

        try:
            # Load DINOv2 backbone. This uses the torch hub cache when present and
            # downloads only when the repo has not already been cached.
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                verbose=False
            )
            self.backbone.eval().to(self.device)
        except Exception as exc:
            print(f"DINOv2 backbone load failed: {exc}")
            print("Using static Gemini API fallback.")
            self.mode = "gemini-static"
            return
        
        # Get embedding dimension
        dummy = torch.randn(1, 3, H, W).to(self.device)
        with torch.no_grad():
            out = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
        embed_dim = out.shape[2]
        
        # Load segmentation head
        self.model = SegmentationHead(
            embed_dim,
            NUM_CLASSES,
            H // 14,
            W // 14
        ).to(self.device)
        
        try:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
        except Exception as exc:
            print(f"Segmentation head load failed: {exc}")
            print("Using static Gemini API fallback.")
            self.backbone = None
            self.model = None
            self.mode = "gemini-static"
            return

        print("Models loaded successfully!")
    
    def predict(self, image):
        """
        Perform segmentation on input image
        
        Args:
            image: BGR image (numpy array from cv2)
            
        Returns:
            segmentation_mask: Class predictions (H, W)
            colored_mask: RGB visualization (H, W, 3)
            obstacle_grid: Binary obstacle map (H, W) - 1=obstacle, 0=navigable
        """
        if self.mode == "gemini-static":
            pred = self.gemini.segment_terrain(image)
            colored_mask = self.colorize(pred)
            obstacle_grid = self.create_obstacle_grid(pred)
            return pred, colored_mask, obstacle_grid

        # Convert BGR to RGB and create PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Store original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Transform and prepare for inference
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
            
            # Get segmentation logits
            logits = self.model(tokens)
            
            # Upsample to original resolution
            logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )
            
            # Get class predictions
            pred = torch.argmax(logits, 1).cpu().numpy()[0]
        
        # Create colored visualization
        colored_mask = self.colorize(pred)
        
        # Create obstacle grid
        obstacle_grid = self.create_obstacle_grid(pred)
        
        return pred, colored_mask, obstacle_grid
    
    def colorize(self, mask):
        """Convert class mask to RGB image"""
        h, w = mask.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for c in range(NUM_CLASSES):
            img[mask == c] = colors[c]
        
        return img
    
    def create_obstacle_grid(self, mask):
        """Convert class mask to binary obstacle grid"""
        obstacle_grid = np.zeros_like(mask, dtype=np.uint8)
        
        for class_id, is_obstacle in OBSTACLE_MAP.items():
            obstacle_grid[mask == class_id] = is_obstacle
        
        return obstacle_grid


# Global model instance (singleton)
_model_instance = None


def get_model():
    """Get or create model instance (singleton pattern)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = TerrainSegmentationModel()
    return _model_instance


def get_inference_mode():
    """Return the active inference backend name for UI/debug display."""
    return get_model().mode


def predict_segmentation(image):
    """
    Main inference function for the frontend
    
    Args:
        image: BGR image (numpy array from cv2)
        
    Returns:
        segmentation_mask: Class predictions
        colored_mask: RGB visualization
        obstacle_grid: Binary obstacle map
    """
    model = get_model()
    return model.predict(image)
