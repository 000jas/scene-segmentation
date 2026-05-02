"""
visualization.py — Image overlay utilities.
"""

import cv2
import numpy as np
import sys
import os

# Ensure segmentation.py (in the same directory) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from segmentation import PALETTE, CLASS_NAMES, NUM_CLASSES
except ImportError:
    # Fallback palette if segmentation module not available
    NUM_CLASSES = 10
    CLASS_NAMES = ["Background","Trees","Bushes","Dry Grass","Rocks",
                   "Logs","Terrain","Sky","Landscape","Other"]
    PALETTE = np.array([
        [20,20,20],[34,139,34],[0,200,80],[210,180,140],[112,112,112],
        [101,67,33],[160,100,40],[135,206,235],[128,128,0],[148,0,211]
    ], dtype=np.uint8)


def colorize_segmentation(seg_map: np.ndarray) -> np.ndarray:
    """(H,W) class IDs -> (H,W,3) BGR image for display."""
    rgb = PALETTE[seg_map.ravel()].reshape(seg_map.shape[0], seg_map.shape[1], 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def overlay_path(orig_bgr: np.ndarray,
                 seg_map: np.ndarray,
                 plan: dict,
                 seg_alpha: float = 0.45) -> np.ndarray:
    """
    Blend segmentation over the original image, then draw:
      - Obstacle pixels in semi-transparent red
      - A* path as a thick green polyline with waypoint dots
      - Start marker (blue circle) and Goal marker (yellow star)

    orig_bgr : original image  (H,W,3) BGR
    seg_map  : class ID map    (H,W)
    plan     : output of plan_path()
    """
    from segmentation import TRAVERSAL_COST

    h, w = orig_bgr.shape[:2]
    canvas = orig_bgr.copy().astype(np.float32)

    # ── Blend segmentation colour ──────────────────────────────────
    seg_rgb = PALETTE[seg_map.ravel()].reshape(h, w, 3).astype(np.float32)
    seg_bgr = seg_rgb[:, :, ::-1]
    canvas  = cv2.addWeighted(canvas, 1.0 - seg_alpha, seg_bgr, seg_alpha, 0)
    canvas  = canvas.astype(np.uint8)

    # ── Red tint on hard obstacles ─────────────────────────────────
    obstacle_mask = np.zeros((h, w), dtype=bool)
    for cls, cost in TRAVERSAL_COST.items():
        if cost == 2:
            obstacle_mask[seg_map == cls] = True
    overlay = canvas.copy()
    overlay[obstacle_mask] = (overlay[obstacle_mask].astype(np.int32)
                               + np.array([0, 0, 60], dtype=np.int32)).clip(0, 255).astype(np.uint8)
    canvas = overlay

    # ── Draw path ──────────────────────────────────────────────────
    path = plan.get("path", [])
    if path:
        pts = np.array([(c, r) for r, c in path], dtype=np.int32)   # (x, y)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False,
                      (0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
        # Waypoint dots every N points
        step = max(1, len(pts) // 20)
        for pt in pts[::step]:
            cv2.circle(canvas, tuple(pt), 4, (0, 220, 0), -1)

    # ── Start marker (blue filled circle) ─────────────────────────
    sr, sc = plan["start"]
    cv2.circle(canvas, (sc, sr), 14, (200, 80, 0),  -1)
    cv2.circle(canvas, (sc, sr), 14, (255, 255, 255), 2)
    cv2.putText(canvas, "S", (sc - 6, sr + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ── Goal marker (yellow filled circle) ────────────────────────
    gr, gc = plan["goal"]
    cv2.circle(canvas, (gc, gr), 14, (0, 200, 200),  -1)
    cv2.circle(canvas, (gc, gr), 14, (255, 255, 255), 2)
    cv2.putText(canvas, "G", (gc - 6, gr + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if not plan["found"]:
        cv2.putText(canvas, "NO PATH FOUND", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return canvas


def make_legend() -> np.ndarray:
    """Return a (H, W, 3) BGR legend image for all 10 classes."""
    cell_h, cell_w = 28, 180
    img = np.ones((cell_h * NUM_CLASSES, cell_w, 3), dtype=np.uint8) * 245

    for i, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        y = i * cell_h
        bgr = color[::-1].tolist()
        cv2.rectangle(img, (4, y + 4), (26, y + cell_h - 4), bgr, -1)
        cv2.rectangle(img, (4, y + 4), (26, y + cell_h - 4), (80, 80, 80), 1)
        cv2.putText(img, name, (32, y + cell_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    return img


# ── Compatibility API used by app.py and test_integration.py ─────────

def overlay_segmentation(image: np.ndarray,
                          colored_mask: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """
    Blend original image with segmentation colour mask.

    image        : BGR image (H, W, 3)
    colored_mask : RGB colored mask (H, W, 3) from model_inference.colorize()
    alpha        : weight of the mask layer (0=all image, 1=all mask)
    Returns BGR blended image.
    """
    # colored_mask may be RGB (from model_inference), convert to BGR
    if colored_mask.shape[2] == 3:
        mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
    else:
        mask_bgr = colored_mask

    # Resize mask to match image if needed
    if image.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    return cv2.addWeighted(image, 1.0 - alpha, mask_bgr, alpha, 0)


def draw_path_on_image(image: np.ndarray,
                       path,
                       color=(0, 255, 0),
                       thickness: int = 3) -> np.ndarray:
    """
    Draw a path (list of (x,y) points) on a BGR image.
    Returns a copy with the path drawn.
    """
    out = image.copy()
    if path is None or len(path) == 0:
        return out
    pts = np.array(path, dtype=np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], False,
                  color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def visualize_obstacle_grid(obstacle_grid: np.ndarray) -> np.ndarray:
    """
    Convert binary obstacle grid (0=navigable, 1=obstacle) to a BGR image.
    Green = navigable, Red = obstacle.
    """
    h, w = obstacle_grid.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[obstacle_grid == 0] = (0, 200, 0)    # navigable — green
    vis[obstacle_grid == 1] = (0, 0, 200)    # obstacle  — red
    return vis


def create_comparison_grid(images: list, labels: list = None) -> np.ndarray:
    """
    Stack a list of BGR images side-by-side into one wide image.
    Optionally add text labels at the top of each panel.
    All images are resized to the same height as the first image.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    target_h = images[0].shape[0]
    panels = []
    for i, img in enumerate(images):
        # Resize to same height
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        panel = img.copy()
        if labels and i < len(labels):
            cv2.putText(panel, labels[i], (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, labels[i], (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        panels.append(panel)
    return np.concatenate(panels, axis=1)