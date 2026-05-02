"""
Quick test script to verify model integration and path planning
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_inference import predict_segmentation
from path_planner import compute_path
from visualization import overlay_segmentation


def test_with_synthetic_image():
    """Test with a synthetic image"""
    print("=" * 60)
    print("Testing Terrain Navigation System")
    print("=" * 60)
    
    # Create a simple test image (640x480)
    print("\n1. Creating synthetic test image...")
    height, width = 480, 640
    test_image = np.ones((height, width, 3), dtype=np.uint8) * 180  # Gray background
    
    # Add some features
    # Sand/terrain region (bottom)
    test_image[300:, :] = [140, 180, 210]  # Tan color
    
    # Add obstacles (rocks)
    cv2.rectangle(test_image, (200, 200), (280, 280), (100, 100, 100), -1)
    cv2.rectangle(test_image, (400, 150), (480, 230), (100, 100, 100), -1)
    
    # Add vegetation
    cv2.circle(test_image, (100, 100), 40, (50, 150, 50), -1)
    cv2.circle(test_image, (500, 120), 50, (50, 150, 50), -1)
    
    print(f"   ✓ Created test image: {width}x{height}")
    
    # Test segmentation
    print("\n2. Testing segmentation model...")
    print("   (This will download DINOv2 model on first run - may take a few minutes)")
    
    try:
        seg_mask, colored_mask, obstacle_grid = predict_segmentation(test_image)
        print(f"   ✓ Segmentation successful!")
        print(f"   - Segmentation shape: {seg_mask.shape}")
        print(f"   - Colored mask shape: {colored_mask.shape}")
        print(f"   - Obstacle grid shape: {obstacle_grid.shape}")
        print(f"   - Unique classes found: {np.unique(seg_mask)}")
        
        # Calculate navigable percentage
        navigable = np.sum(obstacle_grid == 0)
        total = obstacle_grid.size
        print(f"   - Navigable area: {(navigable/total)*100:.1f}%")
        
    except Exception as e:
        print(f"   ✗ Segmentation failed: {e}")
        return False
    
    # Test visualization
    print("\n3. Testing visualization overlay...")
    try:
        overlay = overlay_segmentation(test_image, colored_mask, alpha=0.5)
        print(f"   ✓ Overlay created: {overlay.shape}")
    except Exception as e:
        print(f"   ✗ Overlay failed: {e}")
        return False
    
    # Test path planning
    print("\n4. Testing A* path planning...")
    try:
        start_pos = (width // 2, height - 20)
        goal_pos = (width // 2, 20)
        
        path_image, path = compute_path(test_image, obstacle_grid, start_pos, goal_pos)
        
        if path is not None:
            print(f"   ✓ Path planning successful!")
            print(f"   - Path length: {len(path)} pixels")
            print(f"   - Start: {start_pos}")
            print(f"   - Goal: {goal_pos}")
        else:
            print(f"   ⚠ No path found (might be expected for synthetic image)")
        
    except Exception as e:
        print(f"   ✗ Path planning failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save test outputs
    print("\n5. Saving test outputs...")
    try:
        os.makedirs("test_outputs", exist_ok=True)
        cv2.imwrite("test_outputs/test_original.png", test_image)
        cv2.imwrite("test_outputs/test_segmentation.png", colored_mask)
        cv2.imwrite("test_outputs/test_overlay.png", overlay)
        cv2.imwrite("test_outputs/test_path.png", path_image)
        
        # Save obstacle grid as visualization
        obstacle_vis = np.zeros_like(test_image)
        obstacle_vis[obstacle_grid == 0] = [200, 200, 200]  # Navigable = gray
        obstacle_vis[obstacle_grid == 1] = [0, 0, 255]      # Obstacle = red
        cv2.imwrite("test_outputs/test_obstacle_grid.png", obstacle_vis)
        
        print(f"   ✓ Saved outputs to test_outputs/")
    except Exception as e:
        print(f"   ⚠ Could not save outputs: {e}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run the Streamlit app with:")
    print("  cd frontend && streamlit run app.py")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = test_with_synthetic_image()
    sys.exit(0 if success else 1)
