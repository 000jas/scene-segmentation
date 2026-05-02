import streamlit as st
import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_inference import predict_segmentation
from model_inference import get_inference_mode
from gemini_api import GeminiTerrainAPI
from path_planner import compute_path, AStarPathPlanner
from visualization import (overlay_segmentation, draw_path_on_image, 
                          visualize_obstacle_grid, create_comparison_grid)

# Page configuration
st.set_page_config(
    page_title="Autonomous Terrain Navigation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean professional look
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 600; }
    .stCaption { color: #6b7280; font-size: 0.82rem; }
    .stExpander > details > summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Off-Road Autonomous Terrain Navigation")
st.markdown(
    "Semantic segmentation of terrain imagery with A\\* path planning to identify "
    "navigable routes and avoid obstacles."
)

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    if "venv" not in sys.executable:
        st.warning("This app is not running from the project venv.")

    gemini_client = GeminiTerrainAPI()
    if gemini_client.configured:
        st.success("Gemini API key loaded.")
    else:
        st.warning("GEMINI_API_KEY not found.")

    # Path planning settings
    st.subheader("Path Planning")
    overlay_alpha = st.slider("Segmentation Overlay Transparency", 0.0, 1.0, 0.5, 0.05)

    start_y = st.slider("Start Position (from bottom)", 10, 100, 20)
    goal_y = st.slider("Goal Position (from top)", 10, 100, 20)

    st.markdown("---")
    st.markdown("""
**Class Legend**
- **Brown / Tan** — Navigable terrain
- **Green** — Vegetation (obstacle)
- **Gray** — Rocks (obstacle)
- **Sky Blue** — Sky
""")

    st.markdown("---")
    st.info("Upload off-road terrain images for best results.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Terrain Image",
    type=["jpg", "png", "jpeg", "bmp"],
    help="Upload an image of off-road terrain"
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Error loading image. Please try another file.")
        st.stop()

    height, width = image.shape[:2]

    # Show processing message
    with st.spinner("Processing image..."):
        start_time = time.time()

        # Step 1: Segmentation
        with st.spinner("Running semantic segmentation..."):
            seg_mask, colored_mask, obstacle_grid = predict_segmentation(image)
            inference_mode = get_inference_mode()

        # Step 2: Create overlay
        overlay = overlay_segmentation(image, colored_mask, alpha=overlay_alpha)

        # Step 3: Path planning
        with st.spinner("Computing optimal path..."):
            start_pos = (width // 2, height - start_y)
            goal_pos = (width // 2, goal_y)
            path_image, path = compute_path(image, obstacle_grid, start_pos, goal_pos)

        navigable_pixels = np.sum(obstacle_grid == 0)
        total_pixels = obstacle_grid.size
        navigable_percent = (navigable_pixels / total_pixels) * 100
        obstacle_percent = 100 - navigable_percent

        with st.spinner("Generating Gemini terrain brief..."):
            gemini_brief = gemini_client.navigation_brief({
                "width": width,
                "height": height,
                "navigable_percent": navigable_percent,
                "obstacle_percent": obstacle_percent,
                "path_found": path is not None,
                "path_length": len(path) if path is not None else 0,
                "inference_mode": inference_mode,
            })

        processing_time = time.time() - start_time

    # Success message
    st.success(f"Processing completed in {processing_time:.2f} seconds using {inference_mode}.")

    # Display results in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image, channels="BGR", use_container_width=True)
        st.caption(f"Resolution: {width} x {height}")

    with col2:
        st.subheader("Segmentation Map")
        st.image(overlay, channels="BGR", use_container_width=True)

        st.caption(f"Navigable Area: {navigable_percent:.1f}%")

    with col3:
        st.subheader("Optimal Path")
        st.image(path_image, channels="BGR", use_container_width=True)

        if path is not None:
            path_length = len(path)
            st.caption(f"Path Length: {path_length} pixels")
        else:
            st.caption("No valid path found — try adjusting start / goal positions.")

    # Additional information section
    st.markdown("---")

    st.subheader("Gemini Terrain Brief")
    st.write(gemini_brief["text"])
    if gemini_brief["used_api"]:
        st.caption(f"Generated with Gemini API model: {gemini_brief['model']}")
    else:
        st.caption("Gemini API was not used for this run.")

    # Expandable sections for detailed info
    with st.expander("Detailed Analysis"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Terrain Classification")
            # Count pixels per class
            unique, counts = np.unique(seg_mask, return_counts=True)
            class_names = ["Unknown", "Forest", "Vegetation", "Sand", "Terrain",
                          "Olive", "Rock-Brown", "Rock-Gray", "Sienna", "Sky"]

            for cls, count in zip(unique, counts):
                if cls < len(class_names):
                    percentage = (count / seg_mask.size) * 100
                    st.write(f"**{class_names[cls]}**: {percentage:.2f}%")

        with col_b:
            st.subheader("Navigation Metrics")
            st.write(f"**Image Size**: {width} x {height} pixels")
            st.write(f"**Total Pixels**: {total_pixels:,}")
            st.write(f"**Navigable Pixels**: {navigable_pixels:,}")
            st.write(f"**Obstacle Pixels**: {total_pixels - navigable_pixels:,}")

            if path is not None:
                st.write(f"**Path Found**: Yes")
                st.write(f"**Path Length**: {len(path)} pixels")
                st.write(f"**Start**: ({start_pos[0]}, {start_pos[1]})")
                st.write(f"**Goal**: ({goal_pos[0]}, {goal_pos[1]})")
            else:
                st.write(f"**Path Found**: No")
                st.write("*Try adjusting start / goal positions in the sidebar.*")

    with st.expander("Download Results"):
        st.write("Download processed images:")

        # Convert images for download
        _, seg_encoded = cv2.imencode('.png', overlay)
        _, path_encoded = cv2.imencode('.png', path_image)

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.download_button(
                label="Download Segmentation",
                data=seg_encoded.tobytes(),
                file_name="segmentation_overlay.png",
                mime="image/png"
            )

        with col_d2:
            st.download_button(
                label="Download Path Visualization",
                data=path_encoded.tobytes(),
                file_name="navigation_path.png",
                mime="image/png"
            )

else:
    # Show sample usage
    st.info("Upload an image above to begin analysis.")

    st.markdown("""
### How It Works

1. **Image Upload** — Provide a terrain or off-road image.
2. **Semantic Segmentation** — The AI model classifies each pixel (terrain, rocks, trees, etc.).
3. **Obstacle Mapping** — Segmentation is converted into a navigable vs. obstacle grid.
4. **Path Planning** — A* algorithm computes an optimal safe route.
5. **Visualization** — Results are displayed with the planned path overlaid.

### Capabilities

- Real-time semantic segmentation using DINOv2 + custom decoder
- A* pathfinding for optimal navigation routes
- Interactive parameter adjustment via the sidebar
- Per-class terrain analysis
- Export of segmentation and path visualizations
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.8rem;'>"
    "Autonomous Terrain Navigation System — DINOv2 / static Gemini API + PyTorch + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
