# 🚗 Off-Road Autonomous Terrain Navigation System

A deep learning-based semantic segmentation system for autonomous off-road navigation, featuring real-time terrain classification and optimal path planning using A* algorithm.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

This system combines state-of-the-art computer vision and path planning algorithms to enable autonomous navigation in off-road environments. It performs pixel-wise semantic segmentation to identify navigable terrain and obstacles, then computes optimal safe paths using the A* pathfinding algorithm.

### Key Features

- ✅ **Real-time Semantic Segmentation** using DINOv2 backbone with custom decoder
- ✅ **A* Path Planning Algorithm** for optimal navigation
- ✅ **Interactive Web Interface** built with Streamlit
- ✅ **Multi-class Terrain Classification** (sand, rocks, vegetation, sky, etc.)
- ✅ **Obstacle Detection & Avoidance**
- ✅ **Visual Path Overlay** with start/goal markers
- ✅ **Detailed Terrain Analysis** and metrics

## 🎯 How It Works

### Pipeline

```
Input Image
    ↓
Semantic Segmentation (DINOv2 + Custom Decoder)
    ↓
Terrain Classification (10 classes)
    ↓
Obstacle Grid Generation (Binary: Safe/Obstacle)
    ↓
A* Path Planning
    ↓
Path Overlay Visualization
```

### Terrain Classes

| Class ID | Terrain Type | Color | Navigable |
|----------|-------------|-------|-----------|
| 0 | Unknown | Black | ❌ No |
| 1 | Forest | Forest Green | ❌ No |
| 2 | Vegetation | Bright Green | ❌ No |
| 3 | Sand | Tan | ✅ Yes |
| 4 | Terrain | Brown | ✅ Yes |
| 5 | Olive | Olive | ❌ No |
| 6 | Rock (Brown) | Saddle Brown | ❌ No |
| 7 | Rock (Gray) | Gray | ❌ No |
| 8 | Sienna | Sienna | ✅ Yes |
| 9 | Sky | Sky Blue | ✅ Yes |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU, Apple Silicon MPS, or CPU (GPU recommended for faster inference)
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Semantic-Scene-Segmentation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

   On macOS, install the normal PyPI PyTorch wheels from this file. Do not use CUDA
   wheel tags such as `torch==2.5.1+cu121`; those are Linux/NVIDIA-specific.

3. **Configure Gemini API**
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

The app uses Gemini to generate the terrain/navigation brief from computed
segmentation and path-planning metrics. The `.env` file is ignored by git.

4. **Navigate to frontend**
```bash
cd frontend
pip install -r requirements.txt
```

### Running the Application

**Launch the Streamlit web app:**
```bash
cd frontend
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Streamlit Community Cloud Deployment

Deploy with:
- **Main file path**: `frontend/app.py`
- **Dependencies**: root `requirements.txt`
- **Secret**: `GEMINI_API_KEY = "your_real_gemini_api_key"`

See `DEPLOYMENT.md` for the full checklist.

## 📁 Project Structure

```
Semantic-Scene-Segmentation/
├── frontend/
│   ├── app.py                    # Streamlit web interface
│   ├── model_inference.py        # Model loading and inference
│   ├── path_planner.py           # A* path planning algorithm
│   ├── visualization.py          # Visualization utilities
│   ├── test_integration.py       # Integration tests
│   └── requirements.txt          # Frontend dependencies
├── segmentation_head.pth         # Trained model weights
├── train_segmentation.py         # Training script
├── requirements.txt              # Root dependencies
└── README.md                     # This file
```

## 💻 Usage

### Web Interface

1. **Upload Image**: Click "Upload Terrain Image" and select an off-road terrain image
2. **Adjust Parameters**: Use the sidebar to configure:
   - Segmentation overlay transparency
   - Start and goal positions
3. **View Results**: See three panels:
   - Original image
   - Segmentation overlay
   - Navigation path
4. **Download**: Export segmentation and path visualizations

### Programmatic Usage

```python
from model_inference import predict_segmentation
from path_planner import compute_path
from visualization import draw_path_on_image
import cv2

# Load image
image = cv2.imread('terrain.jpg')

# Perform segmentation
seg_mask, colored_mask, obstacle_grid = predict_segmentation(image)

# Compute path
start_pos = (width // 2, height - 20)
goal_pos = (width // 2, 20)
path_image, path = compute_path(image, obstacle_grid, start_pos, goal_pos)

# Save result
cv2.imwrite('navigation_path.jpg', path_image)
```

## 🧪 Testing

Run integration tests:
```bash
cd frontend
python test_integration.py
```

This will:
- Create a synthetic test image
- Run segmentation
- Compute paths
- Save outputs to `test_outputs/`

## 🎓 Model Architecture

### Backbone: DINOv2
- **Model**: `dinov2_vits14` (Vision Transformer Small)
- **Pre-training**: Self-supervised on large-scale image datasets
- **Embedding Dimension**: 384

### Segmentation Head
```
Input: [B, N, 384] patch tokens
    ↓
Reshape: [B, H/14, W/14, 384]
    ↓
Conv2D(384 → 256) + BatchNorm + GELU
    ↓
Conv2D(256 → 256) + BatchNorm + GELU
    ↓
Conv2D(256 → 10) [1×1 conv for classification]
    ↓
Bilinear Upsampling to original resolution
    ↓
Output: [B, 10, H, W] logits
```

### Path Planning: A* Algorithm
- **Heuristic**: Euclidean distance
- **Grid**: 8-connected (includes diagonals)
- **Cost**: 1.0 for orthogonal, 1.414 for diagonal moves
- **Optimizations**: Early termination, closed set tracking

## 📊 Performance

- **Segmentation Speed**: ~2-3 seconds per image (GPU)
- **Path Planning**: <0.1 seconds for typical image
- **Model Size**: ~87MB (segmentation head only)
- **Input Resolution**: 476×266 (adjustable)

## 🛠️ Training

To train the segmentation model:

```bash
python train_segmentation.py
```

Configuration parameters in the script:
- Learning rate: 1e-4
- Batch size: 8
- Epochs: 50
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau

## 🔧 Customization

### Adding New Terrain Classes

1. Update `NUM_CLASSES` in `model_inference.py`
2. Add colors to `colors` array
3. Update `OBSTACLE_MAP` dictionary
4. Retrain model with new labels

### Adjusting Path Planning

Modify parameters in `path_planner.py`:
- Change heuristic function
- Adjust movement costs
- Modify neighbor search radius

## 📈 Future Improvements

- [ ] Real-time video processing
- [ ] Multi-scale segmentation
- [ ] Advanced path smoothing
- [ ] Integration with robot control systems
- [ ] 3D terrain reconstruction
- [ ] Dynamic obstacle handling

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **DINOv2**: Meta AI Research - Self-supervised vision transformers
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for autonomous navigation research**
