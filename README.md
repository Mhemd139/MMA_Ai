# 🥊 MMA AI Punch Detection System

An advanced computer vision system that combines YOLO object detection with Hugging Face fine-tuned models to analyze MMA fights and detect punch effectiveness in real-time.

## 🚀 Features

- **Real-time Punch Detection**: Uses YOLO to detect punches, guards, and fighters
- **Punch Effectiveness Analysis**: Determines if punches are "landed" or "blocked"
- **Temporal Guard Tracking**: Maintains guard state across multiple frames
- **Hugging Face Integration**: Fine-tuned ResNet-50 model for punch classification
- **Video Processing**: Processes video frames and generates annotated output
- **Counter System**: Tracks landed vs blocked punches with visual counters

## 🛠️ Technical Stack

- **Computer Vision**: OpenCV, YOLO (Roboflow)
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Object Detection**: Roboflow Inference API
- **Video Processing**: OpenCV VideoWriter
- **Data Processing**: NumPy, PIL (Pillow)

## 📁 Project Structure

```
mma-ai-project/
├── scripts/
│   ├── run_yolo_detection.py      # Main detection script
│   ├── extract_frames.py          # Video frame extraction
│   ├── load_huggingface_model.py  # Model loading utility
│   └── enhanced_punch_detection.py # Advanced detection features
├── data/
│   ├── frames/                    # Extracted video frames
│   ├── blocked_punches/          # Training data for blocked punches
│   └── landed_punches/           # Training data for landed punches
├── models/                        # Fine-tuned model storage
├── outputs/                       # Generated annotated videos
├── UFC_huggingface_training.ipynb # Colab training notebook
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🎯 Core Functionality

### 1. YOLO Detection
- Detects fighters, punches, and guard positions
- Uses Roboflow's combat sports dataset
- Real-time bounding box visualization

### 2. Punch Analysis
- **Intersection Detection**: Analyzes punch-guard intersections
- **Temporal Tracking**: Maintains guard states across frames
- **Cooldown System**: Prevents duplicate punch counting
- **Effectiveness Classification**: Determines landed vs blocked

### 3. Hugging Face Integration
- Fine-tuned ResNet-50 model for punch classification
- Custom dataset with blocked/landed punch examples
- Real-time inference during video processing

## 📊 Dataset

The system uses a custom dataset with:
- **Blocked Punches**: 37 frames from various fight sequences
- **Landed Punches**: 25 frames showing successful hits
- **Training**: Fine-tuned on Google Colab with GPU acceleration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mhemd139/MMA_Ai.git
cd MMA_Ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your Roboflow API key to .env
```

### Usage

1. **Extract Frames from Video**:
```bash
python scripts/extract_frames.py
```

2. **Run Detection**:
```bash
python scripts/run_yolo_detection.py
```

3. **Train Custom Model** (Optional):
   - Upload `mma_data.zip` to Google Colab
   - Run `UFC_huggingface_training.ipynb`
   - Download trained model to `models/`

## 🔧 Configuration

### Environment Variables
```bash
ROBOFLOW_API_KEY=your_api_key_here
MODEL_ID=combat-sports-dataset/2
```

### Detection Parameters
- **IoU Threshold**: 0.3 for intersection detection
- **Cooldown Frames**: 3 frames between punch counts
- **Guard Persistence**: 2 frames for temporal tracking

## 📈 Results

The system provides:
- **Annotated Videos**: With bounding boxes and counters
- **Punch Statistics**: Landed vs blocked counts
- **Real-time Analysis**: Frame-by-frame processing
- **Confidence Scores**: For each detection

## 🎓 Training Process

1. **Data Collection**: Manually labeled frames from fight videos
2. **Dataset Preparation**: Organized into blocked/landed categories
3. **Model Fine-tuning**: ResNet-50 on Google Colab
4. **Evaluation**: Accuracy metrics and validation
5. **Integration**: Loaded into main detection pipeline



## 🙏 Acknowledgments

- **Roboflow**: For the combat sports YOLO model
- **Hugging Face**: For the transformers library
- **OpenCV**: For computer vision capabilities
- **PyTorch**: For deep learning framework

---

⭐ **Star this repository if you find it useful!**
