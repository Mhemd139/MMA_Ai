# MMA AI Punch Detection

A computer vision pipeline that analyzes MMA fight footage to detect punches and classify each one as **landed** or **blocked**. It combines a YOLO object detector (served via the Roboflow Inference API) with a fine-tuned ResNet-50 image classifier, and produces an annotated video with running punch statistics.

## How It Works

The pipeline processes a fight video frame by frame:

1. **Frame extraction** — evenly spaced frames are sampled from the source video.
2. **Object detection** — a YOLO model trained on a combat sports dataset (`combat-sports-dataset/2`, Roboflow) detects fighters, punches, kicks, and high/low guard positions in each frame.
3. **Face localization** — an OpenCV Haar cascade locates faces within each detected fighter's bounding box.
4. **Punch classification** — when a punch bounding box overlaps a face, the punch region is cropped and classified as *blocked* or *landed* by a ResNet-50 model fine-tuned on labeled fight frames.
5. **Temporal tracking** — a tracker de-duplicates consecutive detections of the same punch (state resets after 4 inactive frames) so each punch is counted once.
6. **Output** — frames are annotated with bounding boxes, confidence scores, and running landed/blocked counters, then assembled into an MP4 alongside a JSON file of raw detections.

## Repository Structure

```text
MMA_Ai/
├── scripts/
│   ├── extract_frames.py              # Sample evenly spaced frames from a video
│   ├── run_yolo_detection.py          # Main pipeline: YOLO + ResNet-50 classification
│   ├── yolo_detection.py              # YOLO-only detection and annotation
│   ├── process_existing_detections.py # Re-analyze saved detections without API calls
│   └── load_huggingface_model.py      # PunchClassifier wrapper and model test
├── UFC_huggingface_training.ipynb     # Colab notebook for fine-tuning ResNet-50
├── requirements.txt
└── LICENSE
```

The `data/`, `models/`, and `outputs/` directories are created at runtime and are not tracked in git.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/Mhemd139/MMA_Ai.git
cd MMA_Ai

pip install -r requirements.txt
pip install inference-sdk   # Roboflow client used by the detection scripts
```

Create a `.env` file in the project root with your Roboflow API key:

```text
ROBOFLOW_API_KEY=your_api_key_here
```

## Usage

### 1. Extract frames

Edit the video path at the bottom of `scripts/extract_frames.py`, then run:

```bash
python scripts/extract_frames.py
```

Frames are written to `data/frames/`.

### 2. Run detection

```bash
python scripts/run_yolo_detection.py
```

For each frame this calls the Roboflow API for object detection and, if a fine-tuned classifier is present at `models/punch-detection-model`, classifies punch-to-face contacts as landed or blocked. Without the classifier it falls back to YOLO detection only.

Outputs:

- `outputs/annotated/` — annotated frames
- `outputs/annotated_video.mp4` — assembled video with punch counters
- `outputs/detections.json` — raw per-frame predictions

### 3. Re-process saved detections (optional)

To iterate on the analysis logic without repeating API calls:

```bash
python scripts/process_existing_detections.py
```

This re-runs the punch analysis using the predictions stored in `outputs/detections.json`.

## Training the Classifier

The punch classifier is `microsoft/resnet-50` fine-tuned for binary classification (blocked vs. landed) on manually labeled fight frames — 37 blocked and 25 landed examples.

1. Package the labeled frames as `mma_data.zip` and upload it to Google Colab.
2. Run `UFC_huggingface_training.ipynb` (trains for 10 epochs, batch size 4, with a train/validation split and accuracy/loss plots).
3. Download the trained model into `models/punch-detection-model/`.

`scripts/load_huggingface_model.py` provides a standalone `PunchClassifier` class and a quick sanity test for the trained model.

## Tech Stack

- **PyTorch / Hugging Face Transformers** — ResNet-50 fine-tuning and inference
- **Roboflow Inference API** — YOLO object detection
- **OpenCV** — frame extraction, face detection, annotation, video encoding

## Acknowledgments

- [Roboflow](https://roboflow.com/) for the combat sports detection model
- [Hugging Face](https://huggingface.co/) for the Transformers library and model hub

## License

[MIT](LICENSE)
