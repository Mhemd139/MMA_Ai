import cv2
import os
from ultralytics import YOLO

# Load local YOLO model
model = YOLO('yolov8n.pt')  # Using the nano model you have

FRAME_DIR = "data/frames"

# Counters for what was detected
total_detections = {
    "person": 0,
    "other": 0
}

print("🔍 Testing local YOLO model on first 5 frames...")

# Check first 5 frames
frame_count = 0
for frame_file in sorted(os.listdir(FRAME_DIR)):
    if not frame_file.lower().endswith(".jpg"):
        continue
    
    if frame_count >= 5:  # Only check first 5 frames
        break
        
    frame_path = os.path.join(FRAME_DIR, frame_file)
    print(f"🔍 Analyzing: {frame_file}")
    
    try:
        # Run inference
        results = model(frame_path)
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    if class_name == "person":
                        total_detections["person"] += 1
                        print(f"   Found person with confidence {conf:.2f}")
                    else:
                        total_detections["other"] += 1
                        print(f"   Found {class_name} with confidence {conf:.2f}")
        
        frame_count += 1
        
    except Exception as e:
        print(f"❌ Error processing {frame_file}: {e}")

print("\n📊 Local YOLO Detection Summary (first 5 frames):")
for class_name, count in total_detections.items():
    print(f"   {class_name}: {count}")

print(f"\n💡 The local YOLO model can detect people, but it's not trained for combat sports.")
print(f"   The Roboflow combat sports model should be better, but it's not detecting punches/guards.")
print(f"   This suggests your video might not contain the specific actions the model was trained on.") 