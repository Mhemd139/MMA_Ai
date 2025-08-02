import os
import json
import cv2
import numpy as np

# Paths
FRAME_DIR = "data/frames"
ANNOTATED_DIR = "outputs/annotated"
VIDEO_PATH = "outputs/annotated_video.mp4"
DETECTIONS_FILE = "outputs/detections.json"

# Initialize counters
punches_blocked = 0
punches_landed = 0

# Frame size (we get from first frame)
frame_size = None
frame_list = []

# Custom colors
COLORS = {
    "punch": (0, 0, 255),         # Red
    "kick": (255, 0, 0),          # Blue
    "person": (0, 255, 0),        # Green
    "high guard": (255, 0, 0),    # Blue
    "low guard": (0, 255, 255),   # Yellow
    "face": (255, 255, 0),        # Cyan
    "blocked_punch": (0, 255, 0), # Green for blocked
    "landed_punch": (0, 0, 255)   # Red for landed
}

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def detect_face_in_person(image, person_box):
    """Detect face within person bounding box using OpenCV's Haar cascade"""
    x1, y1, x2, y2 = person_box
    
    # Extract person region
    person_region = image[y1:y2, x1:x2]
    if person_region.size == 0:
        return None
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        # Get the largest face (most likely the main face)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        fx, fy, fw, fh = largest_face
        
        # Convert back to original image coordinates
        face_box = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
        return face_box
    
    return None

def draw_counter(image, punches_blocked, punches_landed):
    """Draw punch counters on the image"""
    # Create a semi-transparent overlay for counters
    overlay = image.copy()
    
    # Draw background rectangles
    cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 90), (300, 160), (0, 0, 0), -1)
    
    # Apply transparency
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Punches Blocked counter
    blocked_text = f"Punches Blocked: {punches_blocked}"
    cv2.putText(image, blocked_text, (20, 45), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Punches Landed counter
    landed_text = f"Punches Landed: {punches_landed}"
    cv2.putText(image, landed_text, (20, 125), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    return image

# Load existing detections
print("📂 Loading existing detections...")
with open(DETECTIONS_FILE, 'r') as f:
    detections_data = json.load(f)

print(f"📊 Loaded {len(detections_data)} frames of detection data")

# Process each frame
for frame_data in detections_data:
    frame_file = frame_data["frame"]
    predictions = frame_data["predictions"]
    
    frame_path = os.path.join(FRAME_DIR, frame_file)
    print(f"🔍 Processing: {frame_file}")
    
    # Check if frame exists
    if not os.path.exists(frame_path):
        print(f"⚠️  Frame {frame_file} not found, skipping...")
        continue
    
    image = cv2.imread(frame_path)
    if frame_size is None:
        height, width, _ = image.shape
        frame_size = (width, height)

    # Store detections by type for overlap analysis
    punches = []
    high_guards = []
    low_guards = []
    persons = []
    faces = []

    # First pass: collect all detections
    for pred in predictions:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])
        class_name = pred["class"]
        confidence = pred["confidence"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        box = (x1, y1, x2, y2)
        
        if class_name.lower() == "punch":
            punches.append((box, confidence))
        elif class_name.lower() == "high guard":
            high_guards.append((box, confidence))
        elif class_name.lower() == "low guard":
            low_guards.append((box, confidence))
        elif class_name.lower() == "person":
            persons.append((box, confidence))

    # Second pass: detect faces in person bounding boxes
    for person_box, person_conf in persons:
        face_box = detect_face_in_person(image, person_box)
        if face_box:
            faces.append((face_box, person_conf))

    # Third pass: analyze overlaps and draw everything
    for pred in predictions:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])
        class_name = pred["class"]
        confidence = pred["confidence"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        color = COLORS.get(class_name.lower(), (200, 200, 200))

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw text
        label = f"{class_name.upper()} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_origin = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)

        cv2.putText(image, label, text_origin, font, font_scale, color, thickness + 1, cv2.LINE_AA)

    # Draw detected faces
    for face_box, face_conf in faces:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS["face"], 2)
        cv2.putText(image, f"FACE {face_conf:.2f}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["face"], 2, cv2.LINE_AA)

    # Analyze punch overlaps
    for punch_box, punch_conf in punches:
        punch_x1, punch_y1, punch_x2, punch_y2 = punch_box
        
        # Check for high guard overlap (blocked punch)
        for high_guard_box, high_guard_conf in high_guards:
            iou = calculate_iou(punch_box, high_guard_box)
            if iou > 0.1:  # Threshold for overlap
                punches_blocked += 1
                # Highlight the blocked punch
                cv2.rectangle(image, (punch_x1, punch_y1), (punch_x2, punch_y2), COLORS["blocked_punch"], 3)
                cv2.putText(image, "BLOCKED!", (punch_x1, punch_y1 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["blocked_punch"], 3, cv2.LINE_AA)
                break
        
        # Check for low guard overlap (landed punch)
        for low_guard_box, low_guard_conf in low_guards:
            iou = calculate_iou(punch_box, low_guard_box)
            if iou > 0.1:  # Threshold for overlap
                punches_landed += 1
                # Highlight the landed punch
                cv2.rectangle(image, (punch_x1, punch_y1), (punch_x2, punch_y2), COLORS["landed_punch"], 3)
                cv2.putText(image, "LANDED!", (punch_x1, punch_y1 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["landed_punch"], 3, cv2.LINE_AA)
                break

    # Draw counters on the frame
    image = draw_counter(image, punches_blocked, punches_landed)

    # Save frame
    annotated_path = os.path.join(ANNOTATED_DIR, frame_file)
    cv2.imwrite(annotated_path, image)
    frame_list.append(image)

# Save video
print("🎥 Saving video...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_PATH, fourcc, 10, frame_size)  # 10 FPS
for frame in frame_list:
    out.write(frame)
out.release()

print(f"✅ All frames processed. Final counts:")
print(f"   Punches Blocked: {punches_blocked}")
print(f"   Punches Landed: {punches_landed}")
print("   Video saved to:", VIDEO_PATH) 