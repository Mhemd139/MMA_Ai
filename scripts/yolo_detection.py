import os
import json
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# === ENV + SETUP ===
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

FRAME_DIR = "data/frames"
ANNOTATED_DIR = "outputs/annotated"
VIDEO_PATH = "outputs/annotated_video.mp4"
os.makedirs(ANNOTATED_DIR, exist_ok=True)
ANNOTATED_DIR = "outputs/annotated"
VIDEO_PATH = "outputs/annotated_video.mp4"
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# === Clean Annotated Output ===
for f in os.listdir(ANNOTATED_DIR):
    if f.endswith(".jpg"):
        os.remove(os.path.join(ANNOTATED_DIR, f))
# === Clean old .mp4 videos from outputs ===
for f in os.listdir("outputs"):
    if f.endswith(".mp4"):
        os.remove(os.path.join("outputs", f))
MODEL_ID = "combat-sports-dataset/2"
metadata = []

# === Class Colors (BGR) ===
COLORS = {
    "punch": (0, 0, 255),
    "kick": (255, 0, 0),
    "person": (0, 255, 0),
    "high guard": (255, 0, 0),
    "low guard": (0, 255, 255)
}

# === Counter Drawing Function ===
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
    
    # Punches Blocked counter (Green)
    blocked_text = f"Punches Blocked: {punches_blocked}"
    cv2.putText(image, blocked_text, (20, 45), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Punches Landed counter (Red)
    landed_text = f"Punches Landed: {punches_landed}"
    cv2.putText(image, landed_text, (20, 125), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    return image

# === Face Detection ===
def detect_face_in_person(image, person_box):
    x1, y1, x2, y2 = person_box
    person_region = image[y1:y2, x1:x2]
    if person_region.size == 0:
        return None
    gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda x: x[2] * x[3])
        return (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
    return None

# === Helpers ===
def to_box_coords(obj):
    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
    return (x - w // 2, y - h // 2, x + w // 2, y + h // 2)

def boxes_intersect(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return inter_x2 > inter_x1 and inter_y2 > inter_y1

class PunchTracker:
    def __init__(self, max_inactive_frames=4):
        self.prev_status = None
        self.inactive_count = 0
        self.max_inactive = max_inactive_frames

    def update(self, current_status):
        if current_status != self.prev_status:
            self.inactive_count += 1
            if self.inactive_count > self.max_inactive:
                self.prev_status = None
                self.inactive_count = 0

        if current_status and current_status != self.prev_status:
            self.prev_status = current_status
            self.inactive_count = 0
            return current_status  # count this punch
        return None

# === Tracking State ===
landed_count = 0
blocked_count = 0
frame_list = []
frame_size = None
tracker = PunchTracker()

# === Main Loop ===
for frame_file in sorted(os.listdir(FRAME_DIR)):
    if not frame_file.lower().endswith(".jpg"):
        continue

    frame_path = os.path.join(FRAME_DIR, frame_file)
    image = cv2.imread(frame_path)
    if frame_size is None:
        h, w, _ = image.shape
        frame_size = (w, h)

    result = CLIENT.infer(frame_path, model_id=MODEL_ID)
    predictions = result["predictions"]

    persons = [to_box_coords(p) for p in predictions if p["class"] == "person"]
    punches = [to_box_coords(p) for p in predictions if p["class"] == "punch"]
    guards = [(to_box_coords(p), p["class"]) for p in predictions if "guard" in p["class"]]

    faces = [detect_face_in_person(image, pbox) for pbox in persons]

    # === Punch Logic ===
    punch_status = None
    for punch_box in punches:
        for face_box in faces:
            if face_box and boxes_intersect(punch_box, face_box):
                blocked = False
                for guard_box, guard_type in guards:
                    if boxes_intersect(punch_box, guard_box):
                        if "high" in guard_type:
                            punch_status = "blocked"
                            blocked = True
                            break
                        elif "low" in guard_type:
                            punch_status = "landed"
                if not blocked and punch_status is None:
                    punch_status = "landed"

    counted = tracker.update(punch_status)
    if counted == "landed":
        landed_count += 1
    elif counted == "blocked":
        blocked_count += 1

    # === Draw Boxes ===
    for obj in predictions:
        cls = obj["class"]
        conf = obj["confidence"]
        x1, y1, x2, y2 = to_box_coords(obj)
        color = COLORS.get(cls.lower(), (200, 200, 200))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{cls.upper()} {conf:.2f}"
        cv2.putText(image, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    for face_box in faces:
        if face_box:
            fx1, fy1, fx2, fy2 = face_box
            cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 0, 255), 1)
            cv2.putText(image, "FACE", (fx1, max(fy1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # === Draw Counter Image ===
    image = draw_counter(image, blocked_count, landed_count)

    # Save frame
    cv2.imwrite(os.path.join(ANNOTATED_DIR, frame_file), image)
    frame_list.append(image)
    metadata.append({"frame": frame_file, "predictions": predictions})

# === Save Metadata + Video ===
with open("outputs/detections.json", "w") as f:
    json.dump(metadata, f, indent=2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_PATH, fourcc, 10, frame_size)
for f in frame_list:
    out.write(f)
out.release()

print("✅ Done: annotated video + metadata + punch analysis saved.")