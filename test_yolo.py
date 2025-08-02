from ultralytics import YOLO
import cv2

# Load a pretrained model
model = YOLO("yolov8n.pt")  # small and fast

# Run inference on an image
results = model("C:\\Users\\VagaBond\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-06-22 194835.png")

# Display results
results[0].show()  # show with OpenCV
