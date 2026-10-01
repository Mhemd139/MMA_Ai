import cv2
import os
import sys

def extract_fixed_frame_count(video_path, output_dir, target_frames=300):
    os.makedirs(output_dir, exist_ok=True)

    # === Delete all old JPG frames ===
    for file in os.listdir(output_dir):
        if file.lower().endswith(".jpg"):
            os.remove(os.path.join(output_dir, file))

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps == 0 or total_frames == 0:
        raise ValueError("❌ Could not read video properties.")

    duration = total_frames / original_fps
    print(f"🎞️ Video length: {duration:.2f}s, FPS: {original_fps:.2f}, Total frames: {total_frames}")

    # Compute spacing between selected frames
    spacing = max(int(total_frames / target_frames), 1)
    print(f"📸 Extracting 1 frame every {spacing} frames (~{target_frames} total)")

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % spacing == 0 and saved_count < target_frames:
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {saved_count} frames to '{output_dir}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/extract_frames.py <video.mp4>")
    extract_fixed_frame_count(sys.argv[1], "data/frames", target_frames=500)
