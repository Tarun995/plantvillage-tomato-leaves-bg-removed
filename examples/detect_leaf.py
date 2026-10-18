"""Inference example: find leaves in a photo, crop each one and remove its background.

This is NOT used to build the dataset (see src/preprocess_dataset.py). It shows how the same
two models can be used on new photos, for example in an app.

Differences from the dataset pipeline, on purpose for a demo:
  - it keeps every detection (the dataset pipeline keeps only the top box)
  - it uses a low confidence threshold of 0.1 (the dataset pipeline uses the YOLO default)
The colour handling matches the dataset pipeline (U2-Net gets RGB).

Usage:
    python examples/detect_leaf.py path/to/photo.jpg
Weights are read from models/best.pt and models/u2net.pth, or from the environment variables
YOLO_WEIGHTS and U2NET_WEIGHTS.
"""
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from u2net_remove_bg import load_model, u2net_remove_background_from_numpy  # noqa: E402

YOLO_MODEL_PATH = os.environ.get("YOLO_WEIGHTS", str(ROOT / "models" / "best.pt"))
U2NET_MODEL_PATH = os.environ.get("U2NET_WEIGHTS", str(ROOT / "models" / "u2net.pth"))

# Load both models once
YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
U2NET_MODEL = load_model(U2NET_MODEL_PATH)


def crop_all_leaves(img_path, output_folder="cropped_leaves"):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    results = YOLO_MODEL(img, conf=0.1)[0]
    overlay = results.plot()  # image with bounding boxes

    os.makedirs(output_folder, exist_ok=True)
    boxes = results.boxes.xyxy

    if boxes is None or len(boxes) == 0:
        raise ValueError("No leaf detected.")

    saved_paths = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped = img[y1:y2, x1:x2]

        # Remove the background with U2-Net (RGB input, as in the dataset pipeline)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        clean_rgb = u2net_remove_background_from_numpy(U2NET_MODEL, cropped_rgb)

        # Resize to the classifier input size and save (cv2 writes BGR)
        resized = cv2.resize(clean_rgb, (224, 224))
        save_path = os.path.join(output_folder, f"leaf_{i + 1}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        saved_paths.append((cropped, save_path))  # raw crop and path of the cleaned image

    overlay_path = os.path.join(output_folder, "yolo_detected.jpg")
    cv2.imwrite(overlay_path, overlay)

    return overlay_path, saved_paths


def run_leaf_detection(img_path):
    return crop_all_leaves(img_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python examples/detect_leaf.py path/to/photo.jpg")
    overlay, leaves = run_leaf_detection(sys.argv[1])
    print(f"Saved detections to {overlay}")
    for _, path in leaves:
        print(f"Saved cleaned leaf: {path}")
