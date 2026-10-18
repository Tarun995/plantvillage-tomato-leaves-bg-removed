"""Crop each leaf with a YOLOv8 detector, remove the background with U2-Net, resize to 224x224.

Reads  <input>/{train,val,test}/<class>/<images>
Writes <output>/{train,val,test}/<class>/<images>   (same file names)

Images where YOLO finds no leaf are skipped (they are listed later by audit_counts.py).

Usage:
    python src/preprocess_dataset.py --input data/tomato_disease_cnn --output data/processed

Needs the two weight files described in models/README.md, and the U2-Net model definition
saved as src/u2net/u2net.py.
"""
import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from u2net_remove_bg import load_model as load_u2net_model, u2net_remove_background_from_numpy

ROOT = Path(__file__).resolve().parents[1]


def preprocess_and_save(yolo, u2net, input_dir, output_dir, img_size):
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        save_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        for img_name in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)

                results = yolo(img)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                if len(boxes) == 0:
                    continue

                x1, y1, x2, y2 = boxes[0]
                cropped = img[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cleaned = u2net_remove_background_from_numpy(u2net, cropped_rgb)
                cleaned = cv2.resize(cleaned, img_size)

                save_path = os.path.join(save_class_dir, img_name)
                cv2.imwrite(save_path, cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"[WARN] Skipped {img_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="split folder with train/val/test/<class>/<images>")
    parser.add_argument("--output", required=True, help="output folder for the processed images")
    parser.add_argument("--yolo-weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--u2net-weights", default=str(ROOT / "models" / "u2net.pth"))
    parser.add_argument("--size", type=int, default=224, help="output image size (square)")
    args = parser.parse_args()

    yolo = YOLO(args.yolo_weights)
    u2net = load_u2net_model(args.u2net_weights)
    img_size = (args.size, args.size)

    for split in ["train", "val", "test"]:
        print(f"\nProcessing: {split}")
        preprocess_and_save(yolo, u2net, os.path.join(args.input, split), os.path.join(args.output, split), img_size)

    print("\nAll images preprocessed and saved.")


if __name__ == "__main__":
    main()
