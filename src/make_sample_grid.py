"""Make a before/after picture: one random image per class, original (left) and processed (right).

Use it for the README preview (assets/sample_grid.png) and as the Kaggle cover image.

Usage:
    python src/make_sample_grid.py --raw data/raw_dataset --processed data/final_dataset --out assets/sample_grid.png
"""
import argparse
import math
import os
import random

from PIL import Image, ImageDraw

EXT = (".jpg", ".jpeg", ".png")


def pick_pair(raw_root, proc_root, split, cls, rng):
    proc_dir = os.path.join(proc_root, split, cls)
    names = sorted(f for f in os.listdir(proc_dir) if f.lower().endswith(EXT))
    rng.shuffle(names)
    for name in names:
        raw_path = os.path.join(raw_root, cls, name)
        if os.path.exists(raw_path):
            return raw_path, os.path.join(proc_dir, name)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="raw dataset: class/images (original file names)")
    p.add_argument("--processed", required=True, help="dataset folder: split/class/images")
    p.add_argument("--split", default="test", help="which split to sample from")
    p.add_argument("--out", default="assets/sample_grid.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cell", type=int, default=160, help="size of each image in pixels")
    p.add_argument("--cols", type=int, default=2, help="how many original/processed pairs per row")
    args = p.parse_args()

    rng = random.Random(args.seed)
    classes = sorted(d for d in os.listdir(os.path.join(args.processed, args.split))
                     if os.path.isdir(os.path.join(args.processed, args.split, d)))
    pairs = []
    for cls in classes:
        pair = pick_pair(args.raw, args.processed, args.split, cls, rng)
        if pair is None:
            print(f"[WARN] no original found for class {cls}, skipped")
            continue
        pairs.append((cls, *pair))
    if not pairs:
        raise SystemExit("No image pairs found. Check --raw and --processed.")

    cell, label_h, gap, margin, head = args.cell, 18, 12, 10, 24
    rows = math.ceil(len(pairs) / args.cols)
    pair_w = 2 * cell
    width = margin * 2 + args.cols * pair_w + (args.cols - 1) * gap
    height = head + margin + rows * (cell + label_h) + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 6), "Original (left) and processed (right), one image per class", fill="black")

    for k, (cls, raw_path, proc_path) in enumerate(pairs):
        r, c = divmod(k, args.cols)
        x = margin + c * (pair_w + gap)
        y = head + r * (cell + label_h + gap)
        draw.text((x, y), cls.replace("Tomato___", "").replace("_", " "), fill="black")
        for off, path in enumerate((raw_path, proc_path)):
            with Image.open(path) as im:
                sheet.paste(im.convert("RGB").resize((cell, cell)), (x + off * cell, y + label_h))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    print(f"Saved {args.out} ({len(pairs)} classes)")


if __name__ == "__main__":
    main()
