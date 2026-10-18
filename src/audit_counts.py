"""Count images per class at three stages (raw, after the first split, after preprocessing)
and list the images that preprocessing dropped because YOLO found no leaf.

Writes class_counts_initial.csv and dropped_images.csv into --out (default: current folder).

Usage:
    python src/audit_counts.py --raw data/raw_dataset --split data/tomato_disease_cnn --processed data/processed
"""
import argparse
import csv
import os
from collections import defaultdict

EXT = (".jpg", ".jpeg", ".png")


def files(path):
    return {f for f in os.listdir(path) if f.lower().endswith(EXT)} if os.path.isdir(path) else set()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="raw dataset: class/images")
    p.add_argument("--split", required=True, help="first split: split/class/images")
    p.add_argument("--processed", required=True, help="preprocessed images: split/class/images")
    p.add_argument("--out", default=".", help="where to write the CSV files")
    args = p.parse_args()

    classes = sorted(d for d in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, d)))
    rows, dropped = [], []

    for c in classes:
        raw_n = len(files(os.path.join(args.raw, c)))
        for s in ["train", "val", "test"]:
            split_set = files(os.path.join(args.split, s, c))
            proc_set = files(os.path.join(args.processed, s, c))
            rows.append([c, s, raw_n, len(split_set), len(proc_set)])
            for f in sorted(split_set - proc_set):
                dropped.append([c, s, f])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "class_counts_initial.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "split", "raw_total_for_class", "after_split", "after_preprocess"])
        w.writerows(rows)

    with open(os.path.join(args.out, "dropped_images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "split", "filename"])
        w.writerows(dropped)

    tot = defaultdict(lambda: [0, 0])
    for c, s, _, a, b in rows:
        tot[s][0] += a
        tot[s][1] += b
    for s, (a, b) in tot.items():
        print(f"{s}: split={a} processed={b} dropped={a - b}")
    print(f"Total dropped: {len(dropped)} (see dropped_images.csv)")


if __name__ == "__main__":
    main()
