"""Initial train/val/test split (70/15/15 per class) of the raw PlantVillage tomato folders.

NOTE: the final split of the published dataset is made later by dedup_resplit.py, which keeps
near-duplicate images together. This script only creates the first split that
preprocess_dataset.py reads. (The very first run for the published dataset was not seeded.)

Usage:
    python src/split_tomato_dataset.py --source data/raw_dataset --dest data/tomato_disease_cnn --seed 42

--source  folder with one sub-folder per class (the 10 tomato class folders)
--dest    output folder, must not exist yet or must be empty
"""
import argparse
import os
import random
import shutil

SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test
EXTS = (".jpg", ".png", ".jpeg")


def make_split_folders(dest, categories):
    for split in ["train", "val", "test"]:
        for category in categories:
            os.makedirs(os.path.join(dest, split, category), exist_ok=True)


def split_images(images, rng):
    rng.shuffle(images)
    total = len(images)
    train_end = int(SPLIT_RATIOS[0] * total)
    val_end = train_end + int(SPLIT_RATIOS[1] * total)

    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="raw dataset folder (class/images)")
    parser.add_argument("--dest", required=True, help="output folder (split/class/images)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (omit for an unseeded shuffle)")
    args = parser.parse_args()

    if os.path.isdir(args.dest) and os.listdir(args.dest):
        raise SystemExit(f"Destination {args.dest} already exists and is not empty.")

    rng = random.Random(args.seed)
    categories = sorted(
        d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))
    )
    if not categories:
        raise SystemExit(f"No class folders found in {args.source}")
    make_split_folders(args.dest, categories)

    for category in categories:
        cat_path = os.path.join(args.source, category)
        images = sorted(f for f in os.listdir(cat_path) if f.lower().endswith(EXTS))

        for split_type, split_files in split_images(images, rng).items():
            for img in split_files:
                src = os.path.join(cat_path, img)
                dst = os.path.join(args.dest, split_type, category, img)
                shutil.copyfile(src, dst)

    print("Dataset successfully split into train/val/test folders.")


if __name__ == "__main__":
    main()
