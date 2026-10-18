"""Build md5_manifest.csv (path, split, class, md5) for a dataset folder and check for exact
duplicates (identical file bytes) that appear in more than one split.

Usage:
    python build_manifest.py "path/to/processed"
"""
import csv
import hashlib
import os
import sys
from collections import defaultdict

EXT = (".jpg", ".jpeg", ".png")

if len(sys.argv) < 2:
    sys.exit('Usage: python build_manifest.py "path/to/dataset"')
root = sys.argv[1]

rows = []
by_hash = defaultdict(list)

for split in ("train", "val", "test"):
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        continue
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir)):
            if not name.lower().endswith(EXT):
                continue
            with open(os.path.join(cls_dir, name), "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
            rel = f"{split}/{cls}/{name}"
            rows.append([rel, split, cls, name, digest])
            by_hash[digest].append((split, cls, name))

with open("md5_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "split", "class", "filename", "md5"])
    w.writerows(rows)

cross = []
for digest, items in by_hash.items():
    if len({s for s, _, _ in items}) > 1:
        for s, c, n in items:
            cross.append([digest, s, c, n])

with open("duplicates_across_splits.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["md5", "split", "class", "filename"])
    w.writerows(cross)

same_split = sum(1 for items in by_hash.values()
                 if len(items) > 1 and len({s for s, _, _ in items}) == 1)

print(f"Images listed: {len(rows)}")
print(f"Hash groups duplicated across splits: {len({r[0] for r in cross})}")
print(f"Hash groups duplicated within one split: {same_split}")
print("Wrote md5_manifest.csv and duplicates_across_splits.csv")
