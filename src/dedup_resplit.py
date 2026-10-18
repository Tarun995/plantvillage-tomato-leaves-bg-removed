"""
dedup_resplit.py - find near-duplicate leaf photos and re-split the dataset by group,
so that near-identical images never end up in different splits.

Run the steps in this order:
  1) embed     compute and cache a CNN embedding for every processed image
  2) analyze   sweep similarity thresholds: how many images are near-duplicates, and how
               many val/test images have a near-duplicate in train (the leakage)
  3) preview   contact sheets of image pairs just above / just below a threshold,
               so you can pick the threshold by eye
  4) resplit   group-aware 70/15/15 split per class. Use --dry-run first. The real run
               COPIES files into a new folder; your current dataset is never modified.

Examples:
  python dedup_resplit.py embed   --root "path/to/processed"
  python dedup_resplit.py analyze
  python dedup_resplit.py preview --threshold 0.95
  python dedup_resplit.py resplit --threshold 0.95 --dry-run
  python dedup_resplit.py resplit --threshold 0.95 --out "path/to/final_dataset"

Duplicates are only searched for within the same class.
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict

import numpy as np

EXT = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EMB_FILE = "embeddings.npz"
DEFAULT_SWEEP = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
MAX_EDGES = 40_000_000  # safety stop if the threshold is so low that nearly everything matches


# ----------------------------------------------------------------------------- helpers
def list_images(root):
    items = []
    for split in SPLITS:
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            for name in sorted(os.listdir(cdir)):
                if name.lower().endswith(EXT):
                    items.append((f"{split}/{cls}/{name}", split, cls, name))
    return items


def load_state(root_override=None):
    if not os.path.exists(EMB_FILE):
        sys.exit(f"{EMB_FILE} not found. Run the 'embed' step first (in this same folder).")
    z = np.load(EMB_FILE, allow_pickle=False)
    rel = [str(r) for r in z["rel"]]
    emb = z["emb"]
    root = root_override or str(z["root"])
    split = np.array([r.split("/")[0] for r in rel])
    cls = np.array([r.split("/")[1] for r in rel])
    return root, rel, emb, split, cls


# ----------------------------------------------------------------------------- 1) embed
def cmd_embed(args):
    import torch
    from PIL import Image
    from torchvision import models

    items = list_images(args.root)
    if not items:
        sys.exit(f"No images found under {args.root} (expected train/val/test/<class>/<images>).")
    if os.path.exists(EMB_FILE) and not args.force:
        sys.exit(f"{EMB_FILE} already exists. Delete it or pass --force to recompute.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(items)} images, device={device}, views={args.views}")
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    out = np.zeros((len(items), args.views, 2048), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(items), args.batch):
            chunk = items[start:start + args.batch]
            arr = []
            for rel, _, _, _ in chunk:
                with Image.open(os.path.join(args.root, rel)) as im:
                    arr.append(np.asarray(im.convert("RGB").resize((224, 224)), dtype=np.uint8))
            x = torch.from_numpy(np.stack(arr)).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            views = [x, x.flip(3), x.flip(2), torch.rot90(x, 1, (2, 3))][:args.views]
            for v, xv in enumerate(views):
                out[start:start + len(chunk), v] = net(xv).cpu().numpy().astype(np.float16)
            done = start + len(chunk)
            if (done // args.batch) % 20 == 0 or done == len(items):
                print(f"  embedded {done}/{len(items)}", flush=True)

    np.savez(EMB_FILE, rel=np.array([it[0] for it in items]), emb=out, root=args.root)
    print(f"Saved {EMB_FILE}")


# ----------------------------------------------------------------------------- similarity
def class_edges(E, tmin, chunk=512):
    """E: (n, V, D) embeddings of one class. Returns (I, J, S): index pairs whose cosine
    similarity (best over the V views of the second image) is >= tmin."""
    n, V, D = E.shape
    X = E.astype(np.float32)
    X /= np.linalg.norm(X, axis=2, keepdims=True) + 1e-8
    Q = X[:, 0, :]
    K = X.reshape(n * V, D)
    Is, Js, Ss = [], [], []
    total = 0
    for a in range(0, n, chunk):
        sims = Q[a:a + chunk] @ K.T
        sims = sims.reshape(sims.shape[0], n, V).max(axis=2)
        rows = np.arange(sims.shape[0])
        sims[rows, rows + a] = -1.0  # ignore self
        r, c = np.nonzero(sims >= tmin)
        if V == 1:  # symmetric, keep each pair once
            keep = c > (r + a)
            r, c = r[keep], c[keep]
        total += len(r)
        if total > MAX_EDGES:
            sys.exit("Too many similar pairs: the threshold is too low for this data. Use a higher value.")
        Is.append((r + a).astype(np.int32))
        Js.append(c.astype(np.int32))
        Ss.append(sims[r + 0, c].astype(np.float32))
    if not Is:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(Is), np.concatenate(Js), np.concatenate(Ss)


def build_edges(emb, cls, tmin):
    per_class = {}
    for c in sorted(set(cls.tolist())):
        idx = np.where(cls == c)[0]
        I, J, S = class_edges(emb[idx], tmin)
        per_class[c] = (idx, I, J, S)
        print(f"  {c}: {len(idx)} images, {len(I)} similar pairs >= {tmin}", flush=True)
    return per_class


def components(n, I, J, S, t):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = S >= t
    g = coo_matrix((np.ones(int(m.sum()), dtype=np.int8), (I[m], J[m])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def sweep(per_class, split, thresholds):
    n_total = len(split)
    n_val = int((split == "val").sum())
    n_test = int((split == "test").sum())
    rows = []
    for t in thresholds:
        n_groups, in_groups, largest = 0, 0, 0
        leak = {"val": set(), "test": set()}
        for c, (idx, I, J, S) in per_class.items():
            labels = components(len(idx), I, J, S, t)
            counts = np.bincount(labels)
            n_groups += len(counts)
            in_groups += int(counts[counts > 1].sum())
            largest = max(largest, int(counts.max()))
            m = S >= t
            a, b = split[idx[I[m]]], split[idx[J[m]]]
            for target in ("val", "test"):
                hit = np.concatenate([I[m][(a == target) & (b == "train")],
                                      J[m][(b == target) & (a == "train")]])
                leak[target].update(idx[np.unique(hit)].tolist())
        rows.append({
            "threshold": t,
            "groups": n_groups,
            "pct_images_in_dup_groups": round(100 * in_groups / n_total, 2),
            "largest_group": largest,
            "val_leak_pct": round(100 * len(leak["val"]) / max(n_val, 1), 2),
            "test_leak_pct": round(100 * len(leak["test"]) / max(n_test, 1), 2),
        })
    return rows


def print_sweep(rows):
    print(f"{'threshold':>9} {'groups':>8} {'%imgs in dup groups':>20} {'largest grp':>12} "
          f"{'val leak %':>11} {'test leak %':>12}")
    for r in rows:
        print(f"{r['threshold']:>9.2f} {r['groups']:>8} {r['pct_images_in_dup_groups']:>20} "
              f"{r['largest_group']:>12} {r['val_leak_pct']:>11} {r['test_leak_pct']:>12}")
    print("\nleak % = share of val/test images that have a near-duplicate in train.")


# ----------------------------------------------------------------------------- 2) analyze
def cmd_analyze(args):
    root, rel, emb, split, cls = load_state(args.root)
    ths = sorted(args.thresholds)
    print(f"{len(rel)} images, {emb.shape[1]} view(s). Finding similar pairs...")
    per_class = build_edges(emb, cls, ths[0])
    rows = sweep(per_class, split, ths)
    print()
    print_sweep(rows)
    with open("analysis_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote analysis_sweep.csv")
    print("\nHow to read it: pick the threshold where near-duplicates start to look real in 'preview'.")
    print("If test leak % is only ~1-3% at that threshold, the current split is nearly fine.")
    print("If it is ~10% or more, re-split. Watch 'largest grp': a huge group means the threshold")
    print("is too low (different leaves are being chained together).")


# ----------------------------------------------------------------------------- 3) preview
def draw_sheet(pairs, root, rel, out_path, title):
    from PIL import Image, ImageDraw
    cols = 2
    rows = (len(pairs) + cols - 1) // cols
    cw, ch, cap = 448, 224, 18
    sheet = Image.new("RGB", (cols * cw, cap + rows * (ch + cap)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), title, fill="black")
    for k, (i, j, s, c) in enumerate(pairs):
        r, col = divmod(k, cols)
        x, y = col * cw, cap + r * (ch + cap)
        for off, g in enumerate((i, j)):
            with Image.open(os.path.join(root, rel[g])) as im:
                sheet.paste(im.convert("RGB").resize((224, 224)), (x + off * 224, y + cap))
        d.text((x + 4, y + 2), f"sim={s:.3f}  {c.replace('Tomato___', '')[:30]}", fill="black")
    sheet.save(out_path)


def cmd_preview(args):
    root, rel, emb, split, cls = load_state(args.root)
    t = args.threshold
    print(f"Finding pairs from {t - 0.03:.2f} to {t + 0.02:.2f}...")
    per_class = build_edges(emb, cls, round(t - 0.03, 4))
    above, below = {}, {}
    for c, (idx, I, J, S) in per_class.items():
        for i, j, s in zip(I, J, S):
            key = (int(min(idx[i], idx[j])), int(max(idx[i], idx[j])))
            if t <= s < t + 0.02:
                above[key] = (s, c)
            elif t - 0.03 <= s < t:
                below[key] = (s, c)
    rng = random.Random(args.seed)

    def sample(d):
        keys = list(d.keys())
        rng.shuffle(keys)
        return [(i, j, d[(i, j)][0], d[(i, j)][1]) for i, j in keys[:args.pairs]]

    pa, pb = sample(above), sample(below)
    if pa:
        draw_sheet(pa, root, rel, "preview_above.png",
                   f"Pairs just ABOVE threshold {t} (these would be grouped): are they the same leaf?")
    if pb:
        draw_sheet(pb, root, rel, "preview_below.png",
                   f"Pairs just BELOW threshold {t} (these stay separate): are any the same leaf?")
    print(f"preview_above.png: {len(pa)} pairs | preview_below.png: {len(pb)} pairs")
    print("Good threshold: nearly all 'above' pairs are the same leaf, and few 'below' pairs are.")
    print("Same leaf in 'below' pairs -> lower the threshold. Different leaves in 'above' -> raise it.")


# ----------------------------------------------------------------------------- 4) resplit
def assign_groups(labels, seed):
    """Greedy: biggest groups first, each goes to the split furthest below its target."""
    n = len(labels)
    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[int(l)].append(i)
    order = list(groups.values())
    random.Random(seed).shuffle(order)          # random tie-breaking
    order.sort(key=len, reverse=True)           # stable sort keeps that randomness
    target = {s: RATIOS[s] * n for s in SPLITS}
    count = {s: 0 for s in SPLITS}
    assign = [None] * n
    for g in order:
        s = max(SPLITS, key=lambda k: target[k] - count[k])
        for i in g:
            assign[i] = s
        count[s] += len(g)
    return assign


def cmd_resplit(args):
    root, rel, emb, split, cls = load_state(args.root)
    t = args.threshold
    ths = sorted(set(DEFAULT_SWEEP + [t]))
    print(f"{len(rel)} images. Finding similar pairs (>= {ths[0]})...")
    per_class = build_edges(emb, cls, min(ths[0], t))

    new_split = np.empty(len(rel), dtype="<U5")
    group_id = [""] * len(rel)
    for ci, (c, (idx, I, J, S)) in enumerate(per_class.items()):
        labels = components(len(idx), I, J, S, t)
        assigned = assign_groups(labels, args.seed)
        for local, g in enumerate(idx):
            new_split[g] = assigned[local]
            group_id[g] = f"{ci:02d}_{int(labels[local]):05d}"

    print(f"\nClass counts, old split -> new split (threshold {t}, seed {args.seed}):")
    print(f"{'class':<48} {'train':>12} {'val':>12} {'test':>12}")
    for c, (idx, *_rest) in per_class.items():
        cells = []
        for s in SPLITS:
            old, new = int((split[idx] == s).sum()), int((new_split[idx] == s).sum())
            cells.append(f"{old}->{new}")
        print(f"{c:<48} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")
    tot = [f"{int((split == s).sum())}->{int((new_split == s).sum())}" for s in SPLITS]
    print(f"{'TOTAL':<48} {tot[0]:>12} {tot[1]:>12} {tot[2]:>12}")

    print("\nRemaining leakage with the NEW split (it should be ~0 at the chosen threshold):")
    print_sweep(sweep(per_class, new_split, ths))

    if args.dry_run:
        print("\nDry run: nothing was copied.")
        return

    out = args.out
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"Output folder {out} already exists and is not empty. Choose another --out.")
    os.makedirs(os.path.join(out, "metadata"), exist_ok=True)

    print(f"\nCopying {len(rel)} files to {out} ...")
    manifest, counts = [], defaultdict(int)
    for k, r in enumerate(rel):
        _, c, name = r.split("/", 2)
        s = str(new_split[k])
        dst_dir = os.path.join(out, s, c)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(os.path.join(root, r), os.path.join(dst_dir, name))
        manifest.append([f"{s}/{c}/{name}", s, c, name, group_id[k], split[k]])
        counts[(c, s)] += 1
        if (k + 1) % 3000 == 0:
            print(f"  {k + 1}/{len(rel)}", flush=True)

    with open(os.path.join(out, "metadata", "split_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "class", "filename", "group_id", "previous_split"])
        w.writerows(manifest)
    with open(os.path.join(out, "metadata", "class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "split", "count"])
        for (c, s), n in sorted(counts.items()):
            w.writerow([c, s, n])
    with open(os.path.join(out, "metadata", "split_params.txt"), "w") as f:
        f.write("Group-aware re-split\n"
                f"embedding model: torchvision ResNet50 (IMAGENET1K_V2), avgpool features, views={emb.shape[1]}\n"
                f"cosine similarity threshold: {t}\n"
                f"seed: {args.seed}\n"
                f"target ratios: {RATIOS}\n"
                "near-duplicates searched within each class; groups = connected components\n")
    print(f"Done. New dataset in {out}")


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="compute and cache embeddings")
    e.add_argument("--root", required=True, help="processed dataset folder (train/val/test inside)")
    e.add_argument("--views", type=int, choices=[1, 4], default=1,
                   help="1 = fast; 4 = also match flipped/rotated copies (about 4x slower)")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--force", action="store_true")
    e.set_defaults(fn=cmd_embed)

    a = sub.add_parser("analyze", help="threshold sweep and leakage report")
    a.add_argument("--root", default=None)
    a.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_SWEEP)
    a.set_defaults(fn=cmd_analyze)

    pr = sub.add_parser("preview", help="contact sheets of pairs around a threshold")
    pr.add_argument("--root", default=None)
    pr.add_argument("--threshold", type=float, default=0.95)
    pr.add_argument("--pairs", type=int, default=16)
    pr.add_argument("--seed", type=int, default=42)
    pr.set_defaults(fn=cmd_preview)

    r = sub.add_parser("resplit", help="group-aware re-split (copies files to a new folder)")
    r.add_argument("--root", default=None)
    r.add_argument("--threshold", type=float, required=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", default="processed_v2")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_resplit)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
