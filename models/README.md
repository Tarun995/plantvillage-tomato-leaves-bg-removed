# Model weights and model code (not stored in this repo)

`src/preprocess_dataset.py` and `examples/detect_leaf.py` need the files below. Everything here is git-ignored.

| File | What it is | Where it goes |
|---|---|---|
| `u2net.pth` | U-2-Net weights | `models/u2net.pth` |
| `best.pt` | custom YOLOv8n leaf detector | `models/best.pt` |
| `u2net.py` | U-2-Net model definition | `src/u2net/u2net.py` |

## U-2-Net

Get the weights and the model code from the official repository: https://github.com/xuebinqin/U-2-Net (the weights download link is in its README).

1. Save the weights as `models/u2net.pth`.
2. Save the file `model/u2net.py` from that repository as `src/u2net/u2net.py`. Create the `u2net` folder if needed. `src/u2net_remove_bg.py` imports it with `from u2net.u2net import U2NET`.

## Leaf detector (`best.pt`)

A YOLOv8n detector (single class: "leaf"), fine-tuned from COCO-pretrained weights on about 120 leaf images (256x256 px) annotated with bounding boxes in Roboflow (30 epochs, 640x640). The highest-confidence box is used to crop each leaf.

TODO: add a download link for `best.pt` (for example a GitHub Release asset), or train your own: any YOLOv8 detector that finds a single leaf per PlantVillage image will work.

You can also point the scripts at other weight files:

```bash
python src/preprocess_dataset.py --input ... --output ... --yolo-weights path/to/best.pt --u2net-weights path/to/u2net.pth
```
