import argparse
import json
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .datasets.coco import build_coco, COCODataset
from .models.build import build_faster_rcnn, load_checkpoint
from .utils import collate_fn


def coco_results_from_outputs(dataset: COCODataset, outputs: List[Dict], targets: List[Dict], classes: List[str]) -> List[Dict]:
    results = []
    # Build mapping from our label ids to COCO category ids
    name_to_coco = dataset.name_to_id
    for i, out in enumerate(outputs):
        boxes = out["boxes"].cpu()
        labels = out["labels"].cpu()
        scores = out["scores"].cpu()
        tid = targets[i]["image_id"]
        image_id = int(tid.item()) if hasattr(tid, 'item') else int(tid)
        for b, l, s in zip(boxes, labels, scores):
            cls_idx = int(l.item())
            if cls_idx <= 0 or cls_idx > len(classes):
                continue
            name = classes[cls_idx - 1]
            cat_id = name_to_coco.get(name)
            if cat_id is None:
                # Skip if class name not present in COCO annotations
                continue
            x1, y1, x2, y2 = [float(x) for x in b.tolist()]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            results.append({
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": [x1, y1, w, h],
                "score": float(s.item()),
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None, help="Optional path to write COCO results JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.coco is not None, "COCO paths are required in config (coco: {train_images, train_ann, val_images, val_ann})."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg.base_classes + cfg.new_classes

    ds = build_coco(cfg.coco["val_images"], cfg.coco["val_ann"], classes)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    num_classes = 1 + len(classes)
    model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
    load_checkpoint(model, args.ckpt, strict=False)
    model.to(device)
    model.eval()

    all_results = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval-coco"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            batch_results = coco_results_from_outputs(ds, outputs, targets, classes)
            all_results.extend(batch_results)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f)
        print(f"Saved results to {args.out}")

    # Use COCOeval
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(cfg.coco["val_ann"])
    coco_dt = coco_gt.loadRes(all_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    main()
