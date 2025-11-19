import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from .config import load_config
from .datasets.voc import build_voc_val
from .models.build import build_faster_rcnn, load_checkpoint
from .utils import collate_fn


def voc_ap(rec, prec):
    # 11-point metric (VOC2007)
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.
    return ap


def compute_map(gts, preds, num_classes: int, iou_thresh: float = 0.5):
    # gts, preds: lists per image of dicts
    from torchvision.ops import box_iou
    aps = []
    for c in range(1, num_classes):
        # gather preds and gts of class c
        class_preds = []  # (image_id, score, box)
        class_gts = {}
        for i, (gt, pr) in enumerate(zip(gts, preds)):
            gt_mask = (gt["labels"] == c)
            gt_boxes = gt["boxes"][gt_mask]
            class_gts[i] = {
                "boxes": gt_boxes.cpu().numpy(),
                "detected": np.zeros((gt_boxes.shape[0],), dtype=bool)
            }
            pr_mask = (pr["labels"] == c)
            pr_boxes = pr["boxes"][pr_mask]
            pr_scores = pr["scores"][pr_mask]
            for b, s in zip(pr_boxes, pr_scores):
                class_preds.append((i, float(s.cpu()), b.cpu().numpy()))

        if len(class_preds) == 0:
            aps.append(0.0)
            continue

        # sort by score desc
        class_preds.sort(key=lambda x: -x[1])
        tp = np.zeros((len(class_preds),))
        fp = np.zeros((len(class_preds),))
        npos = sum(len(v["boxes"]) for v in class_gts.values())
        for idx, (img_id, score, box) in enumerate(class_preds):
            gt_info = class_gts[img_id]
            gt_boxes = gt_info["boxes"]
            if gt_boxes.shape[0] == 0:
                fp[idx] = 1
                continue
            ious = []
            for gbox in gt_boxes:
                # compute IoU
                xx1 = max(box[0], gbox[0])
                yy1 = max(box[1], gbox[1])
                xx2 = min(box[2], gbox[2])
                yy2 = min(box[3], gbox[3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                area_p = (box[2] - box[0]) * (box[3] - box[1])
                area_g = (gbox[2] - gbox[0]) * (gbox[3] - gbox[1])
                union = area_p + area_g - inter
                iou = inter / union if union > 0 else 0.0
                ious.append(iou)
            ious = np.array(ious)
            best = ious.argmax()
            if ious[best] >= iou_thresh and not gt_info["detected"][best]:
                tp[idx] = 1
                gt_info["detected"][best] = True
            else:
                fp[idx] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / max(npos, 1)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        aps.append(ap)
    mAP = float(np.mean(aps)) if len(aps) else 0.0
    return mAP, aps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = cfg.base_classes + cfg.new_classes
    val_ds = build_voc_val(cfg.data_root, classes)
    loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    num_classes = 1 + len(classes)
    model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
    load_checkpoint(model, args.ckpt, strict=False)
    model.to(device)
    model.eval()

    preds = []
    gts = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for t, o in zip(targets, outputs):
                gts.append({"boxes": t["boxes"], "labels": t["labels"]})
                preds.append({"boxes": o["boxes"], "labels": o["labels"], "scores": o["scores"]})

    mAP, aps = compute_map(gts, preds, num_classes=num_classes)
    print(f"VOC07 mAP@0.5: {mAP:.4f}")
    for i, ap in enumerate(aps, start=1):
        print(f"  class {i}: AP={ap:.4f}")


if __name__ == "__main__":
    main()

