from typing import Dict, Any, List
import torch
from torchvision.ops import nms


@torch.no_grad()
def teacher_pseudo_labels(teacher, images: List[torch.Tensor], old_class_ids: List[int],
                          score_thresh: float = 0.5, max_per_image: int = 200) -> List[Dict[str, torch.Tensor]]:
    teacher.eval()
    outputs = teacher(images)
    pseudo_targets = []
    for out in outputs:
        boxes = out["boxes"]
        scores = out["scores"]
        labels = out["labels"]
        # Keep only old classes and scores above threshold
        keep = (scores >= score_thresh)
        if len(old_class_ids) > 0:
            mask_old = torch.zeros_like(labels, dtype=torch.bool)
            for cid in old_class_ids:
                mask_old |= (labels == cid)
            keep = keep & mask_old
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        boxes = boxes[idx]
        scores = scores[idx]
        labels = labels[idx]
        # NMS to reduce duplicates
        if boxes.numel() > 0:
            keep_nms = nms(boxes, scores, 0.5)
            if max_per_image > 0:
                keep_nms = keep_nms[:max_per_image]
            boxes = boxes[keep_nms]
            labels = labels[keep_nms]
        pseudo_targets.append({
            "boxes": boxes.detach(),
            "labels": labels.detach(),
        })
    return pseudo_targets

