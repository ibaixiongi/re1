from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F


def kl_divergence_with_temperature(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                                   T: float = 2.0) -> torch.Tensor:
    # student_logits, teacher_logits: [N, C]
    # returns mean KL(softmax(t/T) || softmax(s/T)) * (T^2)
    t_prob = F.softmax(teacher_logits / T, dim=1)
    s_log_prob = F.log_softmax(student_logits / T, dim=1)
    kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T)
    return kl


def match_by_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, iou_thresh: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    # boxes: [N,4], [M,4] in xyxy
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    from torchvision.ops import box_iou
    ious = box_iou(boxes_a, boxes_b)  # [N,M]
    # greedy match: for each b in B, find best a
    b2a = ious.argmax(dim=0)
    b_ious = ious.max(dim=0).values
    keep = b_ious >= iou_thresh
    idx_b = torch.nonzero(keep, as_tuple=False).squeeze(1)
    idx_a = b2a[idx_b]
    return idx_a, idx_b

