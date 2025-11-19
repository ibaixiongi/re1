import argparse
from typing import List
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .datasets.voc import build_voc_train
from .datasets.coco import build_coco
from .models.build import build_faster_rcnn, load_checkpoint, save_checkpoint
from .pseudo import teacher_pseudo_labels
from .models.head_hooks import kl_divergence_with_temperature, match_by_iou
from .models.distill_capture import DistillCapture
from .utils import collate_fn, set_seed, ensure_dir


def merge_targets_with_pseudo(targets, pseudo_targets):
    # Append teacher pseudo labels for old classes to targets
    merged = []
    for t, p in zip(targets, pseudo_targets):
        if p["boxes"].numel() > 0:
            boxes = torch.cat([t["boxes"], p["boxes"]], dim=0)
            labels = torch.cat([t["labels"], p["labels"]], dim=0)
        else:
            boxes = t["boxes"]
            labels = t["labels"]
        merged.append({"boxes": boxes, "labels": labels, "image_id": t["image_id"]})
    return merged


def _rpn_distill_loss(teacher_cap: DistillCapture, student_cap: DistillCapture, distill_cfg, device):
    loss = torch.tensor(0.0, device=device)
    # RPN objectness (binary) via BCE with teacher probabilities
    t_obj = teacher_cap.flatten_rpn_objectness()
    s_obj = student_cap.flatten_rpn_objectness()
    if t_obj is not None and s_obj is not None and t_obj.shape == s_obj.shape and distill_cfg.rpn_cls_weight > 0:
        t_prob = torch.sigmoid(t_obj)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(s_obj, t_prob, reduction='mean')
        loss = loss + distill_cfg.rpn_cls_weight * bce
    # RPN bbox regression: SmoothL1 weighted by teacher objectness
    t_reg = teacher_cap.flatten_rpn_bbox()
    s_reg = student_cap.flatten_rpn_bbox()
    if t_reg is not None and s_reg is not None and t_reg.shape == s_reg.shape and distill_cfg.rpn_reg_weight > 0:
        t_obj = teacher_cap.flatten_rpn_objectness()
        if t_obj is not None and t_obj.numel() == (t_reg.shape[0] * t_reg.shape[1]):
            w = (torch.sigmoid(t_obj).clamp(min=0.0) >= distill_cfg.rpn_reg_weight_obj_thresh).float().view(t_reg.shape[0], t_reg.shape[1], 1)
        else:
            w = torch.ones((t_reg.shape[0], t_reg.shape[1], 1), device=device)
        l1 = torch.nn.functional.smooth_l1_loss(s_reg, t_reg, reduction='none')
        l1 = (l1 * w).mean()
        loss = loss + distill_cfg.rpn_reg_weight * l1
    return loss


def _roi_distill_loss(teacher_cap: DistillCapture, student_cap: DistillCapture, distill_cfg, old_class_ids: List[int], device):
    loss = torch.tensor(0.0, device=device)
    # Split ROI outputs by image and align proposals
    slogits_list, sbbox_list = student_cap.split_roi_outputs_by_image()
    tlogits_list, tbbox_list = teacher_cap.split_roi_outputs_by_image()
    sproposals = student_cap.roi_proposals
    tproposals = teacher_cap.roi_proposals
    if None in (slogits_list, sbbox_list, tlogits_list, tbbox_list) or sproposals is None or tproposals is None:
        return loss
    T = distill_cfg.kl_temperature
    for i in range(len(sproposals)):
        sprops = sproposals[i]
        tprops = tproposals[i]
        if sprops.numel() == 0 or tprops.numel() == 0:
            continue
        idx_s, idx_t = match_by_iou(sprops, tprops, iou_thresh=distill_cfg.roi_match_iou)
        if idx_s.numel() == 0:
            continue
        s_logits = slogits_list[i][idx_s]
        t_logits = tlogits_list[i][idx_t]
        # ROI classification KL on proposals whose teacher label is from old classes
        if distill_cfg.roi_cls_weight > 0:
            t_labels = t_logits.argmax(dim=1)
            if len(old_class_ids) > 0:
                mask_old = torch.zeros_like(t_labels, dtype=torch.bool)
                for cid in old_class_ids:
                    mask_old |= (t_labels == cid)
                if mask_old.any():
                    s_logits_masked = s_logits[mask_old]
                    t_logits_masked = t_logits[mask_old]
                    if s_logits_masked.numel() > 0:
                        kl = kl_divergence_with_temperature(s_logits_masked, t_logits_masked, T=T)
                        loss = loss + distill_cfg.roi_cls_weight * kl
        # ROI bbox regression L1 on matched proposals for teacher's class
        if distill_cfg.roi_reg_weight > 0 and sbbox_list[i] is not None and tbbox_list[i] is not None:
            t_labels = t_logits.argmax(dim=1)
            if len(old_class_ids) > 0:
                mask_old = torch.zeros_like(t_labels, dtype=torch.bool)
                for cid in old_class_ids:
                    mask_old |= (t_labels == cid)
                if mask_old.any():
                    t_labels = t_labels[mask_old]
                    sb = sbbox_list[i][idx_s][mask_old]
                    tb = tbbox_list[i][idx_t][mask_old]
                else:
                    continue
            else:
                sb = sbbox_list[i][idx_s]
                tb = tbbox_list[i][idx_t]
            if sb.numel() == 0 or tb.numel() == 0:
                continue
            num_classes = s_logits.shape[1]
            # pick the 4-deltas corresponding to teacher class
            rows = torch.arange(t_labels.numel(), device=device)
            cols = t_labels.view(-1)
            sb_sel = sb.view(sb.shape[0], num_classes, 4)[rows, cols]
            tb_sel = tb.view(tb.shape[0], num_classes, 4)[rows, cols]
            l1 = torch.nn.functional.smooth_l1_loss(sb_sel, tb_sel, reduction='mean')
            loss = loss + distill_cfg.roi_reg_weight * l1
    return loss


def train_one_epoch(model, teacher, loader, optimizer, device, old_class_ids: List[int], distill_cfg):
    model.train()
    total_loss = 0.0
    # Attach capture hooks once
    student_cap = DistillCapture(model).attach()
    teacher_cap = DistillCapture(teacher).attach()
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            pseudo = teacher_pseudo_labels(
                teacher, images, old_class_ids=old_class_ids,
                score_thresh=distill_cfg.pseudo_conf_thresh,
                max_per_image=distill_cfg.max_pseudo_per_image,
            )
            pseudo = [{k: v.to(device) for k, v in p.items()} for p in pseudo]

        merged_targets = merge_targets_with_pseudo(targets, pseudo)
        # Standard detection losses with augmented targets
        loss_dict = model(images, merged_targets)
        loss = sum(loss for loss in loss_dict.values())

        # Multi-task distillation: RPN cls/reg + ROI cls/reg
        # Run teacher forward pass (eval) to populate captures
        teacher_cap.reset()
        model.eval(); teacher.eval()
        with torch.no_grad():
            _ = teacher(images)
        model.train()
        # Student capture is already populated by the loss forward (model(images, merged_targets))
        rpn_loss = _rpn_distill_loss(teacher_cap, student_cap, distill_cfg, device)
        roi_loss = _roi_distill_loss(teacher_cap, student_cap, distill_cfg, old_class_ids, device)
        loss = loss + rpn_loss + roi_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    ensure_dir(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    if cfg.coco:
        classes_all = cfg.base_classes + cfg.new_classes
        train_ds = build_coco(cfg.coco["train_images"], cfg.coco["train_ann"], classes_all)
    else:
        train_ds = build_voc_train(cfg.data_root, cfg.base_classes, cfg.new_classes, step="new")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    # Teacher (old classes) and student (old+new)
    num_old = 1 + len(cfg.base_classes)
    num_all = 1 + len(cfg.base_classes) + len(cfg.new_classes)

    teacher = build_faster_rcnn(num_classes=num_old, pretrained=False)
    load_checkpoint(teacher, args.teacher_ckpt, strict=True)
    teacher.to(device)
    teacher.eval()

    model = build_faster_rcnn(num_classes=num_all, pretrained=True)
    model.to(device)

    # Map old class ids in teacher to same ids in student
    old_class_ids = list(range(1, num_old))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)

    epochs = args.epochs or cfg.train.epochs
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, teacher, train_loader, optimizer, device, old_class_ids, cfg.distill)
        print(f"Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")

    ckpt_path = os.path.join(args.output, "model_final.pth")
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
