# Multi-Task Incremental Learning for Object Detection (Reproduction)

This repo provides a lightweight, from-scratch reproduction of the core ideas in the paper:

- Xialei Liu, Hao Yang, Avinash Ravichandran, Rahul Bhotika, Stefano Soatto. "Multi-Task Incremental Learning for Object Detection" (arXiv:2002.05347)

It uses PyTorch + torchvision Faster R-CNN (ResNet-50 FPN) and implements a practical incremental training pipeline with teacher-student pseudo-labeling for old classes and optional knowledge-distillation style losses. The goal is to make it easy to run VOC-style incremental experiments on a single machine.

Status: End-to-end pipeline with base training and incremental step for VOC, including multi-task distillation (RPN cls/reg + ROI cls/reg). COCO support and example split included.

## What's implemented
- Base detector training on a set of base classes
- Incremental step training that preserves old classes via teacher pseudo-labels on new-task data
- Multi-task distillation:
  - RPN objectness (BCE on logits vs teacher probs)
  - RPN bbox regression (SmoothL1, weighted by teacher objectness)
  - ROI classification (KL with temperature on matched proposals)
  - ROI bbox regression (SmoothL1 on teacher-class deltas for matched proposals)
- VOC dataset support with configurable class splits (e.g., 10/10)
- COCO dataset support with example 60/20 split config
  - Evaluator: `python -m mtod.eval_coco --config <cfg> --ckpt <pth> [--out results.json]`

## Requirements
- Python 3.9+
- PyTorch >= 2.1 and torchvision >= 0.16 (CUDA recommended)
- numpy, tqdm, pyyaml, opencv-python, matplotlib, scipy, pycocotools

Install deps:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset: PASCAL VOC
By default we use VOC 2007+2012 trainval for training and VOC 2007 test for evaluation.

Folder layout expected by torchvision:

```
VOCdevkit/
  VOC2007/
    Annotations/*.xml
    JPEGImages/*.jpg
    ImageSets/Main/*.txt
  VOC2012/
    Annotations/*.xml
    JPEGImages/*.jpg
    ImageSets/Main/*.txt
```

You can download from the official site or mirrors. Set `data_root` in `configs/voc_10_10.yaml` accordingly.

## Quick start (VOC 10/10 split)

1) Train base detector on first 10 classes:
```
python -m mtod.train_base --config configs/voc_10_10.yaml --output runs/voc10_base
```
This saves a checkpoint at `runs/voc10_base/model_final.pth`.

2) Incremental step: add last 10 classes, preserving the first 10 via teacher pseudo-labels:
```
python -m mtod.train_incremental \
  --config configs/voc_10_10.yaml \
  --teacher_ckpt runs/voc10_base/model_final.pth \
  --output runs/voc10_inc
```

3) Evaluate on VOC 2007 test:
```
python -m mtod.eval_voc --config configs/voc_10_10.yaml --ckpt runs/voc10_inc/model_final.pth
```

## Config
See `configs/voc_10_10.yaml` (VOC), `configs/coco_60_20.yaml` (COCO 60/20), and `configs/coco_paper_voc_overlap_20.yaml` (paper scenario: VOC->COCO overlap 20). Key fields:
- `data_root`: path to VOCdevkit folder
- `base_classes`: list of class names for base step
- `new_classes`: list of class names for incremental step
- `train`: batch size, lr, epochs, weight decay
- `distill`: thresholds and weights for pseudo labels and distillation
  - `rpn_cls_weight`, `rpn_reg_weight`, `roi_cls_weight`, `roi_reg_weight`
  - `kl_temperature`, `roi_match_iou`, `rpn_reg_weight_obj_thresh`

## Notes on paper parity
- Implements multi-task distillation on RPN (cls/reg) and ROI (cls/reg) using forward hooks over torchvision Faster R-CNN.
- Exact parity may require hyperparameter tuning and potentially finer-grained masking as in the paper. The implementation is faithful in spirit and practical for single-GPU runs.

## Next steps
- Reproduce reported numbers after hyperparameter tuning
- Add per-anchor/objectness-adaptive masking as ablated in the paper
- Add mixed-memory or rehearsal buffers if needed by your setup

## Citation
If you use this scaffolding, please cite the original paper.

## Extras

- VOC auto-download (PowerShell): `scripts/get_voc.ps1`
  - Downloads VOC2007 trainval/test and VOC2012 trainval into `VOCdevkit/`
  - Example: `powershell -ExecutionPolicy Bypass -File scripts/get_voc.ps1 -Dest "D:\\datasets"`
  - Then point `data_root` to `D:\\datasets\\VOCdevkit`

### COCO

- Example config: `configs/coco_60_20.yaml`
- If `pycocotools` install fails on Windows, try: `pip install pycocotools-windows`
