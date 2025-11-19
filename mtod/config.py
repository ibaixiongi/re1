import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TrainCfg:
    batch_size: int = 2
    lr: float = 0.005
    weight_decay: float = 0.0005
    epochs: int = 12
    lr_steps: List[int] = None
    warmup_iters: int = 500
    num_workers: int = 4
    seed: int = 42


@dataclass
class DistillCfg:
    # ROI classification distillation
    use_kl: bool = True
    kl_weight: float = 0.5  # deprecated alias of roi_cls_weight
    kl_temperature: float = 2.0
    # Pseudo label generation (teacher)
    pseudo_conf_thresh: float = 0.5
    max_pseudo_per_image: int = 200
    # New: multi-task distillation weights
    rpn_cls_weight: float = 0.5
    rpn_reg_weight: float = 0.5
    roi_cls_weight: float = 0.5
    roi_reg_weight: float = 0.5
    roi_match_iou: float = 0.5
    rpn_reg_weight_obj_thresh: float = 0.3


@dataclass
class Cfg:
    # Non-default fields must come first
    base_classes: List[str]
    new_classes: List[str]
    # Defaults follow
    data_root: str = ""
    train: TrainCfg = field(default_factory=TrainCfg)
    distill: DistillCfg = field(default_factory=DistillCfg)
    coco: Optional[Dict[str, str]] = None


def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    train = TrainCfg(**y.get("train", {}))
    distill = DistillCfg(**y.get("distill", {}))
    cfg = Cfg(
        data_root=y.get("data_root", ""),
        base_classes=y["base_classes"],
        new_classes=y["new_classes"],
        train=train,
        distill=distill,
        coco=y.get("coco", None),
    )
    return cfg
