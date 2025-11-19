from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torchvision


def build_faster_rcnn(num_classes: int, pretrained: bool = True) -> torchvision.models.detection.FasterRCNN:
    # Start from torchvision's standard baseline
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=(
        torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    ))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, path: str, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=strict)


def save_checkpoint(model: nn.Module, path: str):
    to_save = model.state_dict()
    torch.save(to_save, path)

