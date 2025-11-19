from typing import List, Tuple, Dict, Any
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.ops import box_convert
import torch
import os


VOC_ALL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def _voc_parse_target(target_xml: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    objs = target_xml["annotation"]["object"]
    if isinstance(objs, dict):
        objs = [objs]
    boxes = []
    labels = []
    for obj in objs:
        bnd = obj["bndbox"]
        xmin = float(bnd["xmin"]) - 1
        ymin = float(bnd["ymin"]) - 1
        xmax = float(bnd["xmax"]) - 1
        ymax = float(bnd["ymax"]) - 1
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(obj["name"])
    if len(boxes) == 0:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_s = []
    else:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_s = labels
    return boxes_t, labels_s


def _resolve_torchvision_voc_root(root: str) -> str:
    # torchvision expects root such that root/"VOCdevkit"/VOC{year}/...
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, "VOCdevkit")):
        return root
    # If user passed the VOCdevkit directory itself, return its parent
    base = os.path.basename(os.path.normpath(root)).lower()
    if base == "vocdevkit":
        return os.path.dirname(root)
    return root


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, year: str, image_set: str,
                 class_names: List[str], transforms=None):
        tv_root = _resolve_torchvision_voc_root(root)
        self.ds = VOCDetection(root=tv_root, year=year, image_set=image_set, download=False)
        self.class_to_idx = {c: i + 1 for i, c in enumerate(class_names)}  # 0 is background
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, target_xml = self.ds[idx]
        boxes, labels_s = _voc_parse_target(target_xml)
        labels = []
        keep = []
        for i, name in enumerate(labels_s):
            if name in self.class_to_idx:
                labels.append(self.class_to_idx[name])
                keep.append(i)
        if len(keep) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = boxes[torch.tensor(keep, dtype=torch.long)]
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels_t, "image_id": torch.tensor([idx])}
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


def build_voc_train(root: str, base_classes: List[str], new_classes: List[str],
                    step: str = "base"):
        # For simplicity, use VOC2007+2012 trainval. torchvision handles each year separately.
        from torchvision.transforms import functional as F
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        if step == "base":
            classes = base_classes
        elif step == "new":
            classes = base_classes + new_classes
        else:
            raise ValueError("step must be 'base' or 'new'")

        ds07 = VOCDataset(root=root, year="2007", image_set="trainval", class_names=classes, transforms=transform)
        ds12 = VOCDataset(root=root, year="2012", image_set="trainval", class_names=classes, transforms=transform)
        ds = torch.utils.data.ConcatDataset([ds07, ds12])
        return ds


def build_voc_val(root: str, classes: List[str]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return VOCDataset(root=root, year="2007", image_set="test", class_names=classes, transforms=transform)
