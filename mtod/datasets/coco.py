from typing import List, Dict, Any
import os
import torch
import torchvision
from torchvision.datasets import CocoDetection


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, ann_file: str, class_names: List[str], transforms=None):
        self.ds = CocoDetection(img_dir, ann_file)
        self.transforms = transforms
        self.class_to_idx = {c: i + 1 for i, c in enumerate(class_names)}  # 0 is background
        # Build category_id -> name mapping
        cats = self.ds.coco.loadCats(self.ds.coco.getCatIds())
        self.id_to_name = {c['id']: c['name'] for c in cats}
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
        self.class_names = class_names

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, anns = self.ds[idx]
        coco_img_id = self.ds.ids[idx]
        boxes = []
        labels = []
        for a in anns:
            if int(a.get('iscrowd', 0)) == 1:
                continue
            x, y, w, h = a['bbox']
            if w <= 0 or h <= 0:
                continue
            name = self.id_to_name.get(a['category_id'], None)
            if name is None or name not in self.class_to_idx:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.class_to_idx[name])
        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([coco_img_id])}
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


def build_coco(img_dir: str, ann_file: str, classes: List[str]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return COCODataset(img_dir=img_dir, ann_file=ann_file, class_names=classes, transforms=transform)
