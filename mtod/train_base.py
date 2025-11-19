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
from .models.build import build_faster_rcnn, save_checkpoint
from .utils import collate_fn, set_seed, ensure_dir


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    ensure_dir(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and model
    if cfg.coco:
        train_ds = build_coco(cfg.coco["train_images"], cfg.coco["train_ann"], cfg.base_classes)
        num_classes = 1 + len(cfg.base_classes)
    else:
        train_ds = build_voc_train(cfg.data_root, cfg.base_classes, cfg.new_classes, step="base")
        num_classes = 1 + len(cfg.base_classes)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    model = build_faster_rcnn(num_classes=num_classes, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)

    epochs = args.epochs or cfg.train.epochs
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")

    ckpt_path = os.path.join(args.output, "model_final.pth")
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
