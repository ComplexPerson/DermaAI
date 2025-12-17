import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Ham10000Dataset, get_transforms
from utils import build_model, find_latest_checkpoint, load_model_state


def evaluate_checkpoint(
    ckpt_path,
    data_dir="data",
    metadata="data/HAM10000_metadata.csv",
    images_subdir="images",
    image_size=224,
    batch_size=32,
    backbone="resnet34",
    num_workers=0,
    device=None,
):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    images_root = os.path.join(data_dir, images_subdir)
    ds = Ham10000Dataset(
        images_root, metadata, transform=get_transforms(image_size, is_train=False)
    )
    # load model
    num_classes = len(ds.label_names)
    model = build_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model = load_model_state(model, ckpt_path, map_location="cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)

    acc = correct / total if total else 0.0
    avg_loss = total_loss / total if total else 0.0
    return {"accuracy": acc, "loss": avg_loss, "samples": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--images-subdir", type=str, default="images")
    parser.add_argument("--metadata", default="data/HAM10000_metadata.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    ckpt = args.checkpoint or find_latest_checkpoint()
    if ckpt is None:
        print("Error: no checkpoint found. Use --checkpoint or place a model in checkpoints/")
        return

    stats = evaluate_checkpoint(
        ckpt,
        data_dir=args.data_dir,
        metadata=args.metadata,
        images_subdir=args.images_subdir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        backbone=args.backbone,
        num_workers=args.num_workers,
    )
    print(stats)


if __name__ == "__main__":
    main()
