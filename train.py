import os
import argparse
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.backends.cudnn as cudnn
from dataset import Ham10000Dataset, get_transforms
from utils import build_model, cutmix_data, mixup_data, save_checkpoint
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}
    # convert to list ordered by class index
    ordered = [weights[i] for i in sorted(weights.keys())]
    return torch.tensor(ordered, dtype=torch.float)




def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
    scheduler=None,
):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # apply mixup or cutmix if requested
        if mixup_alpha > 0.0:
            mixed_x, y_a, y_b, lam = mixup_data(images, targets, mixup_alpha)
            images_input = mixed_x
        elif cutmix_alpha > 0.0:
            mixed_x, y_a, y_b, lam = cutmix_data(images, targets, cutmix_alpha)
            images_input = mixed_x
        else:
            images_input = images

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images_input)
                if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                        outputs, y_b
                    )
                else:
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images_input)
            if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                    outputs, y_b
                )
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        # compute accuracy: if mixup/cutmix was used compute soft accuracy
        if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
            correct += (
                lam * preds.eq(y_a).sum().item()
                + (1.0 - lam) * preds.eq(y_b).sum().item()
            )
        else:
            correct += preds.eq(targets).sum().item()
        total += images.size(0)
        pbar.set_postfix(
            loss=running_loss / total if total else 0.0,
            acc=(correct / total) if total else 0.0,
        )

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)
            pbar.set_postfix(
                loss=running_loss / total if total else 0.0,
                acc=(correct / total) if total else 0.0,
            )
    return running_loss / total, correct / total


def prepare_dataloaders(args):
    """Prepare and return the training and validation dataloaders."""
    # Prepare dataset
    images_root = os.path.join(args.data_dir, args.images_subdir)
    dataset_full = Ham10000Dataset(
        images_root,
        args.metadata,
        transform=get_transforms(args.image_size, is_train=True),
    )

    # split train/val stratified
    indices = list(range(len(dataset_full)))
    labels = [s[1] for s in dataset_full.samples]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=labels
    )

    train_dataset = Subset(dataset_full, train_idx)
    # for validation use evaluation transforms
    val_full = Ham10000Dataset(
        images_root,
        args.metadata,
        transform=get_transforms(args.image_size, is_train=False),
    )
    val_dataset = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # for class weights
    train_labels_list = [labels[i] for i in train_idx]
    num_classes = len(dataset_full.label_names)

    return train_loader, val_loader, train_labels_list, num_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="root data dir containing images/ and metadata CSV",
    )
    parser.add_argument("--metadata", type=str, default="data/HAM10000_metadata.csv")
    parser.add_argument("--images-subdir", type=str, default="images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        help="backbone: 'resnet34','resnet50' or 'timm:<name>'",
    )
    parser.add_argument(
        "--mixup", type=float, default=0.0, help="mixup alpha (0 to disable)"
    )
    parser.add_argument(
        "--cutmix", type=float, default=0.0, help="cutmix alpha (0 to disable)"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "onecycle", "cosine"],
        help="LR scheduler",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cudnn.benchmark = True
    print(f"Using device: {device}")

    train_loader, val_loader, train_labels, num_classes = prepare_dataloaders(args)

    # build model with requested backbone (centralized in utils)
    model = build_model(
        num_classes=num_classes, backbone=args.backbone, pretrained=True
    )
    model = model.to(device)

    # class weights (from training set)
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Scheduler
    scheduler = None
    if args.scheduler == "onecycle":
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            scheduler=scheduler if args.scheduler == "onecycle" else None,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step epoch-based schedulers
        if scheduler is not None and args.scheduler != "onecycle":
            scheduler.step()

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            # name contains backbone and epoch+acc
            stamp = time.strftime("%Y%m%d-%H%M%S")
            fname = os.path.join(
                args.save_dir,
                f'best_{args.backbone.replace(":","-")}_epoch{epoch}_acc{val_acc:.4f}_{stamp}.pt',
            )
            save_checkpoint(
                {"model_state": model.state_dict(), "epoch": epoch, "acc": val_acc},
                fname,
            )
            print(f"Saved best model to {fname}")

    print("Training finished")


if __name__ == "__main__":
    main()
