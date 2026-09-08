import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mkunet_network import MK_UNet
from utils.dataloader_acdc import get_acdc_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "t", "yes", "y")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BinaryHeartDiceCELoss(nn.Module):
    """Binary whole-heart localizer loss.

    Original ACDC labels are converted to:
      0 = background
      1 = any cardiac foreground (RV, MYO, or LV)
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def foreground_dice_loss(logits, target):
        probs = torch.softmax(logits, dim=1)[:, 1]
        target_fg = (target == 1).float()
        dims = (0, 1, 2)
        inter = (probs * target_fg).sum(dims)
        denom = probs.sum(dims) + target_fg.sum(dims)
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        return 1.0 - dice

    def forward(self, logits, target):
        return (
            self.ce_weight * self.ce(logits, target)
            + self.dice_weight * self.foreground_dice_loss(logits, target)
        )


def adjust_lr(optimizer, base_lr, epoch, total_epochs, power=0.9, warmup_epochs=5):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        lr = base_lr * (1.0 - progress) ** power

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def amp_context(enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def train_epoch(model, loader, optimizer, criterion, scaler, amp=True, clip=0.5):
    model.train()
    running = 0.0
    count = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        # Whole heart = RV or MYO or LV.
        masks = (masks.to(device, non_blocking=True) > 0).long()

        optimizer.zero_grad(set_to_none=True)

        with amp_context(amp):
            outputs = model(images)
            losses = [criterion(out, masks) for out in outputs]
            loss = torch.stack(losses).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        running += loss.item() * images.size(0)
        count += images.size(0)

    return running / max(count, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    intersection = 0.0
    pred_sum = 0.0
    gt_sum = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        target = (masks.to(device, non_blocking=True) > 0)

        logits = model(images)[0]
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pred = logits.argmax(dim=1) == 1
        intersection += (pred & target).sum().item()
        pred_sum += pred.sum().item()
        gt_sum += target.sum().item()

    return (2.0 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)


def main(opt):
    seed_everything(opt.seed)

    configs = {
        "MK_UNet_T": [4, 8, 16, 24, 32],
        "MK_UNet_S": [8, 16, 32, 48, 80],
        "MK_UNet": [16, 32, 64, 96, 160],
        "MK_UNet_M": [32, 64, 128, 192, 320],
        "MK_UNet_L": [64, 128, 256, 384, 512],
    }

    model = MK_UNet(
        num_classes=2,
        in_channels=1,
        channels=configs[opt.network],
        deep_supervision=True,
    ).to(device)

    train_loader = get_acdc_loader(
        opt.data_root,
        "train",
        opt.batchsize,
        opt.img_size,
        opt.augmentation,
        num_workers=opt.num_workers,
    )
    val_loader = get_acdc_loader(
        opt.data_root,
        "val",
        opt.test_batchsize,
        opt.img_size,
        False,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    criterion = BinaryHeartDiceCELoss(
        ce_weight=opt.ce_weight,
        dice_weight=opt.dice_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
    )
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if opt.amp and device.type == "cuda"
        else None
    )

    out_dir = Path(opt.model_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "roi_localizer-best.pth"
    last_path = out_dir / "roi_localizer-last.pth"
    history_path = out_dir / "roi_localizer-history.json"

    best_dice = -1.0
    best_epoch = 0
    history = []

    print("Device:", device)
    print("Training whole-heart ROI localizer")
    print("Full-slice data root:", opt.data_root)

    for epoch in range(1, opt.epochs + 1):
        lr = adjust_lr(
            optimizer,
            opt.lr,
            epoch,
            opt.epochs,
            power=opt.lr_power,
            warmup_epochs=opt.warmup_epochs,
        )

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            amp=opt.amp,
            clip=opt.clip,
        )
        val_dice = evaluate(model, val_loader)

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": loss,
            "val_whole_heart_dice": val_dice,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{opt.epochs} "
            f"lr={lr:.6f} loss={loss:.4f} "
            f"val_whole_heart_dice={val_dice:.4f}"
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "network": opt.network,
            "channels": configs[opt.network],
            "img_size": opt.img_size,
            "num_classes": 2,
            "in_channels": 1,
            "deep_supervision": True,
            "epoch": epoch,
            "val_whole_heart_dice": val_dice,
        }
        torch.save(checkpoint, last_path)

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save(checkpoint, best_path)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print("\nFinished localizer training.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation whole-heart Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data/ACDC_npz")
    p.add_argument("--model_root", default="./model_pth_acdc_roi_localizer")
    p.add_argument(
        "--network",
        default="MK_UNet_S",
        choices=["MK_UNet_T", "MK_UNet_S", "MK_UNet", "MK_UNet_M", "MK_UNet_L"],
        help="A small MK-UNet is usually sufficient for coarse whole-heart localization.",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--test_batchsize", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentation", type=str2bool, default=True)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ce_weight", type=float, default=0.5)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--lr_power", type=float, default=0.9)
    p.add_argument("--warmup_epochs", type=int, default=5)
    main(p.parse_args())
