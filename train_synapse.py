import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mkunet_network import MK_UNet
from utils.dataloader_synapse import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_synapse_train_loader,
    get_synapse_val_loader,
    get_synapse_volume_loader,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "t", "yes", "y")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ce_weight=0.5, dice_weight=0.5, include_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background = include_background
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        if not self.include_background:
            dice = dice[1:]
        return 1 - dice.mean()

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice_loss(logits, target)


def amp_context(enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def train_epoch(model, loader, optimizer, criterion, scaler, opt):
    model.train()
    total_loss = 0.0
    n = 0
    size_rates = [0.75, 1.0, 1.25] if opt.multi_scale else [1.0]

    for images, masks in loader:
        base_images = images.to(device, non_blocking=True)
        base_masks = masks.to(device, non_blocking=True)
        for rate in size_rates:
            optimizer.zero_grad(set_to_none=True)
            x, y = base_images, base_masks
            if rate != 1.0:
                s = max(32, int(round(opt.img_size * rate / 32) * 32))
                x = F.interpolate(base_images, (s, s), mode="bilinear", align_corners=False)
                y = F.interpolate(base_masks.unsqueeze(1).float(), (s, s), mode="nearest").squeeze(1).long()

            with amp_context(opt.amp):
                outputs = model(x)
                losses = [criterion(out, y) for out in outputs]
                loss = torch.stack(losses).mean()  # equal-weight deep supervision

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()

            if rate == 1.0:
                total_loss += loss.item() * base_images.size(0)
                n += base_images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_slices(model, loader, criterion, num_classes=NUM_CLASSES):
    """Fast per-slice validation, meant to run every epoch: same loss the
    model is trained with (averaged over deep-supervision heads) plus
    per-class Dice/IoU accumulated over every validation slice. This is the
    2D analogue of evaluate_volumes below, and is cheap enough to call after
    every epoch (unlike the volumetric eval, which resizes/reconstructs full
    3D cases and is only run periodically)."""
    model.eval()
    total_loss, n = 0.0, 0
    inter = np.zeros(num_classes, dtype=np.float64)
    psum = np.zeros(num_classes, dtype=np.float64)
    gsum = np.zeros(num_classes, dtype=np.float64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        losses = [criterion(out, masks) for out in outputs]
        loss = torch.stack(losses).mean()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)

        pred = torch.softmax(outputs[0], dim=1).argmax(dim=1)
        for c in range(num_classes):
            pc = (pred == c)
            gc = (masks == c)
            inter[c] += (pc & gc).sum().item()
            psum[c] += pc.sum().item()
            gsum[c] += gc.sum().item()

    val_loss = total_loss / max(n, 1)
    dice = (2 * inter + 1e-6) / (psum + gsum + 1e-6)
    iou = (inter + 1e-6) / (psum + gsum - inter + 1e-6)
    return val_loss, dice, iou


@torch.no_grad()
def evaluate_volumes(model, volume_loader, num_classes=NUM_CLASSES, img_size=224):
    """Per-case (per-volume) Dice, the standard Synapse evaluation protocol:
    run inference slice-by-slice, resize each prediction back to the
    original in-plane resolution, reconstruct the 3D volume, then compute
    Dice per organ per case. Reported metric is the mean over cases. Runs on
    the fully held-out test_vol_h5 volumes, so this is kept periodic
    (opt.eval_interval) rather than every epoch."""
    model.eval()
    per_case_dice = []  # list of arrays, shape (num_classes,)

    for images, label, (orig_h, orig_w), case_id in volume_loader:
        images = images.to(device)  # (D, 1, h, w)
        preds = []
        for i in range(0, images.shape[0], 8):  # chunk to bound memory
            chunk = images[i:i + 8]
            logits = model(chunk)[0]
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            preds.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu())
        pred_vol = torch.cat(preds, dim=0)  # (D, H, W)
        label = label.squeeze(0) if label.ndim == 4 else label

        dice = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            pc = (pred_vol == c)
            gc = (label == c)
            inter = (pc & gc).sum().item()
            denom = pc.sum().item() + gc.sum().item()
            dice[c] = (2 * inter + 1e-6) / (denom + 1e-6) if denom > 0 else np.nan
        per_case_dice.append(dice)

    per_case_dice = np.array(per_case_dice)  # (num_cases, num_classes)
    mean_dice_per_class = np.nanmean(per_case_dice, axis=0)
    return mean_dice_per_class, per_case_dice


def main(opt):
    seed_everything(opt.seed)
    configs = {
        "MK_UNet_T": [4, 8, 16, 24, 32], "MK_UNet_S": [8, 16, 32, 48, 80],
        "MK_UNet": [16, 32, 64, 96, 160], "MK_UNet_M": [32, 64, 128, 192, 320],
        "MK_UNet_L": [64, 128, 256, 384, 512],
    }
    model = MK_UNet(num_classes=NUM_CLASSES, in_channels=1, channels=configs[opt.network],
                     deep_supervision=True).to(device)
    criterion = DiceCELoss(num_classes=NUM_CLASSES, include_background=opt.dice_background)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if (opt.amp and device.type == "cuda") else None

    train_loader = get_synapse_train_loader(
        opt.data_root, opt.batchsize, opt.img_size, opt.augmentation,
        split="train", val_fraction=opt.val_fraction, seed=opt.seed,
        num_workers=opt.num_workers)
    val_loader = get_synapse_val_loader(
        opt.data_root, opt.val_batchsize, opt.img_size,
        val_fraction=opt.val_fraction, seed=opt.seed,
        num_workers=max(1, opt.num_workers // 2))
    test_vol_loader = get_synapse_volume_loader(
        opt.data_root, opt.img_size, num_workers=max(1, opt.num_workers // 2))

    print(f"Train slices: {len(train_loader.dataset)} "
          f"({len(train_loader.dataset.case_ids)} cases) | "
          f"Val slices: {len(val_loader.dataset)} "
          f"({len(val_loader.dataset.case_ids)} cases) | "
          f"Test volumes: {len(test_vol_loader.dataset)}")

    run_id = f"Synapse_{opt.network}_bs{opt.batchsize}_sz{opt.img_size}_lr{opt.lr}_e{opt.epochs}_seed{opt.seed}_{time.strftime('%H%M%S')}"
    save_dir = os.path.join(opt.model_root, run_id)
    os.makedirs(save_dir, exist_ok=True)
    best_score, best_epoch = -1, 0
    history = []

    print("Run ID:", run_id)
    print("Device:", device)
    for epoch in range(1, opt.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, scaler, opt)
        torch.save(model.state_dict(), os.path.join(save_dir, f"{run_id}-last.pth"))

        # Fast 2D validation, every epoch: val loss, per-class Dice, per-class IoU.
        val_loss, val_dice, val_iou = evaluate_slices(model, val_loader, criterion, NUM_CLASSES)
        val_fg_dice = float(np.mean(val_dice[1:]))
        val_fg_iou = float(np.mean(val_iou[1:]))

        record = {
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "val_mean_fg_dice": val_fg_dice,
            "val_mean_fg_iou": val_fg_iou,
            **{f"val_dice_{CLASS_NAMES[c]}": float(val_dice[c]) for c in range(NUM_CLASSES)},
            **{f"val_iou_{CLASS_NAMES[c]}": float(val_iou[c]) for c in range(NUM_CLASSES)},
        }

        print(f"Epoch {epoch:03d}/{opt.epochs} loss={loss:.4f} val_loss={val_loss:.4f} "
              f"val_fg_dice={val_fg_dice:.4f} val_fg_iou={val_fg_iou:.4f} | "
              + " ".join(f"{CLASS_NAMES[c]}={val_dice[c]:.3f}" for c in range(1, NUM_CLASSES)))

        # Best checkpoint is selected on the (cheap, every-epoch) val set —
        # test_vol_h5 stays untouched until the periodic/final check below.
        if val_fg_dice > best_score:
            best_score, best_epoch = val_fg_dice, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"{run_id}-best.pth"))

        # Slower, periodic sanity check on the fully held-out test volumes.
        if epoch % opt.eval_interval == 0 or epoch == opt.epochs:
            test_dice_per_class, _ = evaluate_volumes(model, test_vol_loader, NUM_CLASSES, opt.img_size)
            test_fg_mean = float(np.nanmean(test_dice_per_class[1:]))
            record.update({
                "test_mean_fg_dice": test_fg_mean,
                **{f"test_dice_{CLASS_NAMES[c]}": float(test_dice_per_class[c]) for c in range(NUM_CLASSES)},
            })
            print(f"           [periodic test-volume check] test_fg_dice={test_fg_mean:.4f} | "
                  + " ".join(f"{CLASS_NAMES[c]}={test_dice_per_class[c]:.3f}" for c in range(1, NUM_CLASSES)))

        history.append(record)

    best_path = os.path.join(save_dir, f"{run_id}-best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    val_loss, val_dice, val_iou = evaluate_slices(model, val_loader, criterion, NUM_CLASSES)
    test_dice_per_class, per_case = evaluate_volumes(model, test_vol_loader, NUM_CLASSES, opt.img_size)
    summary = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_val_mean_foreground_dice": best_score,
        "final_val_loss": val_loss,
        "final_val_dice_per_class": {CLASS_NAMES[c]: float(val_dice[c]) for c in range(NUM_CLASSES)},
        "final_val_iou_per_class": {CLASS_NAMES[c]: float(val_iou[c]) for c in range(NUM_CLASSES)},
        "final_test_mean_foreground_dice": float(np.nanmean(test_dice_per_class[1:])),
        "test_dice_per_class": {CLASS_NAMES[c]: float(test_dice_per_class[c]) for c in range(NUM_CLASSES)},
        "num_test_cases": int(per_case.shape[0]),
        "num_train_cases": len(train_loader.dataset.case_ids),
        "num_val_cases": len(val_loader.dataset.case_ids),
        "config": vars(opt),
    }
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="MK_UNet",
                   choices=["MK_UNet_T", "MK_UNet_S", "MK_UNet", "MK_UNet_M", "MK_UNet_L"])
    p.add_argument("--data_root", default="./data/Synapse")
    p.add_argument("--model_root", default="./model_pth_synapse")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--val_batchsize", type=int, default=16,
                   help="batch size for the fast per-epoch 2D validation pass")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="fraction of train_npz *cases* (not slices) held out for validation")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentation", type=str2bool, default=True)
    p.add_argument("--amp", type=str2bool, default=False)
    p.add_argument("--multi_scale", type=str2bool, default=True)
    p.add_argument("--dice_background", type=str2bool, default=False)
    p.add_argument("--eval_interval", type=int, default=10,
                   help="Run the (slower) volumetric test-set evaluation every N epochs.")
    p.add_argument("--num_workers", type=int, default=4)
    main(p.parse_args())
