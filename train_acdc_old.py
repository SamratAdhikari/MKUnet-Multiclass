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
from utils.dataloader_acdc import get_acdc_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["background", "RV", "myocardium", "LV"]


def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ("1", "true", "t", "yes", "y")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=4, ce_weight=0.5, dice_weight=0.5, include_background=False):
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


def aggregate_metrics(model, loader, num_classes=4):
    model.eval()
    inter = torch.zeros(num_classes, dtype=torch.float64)
    pred_sum = torch.zeros(num_classes, dtype=torch.float64)
    gt_sum = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            images, masks = batch[0].to(device), batch[1].to(device)
            logits = model(images)[0]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.softmax(logits, dim=1).argmax(dim=1)
            for c in range(num_classes):
                pc = pred == c; gc = masks == c
                inter[c] += (pc & gc).sum().cpu()
                pred_sum[c] += pc.sum().cpu()
                gt_sum[c] += gc.sum().cpu()
    dice = (2 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
    iou = (inter + 1e-6) / (pred_sum + gt_sum - inter + 1e-6)
    return dice.numpy(), iou.numpy()


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


def main(opt):
    seed_everything(opt.seed)
    configs = {
        "MK_UNet_T": [4,8,16,24,32], "MK_UNet_S": [8,16,32,48,80],
        "MK_UNet": [16,32,64,96,160], "MK_UNet_M": [32,64,128,192,320],
        "MK_UNet_L": [64,128,256,384,512],
    }
    model = MK_UNet(num_classes=4, in_channels=1, channels=configs[opt.network], deep_supervision=True).to(device)
    criterion = DiceCELoss(num_classes=4, include_background=opt.dice_background)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if (opt.amp and device.type == "cuda") else None

    train_loader = get_acdc_loader(opt.data_root, "train", opt.batchsize, opt.img_size, opt.augmentation, num_workers=opt.num_workers)
    val_loader = get_acdc_loader(opt.data_root, "val", opt.test_batchsize, opt.img_size, False, shuffle=False, num_workers=opt.num_workers)
    test_loader = get_acdc_loader(opt.data_root, "test", opt.test_batchsize, opt.img_size, False, shuffle=False, num_workers=opt.num_workers)

    run_id = f"ACDC_{opt.network}_bs{opt.batchsize}_sz{opt.img_size}_lr{opt.lr}_e{opt.epochs}_seed{opt.seed}_{time.strftime('%H%M%S')}"
    save_dir = os.path.join(opt.model_root, run_id); os.makedirs(save_dir, exist_ok=True)
    best_score, best_epoch = -1, 0
    history = []

    print("Run ID:", run_id)
    print("Device:", device)
    for epoch in range(1, opt.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, scaler, opt)
        vd, vi = aggregate_metrics(model, val_loader, 4)
        fg_mean = float(vd[1:].mean())
        history.append({"epoch": epoch, "loss": loss, "val_mean_fg_dice": fg_mean,
                        **{f"val_dice_{CLASS_NAMES[c]}": float(vd[c]) for c in range(4)}})
        print(f"Epoch {epoch:03d}/{opt.epochs} loss={loss:.4f} val_fg_dice={fg_mean:.4f} "
              f"RV={vd[1]:.4f} MYO={vd[2]:.4f} LV={vd[3]:.4f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"{run_id}-last.pth"))
        if fg_mean > best_score:
            best_score, best_epoch = fg_mean, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"{run_id}-best.pth"))

    model.load_state_dict(torch.load(os.path.join(save_dir, f"{run_id}-best.pth"), map_location=device))
    td, ti = aggregate_metrics(model, test_loader, 4)
    summary = {
        "run_id": run_id, "best_epoch": best_epoch, "best_val_mean_foreground_dice": best_score,
        "test_mean_foreground_dice": float(td[1:].mean()),
        "test_mean_foreground_iou": float(ti[1:].mean()),
        "test_dice": {CLASS_NAMES[c]: float(td[c]) for c in range(4)},
        "test_iou": {CLASS_NAMES[c]: float(ti[c]) for c in range(4)},
        "config": vars(opt),
    }
    with open(os.path.join(save_dir, "history.json"), "w") as f: json.dump(history, f, indent=2)
    with open(os.path.join(save_dir, "summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="MK_UNet", choices=["MK_UNet_T","MK_UNet_S","MK_UNet","MK_UNet_M","MK_UNet_L"])
    p.add_argument("--data_root", default="./data/ACDC_npz")
    p.add_argument("--model_root", default="./model_pth_acdc")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--test_batchsize", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augmentation", type=str2bool, default=False)
    p.add_argument("--amp", type=str2bool, default=False)
    p.add_argument("--multi_scale", type=str2bool, default=True)
    p.add_argument("--dice_background", type=str2bool, default=False)
    p.add_argument("--num_workers", type=int, default=4)
    main(p.parse_args())
