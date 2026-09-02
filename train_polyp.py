import argparse
import copy
import json
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from utils.utils import AvgMeter, cal_params_flops, clip_gradient


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def structure_loss(pred, mask):
    """Paper-aligned weighted BCE + weighted IoU (1:1)."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum((2, 3))
    union = ((pred_prob + mask) * weit).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_coefficient(pred, gt):
    eps = 1e-6
    pred, gt = pred.reshape(-1), gt.reshape(-1)
    inter = (pred * gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def iou_coefficient(pred, gt):
    eps = 1e-6
    pred, gt = pred.reshape(-1), gt.reshape(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)


def evaluate(model, dataset_root, split, opt):
    loader = get_loader(
        os.path.join(dataset_root, split, "images"),
        os.path.join(dataset_root, split, "masks"),
        opt.test_batchsize,
        opt.img_size,
        shuffle=False,
        split="test",
        color_image=opt.color_image,
        num_workers=opt.num_workers,
    )
    model.eval()
    dsum = isum = n = 0.0
    with torch.no_grad():
        for images, gts, original_shapes, _ in loader:
            images = images.to(device, non_blocking=True)
            gts = gts.float().to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            for i in range(images.size(0)):
                h_orig = int(original_shapes[0][i])
                w_orig = int(original_shapes[1][i])
                p = F.interpolate(logits[i:i+1], (h_orig, w_orig), mode="bilinear", align_corners=False)
                p = torch.sigmoid(p).squeeze()
                p = (p - p.min()) / (p.max() - p.min() + 1e-8)
                g = F.interpolate(gts[i:i+1], (h_orig, w_orig), mode="nearest").squeeze()
                pb = (p >= 0.5).float()
                gb = (g >= 0.2).float()
                dsum += dice_coefficient(pb, gb).item()
                isum += iou_coefficient(pb, gb).item()
                n += 1
    if n == 0:
        raise RuntimeError(f"No samples found for {split}")
    return dsum / n, isum / n


def amp_context(enabled):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def train_one_epoch(loader, model, optimizer, scaler, epoch, opt):
    model.train()
    meter = AvgMeter()
    size_rates = [0.75, 1.0, 1.25] if opt.multi_scale else [1.0]

    for step, (images, gts) in enumerate(loader, 1):
        base_images = images.to(device, non_blocking=True)
        base_gts = gts.float().to(device, non_blocking=True)

        for rate in size_rates:
            optimizer.zero_grad(set_to_none=True)
            scaled_images, scaled_gts = base_images, base_gts
            if rate != 1.0:
                train_size = max(32, int(round(opt.img_size * rate / 32) * 32))
                scaled_images = F.interpolate(base_images, (train_size, train_size), mode="bilinear", align_corners=False)
                scaled_gts = F.interpolate(base_gts, (train_size, train_size), mode="nearest")

            with amp_context(opt.amp):
                outputs = model(scaled_images)
                pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                loss = structure_loss(pred, scaled_gts)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()

            if rate == 1.0:
                meter.update(loss.detach(), base_images.size(0))

        if step % 100 == 0 or step == len(loader):
            print(f"{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}] Step [{step:04d}/{len(loader):04d}] LR {optimizer.param_groups[0]['lr']:.6f} Loss {meter.show():.4f}")
    return float(meter.show())


def main(opt):
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = not opt.deterministic

    configs = {
        "MK_UNet_T": [4, 8, 16, 24, 32],
        "MK_UNet_S": [8, 16, 32, 48, 80],
        "MK_UNet": [16, 32, 64, 96, 160],
        "MK_UNet_M": [32, 64, 128, 192, 320],
        "MK_UNet_L": [64, 128, 256, 384, 512],
    }
    channels = configs[opt.network]
    dataset_root = os.path.join(opt.data_root, opt.dataset_name)
    required = [os.path.join(dataset_root, s, k) for s in ("train", "val", "test") for k in ("images", "masks")]
    for p in required:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)

    os.makedirs(opt.model_root, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    all_results = []

    for run in range(1, opt.num_runs + 1):
        seed = opt.seed + run - 1
        set_seed(seed)
        timestamp = time.strftime("%H%M%S")
        run_id = (f"{opt.dataset_name}_{opt.network}_bs{opt.batchsize}_sz{opt.img_size}_"
                  f"lr{opt.lr}_e{opt.epoch}_amp{opt.amp}_aug{opt.augmentation}_seed{seed}_run{run}_t{timestamp}")
        save_dir = os.path.join(opt.model_root, run_id)
        os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join("logs", f"train_log_{run_id}.log"), level=logging.INFO,
                            format="[%(asctime)s] %(message)s", force=True)

        model = MK_UNet(num_classes=1, in_channels=3, channels=channels, deep_supervision=False).to(device)
        try:
            cal_params_flops(copy.deepcopy(model), opt.img_size, logging)
        except Exception as e:
            print("Complexity calculation skipped:", e)

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=True) if (opt.amp and device.type == "cuda") else None
        loader = get_loader(os.path.join(dataset_root, "train", "images"), os.path.join(dataset_root, "train", "masks"),
                            opt.batchsize, opt.img_size, shuffle=True, augmentation=opt.augmentation,
                            split="train", color_image=opt.color_image, num_workers=opt.num_workers)

        best_dice, best_iou, best_epoch = -1.0, 0.0, 0
        start = time.time()
        for epoch in range(1, opt.epoch + 1):
            loss = train_one_epoch(loader, model, optimizer, scaler, epoch, opt)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{run_id}-last.pth"))
            val_dice, val_iou = evaluate(model, dataset_root, "val", opt)
            print(f"Epoch {epoch}: val Dice={val_dice:.4f}, IoU={val_iou:.4f}")
            logging.info(f"Epoch {epoch}, Loss {loss:.4f}, Val Dice {val_dice:.4f}, Val IoU {val_iou:.4f}")
            if val_dice > best_dice:
                best_dice, best_iou, best_epoch = val_dice, val_iou, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, f"{run_id}-best.pth"))

        model.load_state_dict(torch.load(os.path.join(save_dir, f"{run_id}-best.pth"), map_location=device))
        test_dice, test_iou = evaluate(model, dataset_root, "test", opt)
        elapsed = time.time() - start
        result = dict(run=run, run_id=run_id, seed=seed, best_epoch=best_epoch,
                      val_dice=best_dice, val_iou=best_iou,
                      test_dice=test_dice, test_iou=test_iou, training_time=elapsed,
                      batchsize=opt.batchsize, img_size=opt.img_size, amp=opt.amp)
        all_results.append(result)
        summary = (f"FINAL RESULTS: {run_id}\nBest Epoch: {best_epoch}\nBest Val Dice: {best_dice:.4f}\n"
                   f"Best Val IoU: {best_iou:.4f}\nFinal Test Dice: {test_dice:.4f}\nFinal Test IoU: {test_iou:.4f}\n"
                   f"Total Train Time: {elapsed:.2f}s")
        print(summary)
        logging.info(summary)
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_json = f"polyp_summary_{opt.dataset_name}_{opt.network}_{int(time.time())}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    dices = np.array([r["test_dice"] for r in all_results])
    ious = np.array([r["test_iou"] for r in all_results])
    print(f"Mean Test Dice: {dices.mean():.4f}")
    print(f"Mean Test IoU: {ious.mean():.4f}")
    if len(dices) > 1:
        print(f"Std Test Dice: {dices.std(ddof=1):.4f}")
        print(f"Std Test IoU: {ious.std(ddof=1):.4f}")
    print("Saved:", out_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="MK_UNet", choices=["MK_UNet_T", "MK_UNet_S", "MK_UNet", "MK_UNet_M", "MK_UNet_L"])
    p.add_argument("--dataset_name", default="ClinicDB", choices=["ClinicDB", "ColonDB"])
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--test_batchsize", type=int, default=16)
    p.add_argument("--img_size", type=int, default=352)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--augmentation", type=str2bool, default=False)
    p.add_argument("--color_image", type=str2bool, default=True)
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", type=str2bool, default=False)
    p.add_argument("--multi_scale", type=str2bool, default=True)
    p.add_argument("--deterministic", type=str2bool, default=False)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_root", default="./data/polyp/target")
    p.add_argument("--model_root", default="./model_pth")
    main(p.parse_args())
