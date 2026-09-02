import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95
from tqdm import tqdm

from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ("1", "true", "t", "yes", "y")


def dice(pred, gt):
    eps = 1e-6
    inter = (pred.reshape(-1) * gt.reshape(-1)).sum()
    return ((2 * inter + eps) / (pred.sum() + gt.sum() + eps)).item()


def iou(pred, gt):
    eps = 1e-6
    inter = (pred.reshape(-1) * gt.reshape(-1)).sum()
    union = pred.sum() + gt.sum() - inter
    return ((inter + eps) / (union + eps)).item()


def binary_metrics(pred, gt):
    tp = (pred * gt).sum().item(); tn = ((1-pred)*(1-gt)).sum().item()
    fp = (pred*(1-gt)).sum().item(); fn = ((1-pred)*gt).sum().item()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    try:
        h = float(hd95(pred.cpu().numpy().astype(bool), gt.cpu().numpy().astype(bool))) if pred.sum() > 0 and gt.sum() > 0 else 100.0
    except Exception:
        h = 100.0
    return sens, spec, prec, h


def main(opt):
    configs = {
        "MK_UNet_T": [4,8,16,24,32], "MK_UNet_S": [8,16,32,48,80],
        "MK_UNet": [16,32,64,96,160], "MK_UNet_M": [32,64,128,192,320],
        "MK_UNet_L": [64,128,256,384,512],
    }
    data_path = os.path.join(opt.test_path, opt.dataset_name, opt.split)
    loader = get_loader(os.path.join(data_path, "images"), os.path.join(data_path, "masks"),
                        opt.test_batchsize, opt.img_size, shuffle=False, split="test",
                        color_image=opt.color_image, num_workers=opt.num_workers)

    model = MK_UNet(num_classes=1, in_channels=3, channels=configs[opt.network], deep_supervision=False).to(device)
    ckpt = os.path.join(opt.model_root, opt.run_id, f"{opt.run_id}-best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()

    save_dir = os.path.join(opt.pred_root, opt.run_id, opt.dataset_name, opt.split)
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    with torch.no_grad():
        for images, gts, original_shapes, names in tqdm(loader):
            images = images.to(device); gts = gts.float().to(device)
            logits = model(images)[0]
            for i in range(images.size(0)):
                h, w = int(original_shapes[0][i]), int(original_shapes[1][i])
                p = F.interpolate(logits[i:i+1], (h,w), mode="bilinear", align_corners=False).sigmoid().squeeze()
                p = (p-p.min())/(p.max()-p.min()+1e-8)
                g = F.interpolate(gts[i:i+1], (h,w), mode="nearest").squeeze()
                pb, gb = (p>=0.5).float(), (g>=0.2).float()
                d, j = dice(pb,gb), iou(pb,gb)
                s,sp,pr,h95 = binary_metrics(pb,gb)
                rows.append(dict(Name=names[i], Dice=d, IoU=j, Sensitivity=s, Specificity=sp, Precision=pr, HD95=h95))
                cv2.imwrite(os.path.join(save_dir, names[i]), (pb.cpu().numpy()*255).astype(np.uint8))

    df = pd.DataFrame(rows)
    mean = df.mean(numeric_only=True).to_dict(); mean["Name"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    os.makedirs(opt.results_root, exist_ok=True)
    out = os.path.join(opt.results_root, f"Results_{opt.run_id}_{opt.dataset_name}_{opt.split}.xlsx")
    df.to_excel(out, index=False)
    print(df.tail(1).to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--network", default="MK_UNet")
    p.add_argument("--dataset_name", default="ClinicDB")
    p.add_argument("--split", default="test")
    p.add_argument("--img_size", type=int, default=352)
    p.add_argument("--test_batchsize", type=int, default=1)
    p.add_argument("--color_image", type=str2bool, default=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--test_path", default="./data/polyp/target")
    p.add_argument("--model_root", default="./model_pth")
    p.add_argument("--pred_root", default="./predictions_polyp")
    p.add_argument("--results_root", default="./results_polyp")
    main(p.parse_args())
