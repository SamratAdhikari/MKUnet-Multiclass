import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95
from tqdm import tqdm

from mkunet_network import MK_UNet
from utils.dataloader_acdc import get_acdc_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["background", "RV", "myocardium", "LV"]


def main(opt):
    configs = {
        "MK_UNet_T": [4,8,16,24,32], "MK_UNet_S": [8,16,32,48,80],
        "MK_UNet": [16,32,64,96,160], "MK_UNet_M": [32,64,128,192,320],
        "MK_UNet_L": [64,128,256,384,512],
    }
    model = MK_UNet(num_classes=4, in_channels=1, channels=configs[opt.network], deep_supervision=True).to(device)
    ckpt = os.path.join(opt.model_root, opt.run_id, f"{opt.run_id}-best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()

    loader = get_acdc_loader(opt.data_root, "test", opt.batchsize, opt.img_size, False, shuffle=False, num_workers=opt.num_workers)
    pred_dir = os.path.join(opt.pred_root, opt.run_id); os.makedirs(pred_dir, exist_ok=True)

    inter = np.zeros(4, dtype=np.float64); psum = np.zeros(4); gsum = np.zeros(4)
    hd_values = {c: [] for c in range(1,4)}
    rows = []
    with torch.no_grad():
        for images, masks, orig_shapes, names in tqdm(loader, desc="ACDC test"):
            images = images.to(device); masks = masks.to(device)
            logits = model(images)[0]
            for i in range(images.size(0)):
                h, w = int(orig_shapes[0][i]), int(orig_shapes[1][i])
                logit = F.interpolate(logits[i:i+1], (h,w), mode="bilinear", align_corners=False)
                pred = torch.softmax(logit, dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                gt = F.interpolate(masks[i:i+1].unsqueeze(1).float(), (h,w), mode="nearest").squeeze().cpu().numpy().astype(np.uint8)

                row = {"Name": names[i]}
                for c in range(4):
                    pc, gc = pred == c, gt == c
                    it = np.logical_and(pc, gc).sum(); den = pc.sum() + gc.sum()
                    inter[c] += it; psum[c] += pc.sum(); gsum[c] += gc.sum()
                    row[f"Dice_{CLASS_NAMES[c]}"] = float((2*it+1e-6)/(den+1e-6))
                    if c > 0 and pc.any() and gc.any():
                        try: hd_values[c].append(float(hd95(pc, gc)))
                        except Exception: pass
                rows.append(row)
                np.save(os.path.join(pred_dir, names[i] + ".npy"), pred)
                cv2.imwrite(os.path.join(pred_dir, names[i] + ".png"), (pred * 85).astype(np.uint8))

    dice = (2*inter+1e-6)/(psum+gsum+1e-6)
    iou = (inter+1e-6)/(psum+gsum-inter+1e-6)
    summary = {
        "run_id": opt.run_id,
        "mean_foreground_dice": float(dice[1:].mean()),
        "mean_foreground_iou": float(iou[1:].mean()),
        "dice": {CLASS_NAMES[c]: float(dice[c]) for c in range(4)},
        "iou": {CLASS_NAMES[c]: float(iou[c]) for c in range(4)},
        "hd95_mean_over_present_slices": {CLASS_NAMES[c]: (float(np.mean(hd_values[c])) if hd_values[c] else None) for c in range(1,4)},
    }
    os.makedirs(opt.results_root, exist_ok=True)
    pd.DataFrame(rows).to_excel(os.path.join(opt.results_root, f"ACDC_{opt.run_id}_per_slice.xlsx"), index=False)
    with open(os.path.join(opt.results_root, f"ACDC_{opt.run_id}_summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--network", default="MK_UNet")
    p.add_argument("--data_root", default="./data/ACDC_npz")
    p.add_argument("--model_root", default="./model_pth_acdc")
    p.add_argument("--pred_root", default="./predictions_acdc")
    p.add_argument("--results_root", default="./results_acdc")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batchsize", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    main(p.parse_args())
