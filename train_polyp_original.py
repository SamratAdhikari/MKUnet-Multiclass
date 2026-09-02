# ============================================================
# train_polyp.py
# Paper-aligned MK-UNet training for ClinicDB / ColonDB
# ============================================================

import os
import time
import copy
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Project-specific imports
from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from utils.utils import clip_gradient, AvgMeter, cal_params_flops


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# BOOLEAN ARGUMENT PARSER
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    if v.lower() in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError(
        "Boolean value expected."
    )


# ============================================================
# OPTIONAL REPRODUCIBLE SEED
# Each run uses a different seed.
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOSS
#
# Paper:
# weighted BCE + weighted IoU, 1:1
# ============================================================

def structure_loss(pred, mask):

    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(
            mask,
            kernel_size=31,
            stride=1,
            padding=15
        ) - mask
    )

    # Weighted BCE
    wbce = F.binary_cross_entropy_with_logits(
        pred,
        mask,
        reduction="none"
    )

    wbce = (
        (weit * wbce).sum(dim=(2, 3))
        /
        weit.sum(dim=(2, 3))
    )

    # Weighted IoU
    pred_prob = torch.sigmoid(pred)

    inter = (
        (pred_prob * mask) * weit
    ).sum(dim=(2, 3))

    union = (
        (pred_prob + mask) * weit
    ).sum(dim=(2, 3))

    wiou = 1 - (
        (inter + 1)
        /
        (union - inter + 1)
    )

    return (wbce + wiou).mean()


# ============================================================
# METRICS
# ============================================================

def dice_coefficient(predicted, labels):

    smooth = 1e-6

    predicted = predicted.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    intersection = (
        predicted * labels
    ).sum()

    return (
        2.0 * intersection + smooth
    ) / (
        predicted.sum()
        + labels.sum()
        + smooth
    )


def iou_coefficient(predicted, labels):

    smooth = 1e-6

    predicted = predicted.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    intersection = (
        predicted * labels
    ).sum()

    union = (
        predicted.sum()
        + labels.sum()
        - intersection
    )

    return (
        intersection + smooth
    ) / (
        union + smooth
    )


# ============================================================
# EVALUATION
#
# Used for validation during training and final test.
# This follows the same prediction post-processing approach
# used by your test_polyp.py.
# ============================================================

def evaluate(model, dataset_root, split, opt):

    data_path = os.path.join(
        dataset_root,
        split
    )

    image_root = os.path.join(
        data_path,
        "images"
    )

    gt_root = os.path.join(
        data_path,
        "masks"
    )

    loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        split="test",
        color_image=opt.color_image
    )

    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_images = 0

    with torch.no_grad():

        for pack in loader:

            images, gts, original_shapes, _ = pack

            images = images.to(device)
            gts = gts.float().to(device)

            outputs = model(images)

            predictions = (
                outputs[0]
                if isinstance(outputs, (list, tuple))
                else outputs
            )

            for i in range(images.size(0)):

                h_orig = int(
                    original_shapes[0][i]
                )

                w_orig = int(
                    original_shapes[1][i]
                )

                # --------------------------------------------
                # Resize prediction to original image size
                # --------------------------------------------

                pred = predictions[i].unsqueeze(0)

                pred = F.interpolate(
                    pred,
                    size=(h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False
                )

                pred = torch.sigmoid(
                    pred
                ).squeeze()

                # Same normalization as test_polyp.py
                pred_min = pred.min()
                pred_max = pred.max()

                pred = (
                    pred - pred_min
                ) / (
                    pred_max
                    - pred_min
                    + 1e-8
                )

                # --------------------------------------------
                # Resize GT
                # --------------------------------------------

                gt = gts[i].unsqueeze(0)

                gt = F.interpolate(
                    gt,
                    size=(h_orig, w_orig),
                    mode="nearest"
                ).squeeze()

                # --------------------------------------------
                # Binarization
                # --------------------------------------------

                pred_binary = (
                    pred >= 0.5
                ).float()

                gt_binary = (
                    gt >= 0.2
                ).float()

                # --------------------------------------------
                # Metrics
                # --------------------------------------------

                dice = dice_coefficient(
                    pred_binary,
                    gt_binary
                )

                iou = iou_coefficient(
                    pred_binary,
                    gt_binary
                )

                total_dice += dice.item()
                total_iou += iou.item()
                total_images += 1

    if total_images == 0:
        raise RuntimeError(
            f"No images found in {data_path}"
        )

    mean_dice = (
        total_dice / total_images
    )

    mean_iou = (
        total_iou / total_images
    )

    return mean_dice, mean_iou


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    train_loader,
    model,
    optimizer,
    epoch,
    opt
):

    model.train()

    loss_record = AvgMeter()

    # Paper multi-scale settings
    size_rates = [
        0.75,
        1.0,
        1.25
    ]

    total_step = len(train_loader)

    for step, (images, gts) in enumerate(
        train_loader,
        start=1
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        # Preserve the ORIGINAL batch.
        #
        # Previous code modified `images` during the first
        # scale and reused the resized tensor at later scales.
        # ----------------------------------------------------

        base_images = images.to(device)
        base_gts = gts.float().to(device)

        for rate in size_rates:

            optimizer.zero_grad()

            # Always derive each scale from ORIGINAL tensors.
            scaled_images = base_images
            scaled_gts = base_gts

            if rate != 1.0:

                train_size = int(
                    round(
                        opt.img_size
                        * rate
                        / 32
                    ) * 32
                )

                scaled_images = F.interpolate(
                    base_images,
                    size=(
                        train_size,
                        train_size
                    ),
                    mode="bilinear",
                    align_corners=True
                )

                scaled_gts = F.interpolate(
                    base_gts,
                    size=(
                        train_size,
                        train_size
                    ),
                    mode="nearest"
                )

            # Forward
            outputs = model(
                scaled_images
            )

            prediction = (
                outputs[0]
                if isinstance(
                    outputs,
                    (list, tuple)
                )
                else outputs
            )

            # Loss
            loss = structure_loss(
                prediction,
                scaled_gts
            )

            # Backpropagation
            loss.backward()

            # Paper gradient clipping = 0.5
            clip_gradient(
                optimizer,
                opt.clip
            )

            optimizer.step()

            # Record normal-resolution loss
            if rate == 1.0:

                loss_record.update(
                    loss.detach(),
                    base_images.size(0)
                )

        if (
            step % 100 == 0
            or step == total_step
        ):

            print(
                f"{datetime.now()} "
                f"Epoch "
                f"[{epoch:03d}/{opt.epoch:03d}], "
                f"Step "
                f"[{step:04d}/{total_step:04d}], "
                f"LR: "
                f"{optimizer.param_groups[0]['lr']:.6f}, "
                f"Loss: "
                f"{loss_record.show():.4f}"
            )

    return float(
        loss_record.show()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Dataset / network
    # --------------------------------------------------------

    parser.add_argument(
        "--network",
        type=str,
        default="MK_UNet",
        choices=[
            "MK_UNet_T",
            "MK_UNet_S",
            "MK_UNet",
            "MK_UNet_M",
            "MK_UNet_L"
        ]
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ClinicDB",
        choices=[
            "ClinicDB",
            "ColonDB"
        ]
    )

    # --------------------------------------------------------
    # Paper hyperparameters
    # --------------------------------------------------------

    parser.add_argument(
        "--epoch",
        type=int,
        default=200
    )

    # Paper: LR = 1e-4
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    # Paper: batch size 16
    parser.add_argument(
        "--batchsize",
        type=int,
        default=16
    )

    parser.add_argument(
        "--test_batchsize",
        type=int,
        default=16
    )

    # ClinicDB / ColonDB = 352 x 352
    parser.add_argument(
        "--img_size",
        type=int,
        default=352
    )

    # Paper gradient clipping
    parser.add_argument(
        "--clip",
        type=float,
        default=0.5
    )

    # Paper explicitly says no augmentation
    parser.add_argument(
        "--augmentation",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--color_image",
        type=str2bool,
        default=True
    )

    # --------------------------------------------------------
    # Number of independent runs
    # Paper reports mean of 5.
    # --------------------------------------------------------

    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help=(
            "Number of independent runs. "
            "Paper reports 5-run averages."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/polyp/target"
    )

    parser.add_argument(
        "--model_root",
        type=str,
        default="./model_pth"
    )

    opt = parser.parse_args()


    # ========================================================
    # NETWORK CONFIGURATIONS
    # ========================================================

    NET_CONFIGS = {

        "MK_UNet_T":
            [4, 8, 16, 24, 32],

        "MK_UNet_S":
            [8, 16, 32, 48, 80],

        "MK_UNet":
            [16, 32, 64, 96, 160],

        "MK_UNet_M":
            [32, 64, 128, 192, 320],

        "MK_UNet_L":
            [64, 128, 256, 384, 512]
    }

    channels = NET_CONFIGS[
        opt.network
    ]


    # ========================================================
    # DATASET PATHS
    # ========================================================

    dataset_root = os.path.join(
        opt.data_root,
        opt.dataset_name
    )

    train_path = os.path.join(
        dataset_root,
        "train"
    )

    train_image_root = os.path.join(
        train_path,
        "images"
    )

    train_gt_root = os.path.join(
        train_path,
        "masks"
    )

    print("\nDataset:", opt.dataset_name)
    print("Dataset root:", dataset_root)
    print("Training images:", train_image_root)
    print("Training masks:", train_gt_root)


    # ========================================================
    # BASIC PATH CHECKS
    # ========================================================

    required_paths = [

        train_image_root,
        train_gt_root,

        os.path.join(
            dataset_root,
            "val",
            "images"
        ),

        os.path.join(
            dataset_root,
            "val",
            "masks"
        ),

        os.path.join(
            dataset_root,
            "test",
            "images"
        ),

        os.path.join(
            dataset_root,
            "test",
            "masks"
        )
    ]

    for path in required_paths:

        if not os.path.isdir(path):

            raise FileNotFoundError(
                f"Required dataset directory "
                f"not found:\n{path}"
            )


    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    os.makedirs(
        opt.model_root,
        exist_ok=True
    )

    os.makedirs(
        "logs",
        exist_ok=True
    )


    # ========================================================
    # STORE RESULTS FROM INDEPENDENT RUNS
    # ========================================================

    all_run_results = []


    # ========================================================
    # INDEPENDENT RUNS
    # ========================================================

    for run in range(
        1
    ):

        print("\n")
        print("=" * 70)
        print(
            f"STARTING RUN "
            f"{run}/{opt.num_runs}"
        )
        print("=" * 70)


        # ----------------------------------------------------
        # Each run starts from a fresh initialization
        # ----------------------------------------------------

        run_seed = (
            opt.seed + run - 1
        )

        set_seed(
            run_seed
        )

        timestamp = time.strftime(
            "%H%M%S"
        )

        run_id = (
            f"{opt.dataset_name}_"
            f"{opt.network}_"
            f"bs{opt.batchsize}_"
            f"lr{opt.lr}_"
            f"e{opt.epoch}_"
            f"aug{opt.augmentation}_"
            f"run{run}_"
            f"t{timestamp}"
        )

        save_path = os.path.join(
            opt.model_root,
            run_id
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logging.basicConfig(
            filename=os.path.join(
                "logs",
                f"train_log_{run_id}.log"
            ),
            level=logging.INFO,
            format=(
                "[%(asctime)s] "
                "%(message)s"
            ),
            force=True
        )


        # ----------------------------------------------------
        # Fresh model
        # ----------------------------------------------------

        model = MK_UNet(
            num_classes=1,
            in_channels=3,
            channels=channels
        )

        model = model.to(
            device
        )

        print(
            f"Network: {opt.network}"
        )

        print(
            f"Channels: {channels}"
        )

        print(
            f"Run seed: {run_seed}"
        )


        # ----------------------------------------------------
        # Model complexity
        #
        # deepcopy prevents THOP buffers from polluting
        # the actual training model.
        # ----------------------------------------------------

        cal_params_flops(
            copy.deepcopy(model),
            opt.img_size,
            logging
        )


        # ----------------------------------------------------
        # Optimizer
        #
        # PAPER:
        # AdamW
        # LR = 1e-4
        # WD = 1e-4
        #
        # IMPORTANT:
        # NO CosineAnnealingLR here.
        # ----------------------------------------------------

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt.lr,
            weight_decay=1e-4
        )


        # ----------------------------------------------------
        # Data loader
        #
        # Paper: augmentation = False
        # ----------------------------------------------------

        train_loader = get_loader(
            image_root=train_image_root,
            gt_root=train_gt_root,
            batchsize=opt.batchsize,
            trainsize=opt.img_size,
            shuffle=True,
            augmentation=opt.augmentation,
            split="train",
            color_image=opt.color_image
        )


        # ----------------------------------------------------
        # Best model tracking
        # ----------------------------------------------------

        best_val_dice = -1.0
        best_val_iou = 0.0
        best_epoch = 0

        run_start_time = time.time()


        # ====================================================
        # TRAINING
        # ====================================================

        for epoch in range(
            1,
            opt.epoch + 1
        ):

            # -----------------------------------------------
            # Train
            # -----------------------------------------------

            train_loss = train_one_epoch(
                train_loader,
                model,
                optimizer,
                epoch,
                opt
            )


            # -----------------------------------------------
            # Save latest checkpoint
            # -----------------------------------------------

            last_model_path = os.path.join(
                save_path,
                f"{run_id}-last.pth"
            )

            torch.save(
                model.state_dict(),
                last_model_path
            )


            # -----------------------------------------------
            # VALIDATION ONLY
            #
            # Do NOT repeatedly use the test set for
            # checkpoint selection.
            # -----------------------------------------------

            val_dice, val_iou = evaluate(
                model,
                dataset_root,
                "val",
                opt
            )

            print(
                f"Epoch: {epoch}, "
                f"Dataset: val, "
                f"Dice: {val_dice:.4f}, "
                f"IoU: {val_iou:.4f}"
            )

            logging.info(
                f"Epoch: {epoch}, "
                f"Loss: {train_loss:.4f}, "
                f"Val Dice: {val_dice:.4f}, "
                f"Val IoU: {val_iou:.4f}"
            )


            # -----------------------------------------------
            # Save best validation model
            # -----------------------------------------------

            if val_dice > best_val_dice:

                old_best = best_val_dice

                best_val_dice = val_dice
                best_val_iou = val_iou
                best_epoch = epoch

                best_model_path = os.path.join(
                    save_path,
                    f"{run_id}-best.pth"
                )

                torch.save(
                    model.state_dict(),
                    best_model_path
                )

                print(
                    "### Best Model Saved "
                    f"(Dice improved from "
                    f"{max(old_best, 0):.4f} "
                    f"to "
                    f"{best_val_dice:.4f}) ###"
                )


        # ====================================================
        # TRAINING COMPLETE
        # ====================================================

        training_time = (
            time.time()
            - run_start_time
        )


        # ====================================================
        # LOAD BEST VALIDATION CHECKPOINT
        # ====================================================

        best_model_path = os.path.join(
            save_path,
            f"{run_id}-best.pth"
        )

        checkpoint = torch.load(
            best_model_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint,
            strict=True
        )

        model.eval()


        # ====================================================
        # TEST SET
        #
        # Evaluate only AFTER training using the best
        # validation checkpoint.
        # ====================================================

        test_dice, test_iou = evaluate(
            model,
            dataset_root,
            "test",
            opt
        )


        # ====================================================
        # RUN SUMMARY
        # ====================================================

        summary = (
            "\n"
            + "=" * 60
            + "\n"
            + f"FINAL RESULTS: {run_id}\n"
            + f"Best Epoch: {best_epoch}\n"
            + f"Best Val Dice: "
              f"{best_val_dice:.4f}\n"
            + f"Best Val IoU: "
              f"{best_val_iou:.4f}\n"
            + f"Final Test Dice: "
              f"{test_dice:.4f}\n"
            + f"Final Test IoU: "
              f"{test_iou:.4f}\n"
            + f"Total Train Time: "
              f"{training_time:.2f}s\n"
            + "=" * 60
        )

        print(
            summary
        )

        logging.info(
            summary
        )


        # ----------------------------------------------------
        # Save run result
        # ----------------------------------------------------

        all_run_results.append({

            "run": run,

            "run_id": run_id,

            "seed": run_seed,

            "best_epoch":
                best_epoch,

            "val_dice":
                best_val_dice,

            "val_iou":
                best_val_iou,

            "test_dice":
                test_dice,

            "test_iou":
                test_iou,

            "training_time":
                training_time
        })


        # Free some memory before next run
        del model
        del optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ========================================================
    # MULTI-RUN SUMMARY
    # ========================================================

    test_dices = np.array([
        x["test_dice"]
        for x in all_run_results
    ])

    test_ious = np.array([
        x["test_iou"]
        for x in all_run_results
    ])


    print("\n")
    print("=" * 70)
    print(
        f"OVERALL RESULTS — "
        f"{opt.dataset_name}"
    )
    print("=" * 70)

    for result in all_run_results:

        print(
            f"Run {result['run']}: "
            f"Dice = "
            f"{result['test_dice']:.4f}, "
            f"IoU = "
            f"{result['test_iou']:.4f}, "
            f"Best epoch = "
            f"{result['best_epoch']}"
        )


    print("-" * 70)

    print(
        f"Mean Test Dice: "
        f"{test_dices.mean():.4f}"
    )

    print(
        f"Std Test Dice: "
        f"{test_dices.std(ddof=1):.4f}"
        if len(test_dices) > 1
        else
        "Std Test Dice: N/A "
        "(only one run)"
    )

    print(
        f"Mean Test IoU: "
        f"{test_ious.mean():.4f}"
    )

    print(
        f"Std Test IoU: "
        f"{test_ious.std(ddof=1):.4f}"
        if len(test_ious) > 1
        else
        "Std Test IoU: N/A "
        "(only one run)"
    )

    print("=" * 70)