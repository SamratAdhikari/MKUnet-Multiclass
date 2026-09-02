# MK-UNet Step-2 Extension Package

This package extends the public MK-UNet code in two directions:

1. **Training-sensitivity / robustness study** on the original binary polyp task.
2. **Multi-class 2D medical segmentation** on **ACDC cardiac MRI**.

## Datasets in this package

### A. ClinicDB — robustness sweep
Used by `sweep_robustness.py` to compare batch size, input resolution, AMP/FP32, and random seed while keeping the original MK-UNet binary task.

Expected structure:

```text
data/polyp/target/ClinicDB/
  train/images  train/masks
  val/images    val/masks
  test/images   test/masks
```

### B. ColonDB — binary confirmation / reproduction
Supported by `train_polyp.py` and `test_polyp.py`. It is not used by the default robustness sweep, but can be run as a confirmation dataset.

### C. ACDC 2017 — NEW multi-class extension
This is the actual Step-2 architectural extension dataset.

Classes:

- 0 = background
- 1 = right ventricle (RV)
- 2 = myocardium
- 3 = left ventricle (LV)

`prepare_acdc.py` converts raw labeled NIfTI volumes into 2D `.npz` slices. The default split implemented here is patient-level 70/10/20:

- patient001–070: train
- patient071–080: validation
- patient081–100: test

This split is explicitly defined by this extension package for reproducibility; change `--train_end` / `--val_end` if you need to match a different published protocol.

## Important MK-UNet changes

### `mkunet_network.py`

- Every segmentation head already had `num_classes`; the extension exposes all four heads for multi-class deep supervision.
- `deep_supervision=False` preserves the original binary behavior: returns `[final_logits]`.
- `deep_supervision=True` returns four full-resolution raw-logit tensors.
- Fixed grayscale support: `in_channels=1` now really accepts one-channel MRI instead of always repeating grayscale to three channels.
- No Sigmoid/Softmax is embedded in the network. Training receives logits; inference applies Sigmoid for binary or Softmax for multi-class.

**Naming note:** the public code calls the last full-resolution tensor `p4` and returns `[p4]`, while the paper text calls `p1` the final prediction. This package treats the *actual full-resolution returned head* as `final` to avoid that naming ambiguity.

### `train_acdc.py`

- `MK_UNet(num_classes=4, in_channels=1, deep_supervision=True)`
- Dice + Cross Entropy loss, 0.5 / 0.5
- Foreground Dice is computed for RV, myocardium, and LV separately
- Best checkpoint selected by mean foreground validation Dice
- Deep supervision uses equal loss weight across all four heads

### `train_polyp.py`

- Fixes the independent-run loop so `--num_runs` is actually respected.
- Adds `--amp` and `--img_size`/batch-size sweep support.
- Keeps the paper-aligned binary weighted BCE + weighted IoU loss.
- Keeps multi-scale training `[0.75, 1.0, 1.25]` unless disabled.

### `utils/dataloader_polyp.py`

- Pairs image/mask by basename rather than relying on two separately sorted file lists.
- Returns original size as `(H, W)` (the previous PIL `.size` path returned `(W, H)`).

## Install

Start from the MK-UNet repository environment, then:

```bash
pip install -r requirements.txt
pip install -r requirements_extension.txt
```

## 1. Binary reproduction / robustness

Paper-like single run:

```bash
python train_polyp.py \
  --dataset_name ClinicDB \
  --network MK_UNet \
  --batchsize 16 \
  --img_size 352 \
  --lr 1e-4 \
  --epoch 200 \
  --augmentation false \
  --amp false \
  --num_runs 1
```

Three independent runs:

```bash
python train_polyp.py --dataset_name ClinicDB --num_runs 3 --seed 42
```

Robustness sweep:

```bash
python sweep_robustness.py --epochs 100
```

This defaults to:

- batch size: 4, 8, 16
- input: 256, 352
- precision: FP32, AMP
- seeds: 42, 43, 44
- dataset: ClinicDB

## 2. Prepare ACDC

If raw ACDC labeled NIfTI files are under `/content/ACDC_raw`:

```bash
python prepare_acdc.py \
  --raw_root /content/ACDC_raw \
  --out_root ./data/ACDC_npz
```

Result:

```text
data/ACDC_npz/
  train/*.npz
  val/*.npz
  test/*.npz
```

Each `.npz` contains:

- `image`: normalized 2D MRI slice
- `label`: integer mask in `{0,1,2,3}`

## 3. Train multi-class MK-UNet on ACDC

Recommended first Colab run:

```bash
python train_acdc.py \
  --network MK_UNet \
  --data_root ./data/ACDC_npz \
  --epochs 150 \
  --batchsize 16 \
  --img_size 224 \
  --lr 1e-4 \
  --augmentation false \
  --amp true \
  --seed 42
```

For a strict reproducibility study, run at least 3 seeds, e.g. 42, 43, 44.

## 4. Test ACDC

```bash
python test_acdc.py \
  --run_id <RUN_ID_FROM_TRAINING> \
  --network MK_UNet \
  --data_root ./data/ACDC_npz
```

Outputs:

- per-slice Excel metrics
- JSON summary
- class-index `.npy` prediction masks
- grayscale `.png` visualizations (0, 85, 170, 255)

## What I would report in the Step-2 experiment

For ACDC, report at minimum:

| Metric | RV | Myocardium | LV | Mean foreground |
|---|---:|---:|---:|---:|
| Dice | ... | ... | ... | ... |
| IoU | ... | ... | ... | ... |
| HD95 | ... | ... | ... | ... |

Then compare resource use (#Params/FLOPs) and multi-class performance against a published ACDC baseline.
