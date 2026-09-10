# ACDC Diagnosis Extension — MK-UNet + XGBoost

This folder implements the Step 2 extension from cardiac segmentation to patient-level diagnosis.

## Diagnosis classes

- `NOR`: normal
- `MINF`: previous myocardial infarction
- `DCM`: dilated cardiomyopathy
- `HCM`: hypertrophic cardiomyopathy
- `RV`: abnormal right ventricle (sometimes described as ARV)

## Two experiments

### Experiment 1 — diagnosis without MK-UNet

Use the ACDC **ground-truth ED/ES segmentation masks** to derive cardiac biomarkers, then train XGBoost.

```text
Ground-truth masks -> physiological features -> XGBoost -> diagnosis
```

This establishes an upper/reference diagnostic baseline before segmentation errors are introduced.

### Experiment 2 — diagnosis using MK-UNet

Use your trained MK-UNet checkpoint to segment ED/ES MRI volumes, derive the **same** biomarkers from predicted masks, then train/evaluate XGBoost.

```text
MRI -> MK-UNet -> predicted masks -> physiological features -> XGBoost -> diagnosis
```

A third useful analysis is included: apply the XGBoost model trained on ground-truth features directly to MK-UNet-derived features on the test set. This isolates how segmentation error changes downstream diagnosis.

## Expected location

Place this folder inside your repository:

```text
MKUnet-Multiclass/
├── mkunet_network.py
├── prepare_acdc.py
├── train_acdc.py
├── test_acdc.py
├── model_pth_acdc/
└── diagnosis/
```

The code is written for the Kaggle dataset used in your notebook:

```text
ipythonx/automated-cardiac-diagnosis
```

It searches recursively for `patientXXX`, `Info.cfg`, and ED/ES `.nii`/`.nii.gz` files, including the unusual Kaggle layout where an item ending in `.nii` can itself be a directory.

## Patient split

The split matches your ACDC notebook/code:

- train: patient001–patient080
- validation: patient081–patient100
- test: patient101–patient150

## Biomarkers/features

The same feature vector is used in both experiments:

- height, weight, body-surface area (BSA)
- LV EDV, ESV, stroke volume, EF
- RV EDV, ESV, stroke volume, EF
- myocardial ED/ES volume
- estimated myocardial mass (`ED myocardial volume x 1.05 g/mL`)
- BSA-indexed LV/RV volumes and myocardial mass
- LV/RV and MYO/LV ratios
- simple slice-wise contraction statistics for LV/RV

The slice-wise contraction statistics are research descriptors, not clinical wall-motion scores.

## Install

```bash
pip install -r diagnosis/requirements.txt
```

## 1. Extract ground-truth features

```bash
python diagnosis/extract_gt_features.py \
  --data_root /content/acdc_raw \
  --output diagnosis/results/gt_features.csv \
  --strict
```

## 2. Train XGBoost on ground-truth features

```bash
python diagnosis/train_xgboost.py \
  --features diagnosis/results/gt_features.csv \
  --output_dir diagnosis/results/xgb_ground_truth \
  --seed 7
```

Outputs include `metrics.json`, train/val/test predictions, confusion matrices, feature importance, and `xgboost_diagnosis.joblib`.

## 3. Extract MK-UNet-derived features

For your checkpoint:

```text
ACDC_MK_UNet_bs16_sz224_lr0.0001_e150_seed7_111345-best.pth
```

run:

```bash
python diagnosis/extract_mkunet_features.py \
  --data_root /content/acdc_raw \
  --checkpoint /content/MKUnet-Multiclass/model_pth_acdc/ACDC_MK_UNet_bs16_sz224_lr0.0001_e150_seed7_111345-best.pth \
  --network MK_UNet \
  --img_size 224 \
  --batch_size 8 \
  --with_gt_metrics \
  --save_masks_dir diagnosis/results/mkunet_masks \
  --output diagnosis/results/mkunet_features.csv \
  --strict
```

The preprocessing intentionally matches your existing ACDC preparation: volume-wise 1st/99th percentile clipping, mapping to `[0,1]`, resizing slices to 224, then mapping approximately to `[-1,1]` before MK-UNet inference.

## 4. Train XGBoost using MK-UNet-derived features

```bash
python diagnosis/train_xgboost.py \
  --features diagnosis/results/mkunet_features.csv \
  --output_dir diagnosis/results/xgb_mkunet \
  --seed 7
```

## 5. Recommended research comparison

Train XGBoost on perfect/GT features, but test that *same* classifier on MK-UNet-derived test features:

```bash
python diagnosis/evaluate_xgboost.py \
  --model diagnosis/results/xgb_ground_truth/xgboost_diagnosis.joblib \
  --features diagnosis/results/mkunet_features.csv \
  --split test \
  --output_dir diagnosis/results/gt_xgb_on_mkunet_test
```

Report these three rows:

| Diagnostic pipeline | Accuracy | Macro-F1 |
|---|---:|---:|
| GT masks -> features -> XGBoost | TBD | TBD |
| MK-UNet masks -> features -> XGBoost | TBD | TBD |
| GT-trained XGBoost -> MK-UNet test features | TBD | TBD |

The third row is especially useful for your research proposal because it measures **downstream diagnostic degradation caused by imperfect segmentation**.

## Important research note

This code is an experimental patient-level classification pipeline for ACDC and should be described as automated diagnostic decision support/research, not as a clinically validated diagnostic device.
