# Minimal ACDC Diagnosis Module — 5-Fold CV + Independent Test

This folder supports three ACDC diagnosis experiments:

1. **GT masks → biomarkers → XGBoost** — uses the ACDC expert segmentation masks.
2. **Raw ED/ES MRI → XGBoost** — no segmentation masks and no MK-UNet.
3. **MRI → MK-UNet → predicted masks → biomarkers → XGBoost** — no ACDC expert segmentation masks are used for diagnosis feature extraction or inference.

All XGBoost models use the ACDC `Group` field as the supervised diagnosis target.

## Evaluation design

- **Patients 001–100:** development cohort used for 5-fold stratified cross-validation.
- **Patients 101–150:** untouched independent final test cohort.
- After CV reporting, a final XGBoost model is trained on all 100 development patients and evaluated once on patients 101–150.

The disease label is read directly from each patient's `Info.cfg`; it is not inferred from patient number.
