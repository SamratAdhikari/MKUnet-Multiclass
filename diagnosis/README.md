# Minimal ACDC Diagnosis Module

This folder contains one implementation file: `acdc_diagnosis.py`.

It supports three patient-level diagnosis experiments on ACDC:

1. **GT masks → cardiac biomarkers → XGBoost**  
   Uses the provided ACDC expert segmentation masks. This is the upper-bound/reference experiment.

2. **Raw MRI → simple MRI features → XGBoost**  
   Uses no ground-truth segmentation masks and no MK-UNet. This is the mask-free XGBoost-only baseline.

3. **MRI → MK-UNet predicted masks → cardiac biomarkers → XGBoost**  
   Uses no ground-truth segmentation masks during diagnosis feature extraction or inference. MK-UNet itself was previously trained using segmentation ground truth.

All XGBoost experiments still use the ACDC `Group` field (`NOR`, `MINF`, `DCM`, `HCM`, `RV`) as the supervised diagnosis target during training.

The development cohort (patients 1–100) is stratified into 80 train / 20 validation patients, and patients 101–150 remain the final test cohort.
