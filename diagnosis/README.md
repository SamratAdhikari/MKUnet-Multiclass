# ACDC Diagnosis — Full Features + Exact 5-Fold CV

This minimal diagnosis package supports three experiments:

1. **Expert GT masks → full cardiac biomarkers → XGBoost**
2. **Raw ED/ES MRI features → XGBoost** (no segmentation masks)
3. **MRI → MK-UNet predicted masks → full cardiac biomarkers → XGBoost**

## Evaluation protocol

- Patients **001–100**: development cohort.
- Exact **5-fold StratifiedKFold** on the development cohort.
- Every fold records **training and validation** Accuracy, Balanced Accuracy, and Macro-F1.
- A fresh final XGBoost model is then fit on **all 100 development patients**.
- Patients **101–150** remain untouched until the final independent test.

## Full segmentation-derived feature set

Experiments 1 and 3 use the same features:
- height, weight, BSA
- LV EDV, ESV, stroke volume, EF
- RV EDV, ESV, stroke volume, EF
- myocardial ED/ES volume and estimated myocardial mass
- BSA-indexed LV/RV volumes and myocardial mass
- LV/RV EDV ratio and MYO/LV ED ratio
- LV slice contraction mean/std/min/max
- RV slice contraction mean/std

Experiment 1 calculates these from expert ACDC masks.
Experiment 3 calculates the same features from MK-UNet predicted masks.

The XGBoost setup is kept regularized and identical across the three experiments.
