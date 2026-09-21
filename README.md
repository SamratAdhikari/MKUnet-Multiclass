# MK-UNet for Multiclass Cardiac Segmentation and Diagnosis

[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/SamratAdhikari/MKUnet-Multiclass)

This repository provides a PyTorch implementation of the MK-UNet architecture for multiclass medical image segmentation. The project includes a complete pipeline for training, testing, and evaluating the model on the ACDC (Automated Cardiac Diagnosis Challenge) dataset for segmenting the right ventricle (RV), myocardium (MYO), and left ventricle (LV).

A key feature of this repository is the downstream cardiac diagnosis module. This module uses the segmentation masks (either ground truth or model-predicted) to extract a comprehensive set of cardiac biomarkers and then trains a classifier to diagnose cardiac conditions, demonstrating a full-fledged clinical application.

## Key Features

- **MK-UNet Architecture**: Includes the implementation of MK-UNet and its variants (`T`, `S`, `M`, `L`) with multi-kernel inverted residual blocks and attention mechanisms.
- **Multiclass & Deep Supervision**: The model is configured for 4-class segmentation (Background, RV, MYO, LV) and utilizes deep supervision for improved training stability and performance.
- **End-to-End Pipeline**: Provides scripts for data preprocessing, training, and testing on the ACDC dataset.
- **Cardiac Diagnosis Module**: A complete pipeline to:
    1.  Extract biomarkers from segmentation masks (e.g., volumes, ejection fraction, wall thickness).
    2.  Train an XGBoost classifier for disease diagnosis based on these biomarkers.
    3.  Compare diagnosis performance using ground-truth masks vs. model-predicted masks.
- **Pre-trained Model**: Includes a pre-trained MK-UNet model for immediate inference and evaluation.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/SamratAdhikari/MKUnet-Multiclass.git
    cd MKUnet-Multiclass
    ```

2.  Install the required packages. The project has two separate requirements files: one for the core segmentation pipeline and one for the diagnosis module.

    ```bash
    # For the segmentation pipeline
    pip install -r requirements.txt

    # For the diagnosis module
    pip install -r diagnosis/requirements.txt
    ```

## Usage

The pipeline is structured into four main steps: data preparation, model training, testing, and cardiac diagnosis.

### Step 1: Data Preparation

This script processes the raw ACDC dataset (NifTi format) into 2D slices stored in `.npz` format for efficient loading.

1.  Download the ACDC dataset from the official challenge website.
2.  Run the `prepare_acdc.py` script, pointing to your raw data directory.

    ```bash
    python prepare_acdc.py --raw_root /path/to/your/acdc_dataset --out_root ./data/ACDC_npz
    ```

    This will create a directory `./data/ACDC_npz` with `train`, `val`, and `test` subfolders containing the processed slices.

### Step 2: Training

The `train_acdc.py` script handles the model training process. It supports multi-scale training, mixed-precision (AMP), and saves the best-performing model based on validation Dice score.

```bash
python train_acdc.py \
    --network MK_UNet \
    --data_root ./data/ACDC_npz \
    --epochs 150 \
    --batchsize 16 \
    --lr 1e-4 \
    --seed 42
```

A new directory will be created under `model_pth_acdc/` containing the `best` and `last` model checkpoints (`.pth`), along with `history.json` and `summary.json` files detailing the training process and final test metrics.

### Step 3: Testing

To evaluate a trained model, use the `test_acdc.py` script. You need to provide the `run_id`, which is the name of the folder containing the trained model checkpoint.

To test the provided pre-trained model:

```bash
python test_acdc.py \
    --run_id ACDC_MK_UNet_bs16_sz224_lr0.0001_e150_seed7_111345 \
    --network MK_UNet \
    --data_root ./data/ACDC_npz
```

This will:

- Generate segmentation predictions in the `predictions_acdc/<run_id>` directory.
- Create a per-slice performance report `results_acdc/ACDC_<run_id>_per_slice.xlsx`.
- Save a summary of aggregate metrics (Dice, IoU, HD95) to `results_acdc/ACDC_<run_id>_summary.json`.

### Step 4: Cardiac Diagnosis Pipeline

The `diagnosis` directory contains a powerful module for performing cardiac disease classification using features derived from the segmentations. It allows for a direct comparison between using expert-level ground truth masks and the automatically generated MK-UNet masks.

The core logic is in `diagnosis/acdc_diagnosis.py`, which provides functions for three main experiments:

1.  **Expert GT masks → biomarkers → XGBoost**
2.  **Raw MRI features → XGBoost (no segmentation)**
3.  **MRI → MK-UNet predicted masks → biomarkers → XGBoost**

A typical workflow involves calling the functions within this script to extract features and then run the cross-validation and testing protocol. The module produces detailed reports, confusion matrices, and feature importance plots to analyze the diagnostic power of the segmentation-derived biomarkers. For more details, please refer to the `diagnosis/README.md` file.

## Model Architecture

The MK-UNet is a U-shaped encoder-decoder network designed for efficient and accurate segmentation. Its key components include:

- **Multi-Kernel Inverted Residual Blocks (MK-IRB)**: The main building block of the network, which uses depthwise convolutions with multiple kernel sizes (`[1, 3, 5]`) to capture features at different scales.
- **Attention Mechanisms**: Integrates Channel Attention, Spatial Attention, and Grouped Attention Gates (in the skip connections) to help the model focus on relevant features and regions.
- **Deep Supervision**: The model has multiple output heads at different decoder stages, which are combined during training. This helps with gradient flow and leads to better performance.

The repository supports several model sizes (`MK_UNet_T`, `MK_UNet_S`, `MK_UNet`, `MK_UNet_M`, `MK_UNet_L`) by scaling the number of channels in each block.
