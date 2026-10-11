# Alzheimer MRI Classification 4 Class - MambaVision Tiny

This package runs as a normal Python project on a Linux workstation. It has no notebook magics, no TensorFlow pipeline, and no Google Colab paths. Each source file handles one part of the workflow.

## Folder structure

```text
alzheimer_mambavision_project/
├── README.md                         # Setup, data layout and run instructions
├── requirements.txt                  # General Python dependencies
├── install_mamba.sh                  # CUDA 13 / PyTorch 2.11 binary wheels
├── train.py                          # CLI entry point; connects the project blocks
└── mri_mamba/
    ├── __init__.py
    ├── config.py                     # Class labels, image extensions, normalization
    ├── data/
    │   ├── __init__.py
    │   ├── archive.py                # Safe ZIP extraction
    │   ├── discovery.py              # Locate the four class folders
    │   ├── dedup.py                  # Scan files, SHA-256 and conflict removal
    │   ├── split.py                  # Grouped train/validation/test split + manifest
    │   ├── transforms.py             # Train and evaluation image transforms
    │   └── loaders.py                # ImageFolder datasets and DataLoaders
    ├── models/
    │   ├── __init__.py
    │   ├── attention.py              # ECA channel attention
    │   ├── local_branch.py           # CNN branch for local MRI details
    │   ├── fusion.py                 # Gated local/global feature fusion
    │   └── classifier.py             # MambaVision-Tiny classifier assembly
    ├── engine/
    │   ├── __init__.py
    │   ├── losses.py                 # Class-weighted cross entropy
    │   ├── metrics.py                # Accuracy and macro-F1
    │   ├── train_epoch.py            # One train or validation epoch + tqdm
    │   ├── checkpoint.py             # Save/load best checkpoint
    │   ├── trainer.py                # Epoch loop, early stopping and history
    │   └── evaluator.py              # Test report and confusion matrix
    └── utils/
        ├── __init__.py
        └── plots.py                  # Training curves
```

## How the code is organized

1. `train.py` reads command-line options and orchestrates each stage.
2. `data/discovery.py` finds the dataset root. `data/archive.py` handles ZIP extraction.
3. `data/dedup.py` scans the four class folders, computes exact-file hashes and discards conflicting labels.
4. `data/split.py` groups byte-identical images so they cannot cross train/validation/test; it writes `split_manifest.csv`.
5. `data/transforms.py` and `data/loaders.py` prepare grayscale MRI as three channels for the pretrained backbone.
6. `models/` contains the MambaVision global encoder, local CNN branch, gated fusion, ECA attention and classifier head.
7. `engine/` contains weighted loss, metrics, epoch execution, checkpointing, early stopping and final test evaluation.
8. `utils/plots.py` saves training curves.

The model follows the useful local/global-feature and attention ideas from FusionMamba, adapted to classify one MRI image. The paper's two-modality fusion decoder and image-reconstruction losses are not used here.

## Environment and installation

The supplied runtime is Python 3.13.15, PyTorch 2.11.0+cu130 and CUDA 13.0 on a Tesla T4. First install PyTorch and torchvision as a matching CUDA pair for the workstation. Then, from this project folder, run:

```bash
bash install_mamba.sh
python -m pip install -r requirements.txt
```

The script pins `mamba-ssm==2.3.2.post1+cu.13.0.torch.2.11` and `causal-conv1d==1.6.2.post1+cu.13.0.torch.2.11` from the Astral CUDA 13.0 index, and installs `mambavision==1.2.0` without allowing it to replace the pinned Mamba wheel. `--only-binary=:all:` prevents a source build. These provided CUDA extension wheels target Linux; use a Linux workstation environment for this install script.

The package imports are separate by design:

```python
import mamba_ssm
from mambavision import create_model
```

## Dataset layout

Pass either the ZIP file or a directory containing these folders (the code also searches nested folders):

```text
dataset/
├── MildDemented/
├── ModerateDemented/
├── NonDemented/
└── VeryMildDemented/
```

SHA-256 grouping catches byte-identical files. This does not guarantee patient-level separation; if patient IDs exist, use them as groups to prevent scans from one patient appearing in multiple splits. The resulting split is approximately 70/15/15; exact percentages may shift because splitting keeps whole groups together.

## Run training

From this project folder:

```bash
python train.py --data-root "/path/to/archive.zip"
```

Or pass the extracted folder:

```bash
python train.py --data-root "/path/to/dataset" --output-dir "/path/to/mri_output"
```

Optional settings:

```bash
python train.py --data-root "/path/to/archive.zip" --batch-size 8 --epochs 30 --learning-rate 2e-5 --patience 6 --num-workers 0
```

The default `--num-workers 0` is safe for Windows and works on Linux. The output folder contains extracted data (for ZIP input), split folders, `split_manifest.csv`, the best checkpoint, training curves and the test confusion matrix.

## Notes

- The code has been organized and syntax-checked; model training has not been run in this environment.
- If the class folders have fewer than four unique image groups in any class, grouped splitting stops with a clear error.
- The local and global branch names describe model functions; this is an adaptation for single-image classification, not a reproduction of FusionMamba's multimodal image-fusion task.
