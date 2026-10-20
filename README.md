# Colorectal Cancer Histopathology Classification — Hybrid CNN + Quantum ML

Research prototype for **tumor vs. non-tumor classification** using NCT-CRC-HE-100K and independent CRC-VAL-HE-7K validation.

## Status
**Implementation only. Models have not been trained or validated.** No accuracy claims are made. Not for clinical use.

## Experiments
1. Fine-tune ImageNet-pretrained ResNet50 for binary classification (TUM vs. eight other tissue categories).
2. Extract 2048-dimensional CNN features and reduce to four PCA components.
3. Compare logistic regression against a simulated four-qubit variational quantum classifier (PennyLane).
4. Evaluate CNN, PCA+logistic regression and PCA+VQC on CRC-VAL-HE-7K, with ROC curves, confusion matrices and a Grad-CAM example.

## Datasets
Obtain NCT-CRC-HE-100K and CRC-VAL-HE-7K from the original dataset provider (Zenodo record 1214456). Do **not** commit dataset images or model checkpoints.

Expected structure:
```
data/
  NCT-CRC-HE-100K/
    ADI/ BACK/ DEB/ LYM/ MUC/ MUS/ NORM/ STR/ TUM/
  CRC-VAL-HE-7K/
    ADI/ BACK/ DEB/ LYM/ MUC/ MUS/ NORM/ STR/ TUM/
```

## Setup
```bash
pip install -r requirements.txt
python pipeline.py --data data --output results --epochs 8
```

For a quick smoke test, use `--max-train 200 --max-external 100 --epochs 1 --quantum-train-limit 40 --quantum-epochs 1`.

**Important:** Pretrained ResNet50 weights require an initial download. Quantum simulation on thousands of images can be very slow. Use a GPU for CNN training and a small balanced subset for VQC training. The provided script currently samples the quantum subset randomly rather than stratifying it; improve this before interpreting results.

## Research limitations
- Patch-level labels are not equivalent to patient-level diagnosis.
- Potential stain/scanner/site shift must be assessed.
- Independent validation does not guarantee patient-level separation without dataset provenance checks.
- Quantum advantage is **not established**; compare accuracy, runtime and uncertainty against strong classical baselines.
- Binary TUM vs. other-tissue classification does not distinguish all colorectal cancer subtypes.

## Files
- `pipeline.py`: training and evaluation implementation
- `requirements.txt`: Python dependencies
- `.gitignore`: excludes local data, model weights and outputs

## Attribution
Dataset: Kather et al., NCT-CRC-HE-100K / CRC-VAL-HE-7K. Follow the dataset's citation and license terms.

Research prototype by Md Nazmussakib.