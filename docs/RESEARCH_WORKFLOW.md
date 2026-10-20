# Research workflow and missing work

## Stage 1 — Dataset provenance
Download NCT-CRC-HE-100K and CRC-VAL-HE-7K from the original publisher. Record checksums, source URL, dataset version, licensing and any documented patient/site provenance. Never upload medical image datasets to this repository.

## Stage 2 — Decoding and EDA
Run `python analysis/eda.py --data data --max-per-class 100`. The script uses Pillow to decode image pixels, verifies sampled files, computes RGB channel means and standard deviations, intensity statistics, approximate tissue coverage, image dimensions, sampled exact-pixel duplicates across splits, sampled class counts, and PCA visualization. It saves CSV, PNG and JSON reports under `results/eda`.

**Limitations:** the default sample is not the full dataset; class counts in the plots are sampled counts, not population counts. Hash equality finds exact duplicates only, not near-duplicates or patient overlap. Tissue coverage is a heuristic, not segmentation. Color statistics are not stain deconvolution.

## Stage 3 — Feature engineering
Handcrafted color/intensity features from EDA provide descriptive analysis, not the CNN's learned feature representation. The training pipeline extracts 2048-dimensional ResNet50 penultimate-layer embeddings, standardizes and reduces them to four PCA components for logistic regression and the quantum model. StandardScaler and PCA are fitted only on the selected training subset to avoid test leakage.

## Stage 4 — Modeling
Compare CNN probabilities, CNN+PCA+logistic regression, and CNN+PCA+simulated VQC on the external validation set. Use the same evaluation cohort for all models. The quantum simulator is not quantum hardware and does not establish a quantum advantage.

## Stage 5 — Planned improvements
- Full-dataset integrity scan, class distributions and near-duplicate screening
- Stain normalization experiments fitted only on training data
- Robust augmentation and ablation experiments
- CNN+SVM and gradient-boosting baselines
- Patient-level grouping if provenance permits
- Hyperparameter search on an internal validation split
- Grad-CAM and error-case galleries
- Confidence calibration, bootstrap confidence intervals and subgroup analysis
- Kaggle notebook with version-pinned environment and saved artifacts

## Reporting ethics
This repository is an **untrained research prototype** until execution logs, checkpoints and metrics are generated. No claims of clinical utility or model superiority should be made.
