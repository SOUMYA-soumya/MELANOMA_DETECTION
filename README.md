# Melanoma Detection 🔬
### Multimodal EfficientNet-B0 for Melanoma Classification — Image + Tabular Fusion

<p align="left">
  <img src="https://img.shields.io/badge/Task-Binary_Classification-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-SIIM--ISIC_2020-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backbone-EfficientNet--B0-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Modality-Image_+_Tabular-purple?style=for-the-badge" />
</p>

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
</p>

<a href="https://www.kaggle.com/code/soumyaranjansahoo33/melanoma-final?scriptVersionId=285656856" target="_blank">
  <img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg">
</a>

<br><br>

---

## 📌 Overview

This project presents a **multimodal deep learning pipeline** for binary melanoma classification (Benign vs. Malignant) using the **SIIM-ISIC Melanoma Classification** dataset. Rather than relying purely on image data, the model fuses dermoscopic image features extracted via **EfficientNet-B0** with structured patient metadata (age, sex, anatomical site) through a late-fusion MLP, enabling clinically richer predictions.

Melanoma is the deadliest form of skin cancer, yet it is highly treatable when detected early. Automating its detection from dermoscopic imagery can significantly reduce diagnostic delays in clinical screening workflows.

---

## 🧠 Model Architecture: MultimodalEfficientNet

The model follows a **dual-branch late-fusion design**:

```
Dermoscopic Image ──► EfficientNet-B0 ──► 128-dim Image Features ──┐
                                                                     ├──► Fusion Layer (160→64) ──► Classifier ──► Benign / Malignant
Patient Metadata  ──► MLP Tabular Branch ──► 32-dim Tab Features ──┘
```

**Image Branch — EfficientNet-B0**
- Pre-trained on ImageNet (transfer learning)
- Custom classifier head replaced with `Linear(in_features → 128)` + Dropout(0.2)
- Input: 224×224 RGB dermoscopic images

**Tabular Branch — MLP**
- Input: age (standardised), sex (encoded), anatomical site (one-hot encoded)
- Architecture: `Linear → ReLU → Dropout(0.3) → Linear → ReLU → Dropout(0.3)` → 32-dim output

**Fusion**
- Concatenation of image (128-dim) + tabular (32-dim) → 160-dim
- `Linear(160 → 64) → ReLU → Dropout(0.5) → Linear(64 → 1)` with sigmoid output

---

## 📊 Training Strategy

### Data Balancing & Augmentation

The SIIM-ISIC dataset is severely class-imbalanced (far more benign than malignant cases). The pipeline addresses this with a two-step strategy:

1. **Anatomically stratified sampling** — Benign samples drawn proportionally from each anatomical site to match the malignant count
2. **3× augmentation** — Each image in the balanced set is tripled via:
   - Original image
   - Random horizontal flip
   - Color jitter (brightness, contrast, saturation, hue)

### Training Augmentation (training set only)
- `RandomResizedCrop(224, scale=(0.8, 1.0))`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomErasing(p=0.5)` — simulates occlusion/artefacts

### Two-Phase Transfer Learning

| Phase | What trains | LR | Epochs (max) |
|---|---|---|---|
| **Phase 1 — Head only** | Classifier head + tabular branch + fusion | `1e-3` | 5 |
| **Phase 2 — Full fine-tune** | All parameters | `1e-5` | 50 |

Both phases use:
- **Optimiser:** Adam
- **Loss:** BCEWithLogitsLoss
- **Scheduler:** CosineAnnealingLR
- **Early Stopping:** patience = 5 (monitors validation loss)
- **Gradient clipping:** max norm = 1.0

### Data Split

| Split | Proportion |
|---|---|
| Train | 80% |
| Validation | 10% |
| Test (hold-out) | 10% |

Stratified by target label across all splits.

---

## 📁 Repository Structure

```
MELANOMA_DETECTION/
│
└── melanoma-final.ipynb    # Full pipeline: data balancing, augmentation,
                            # model definition, two-phase training, evaluation
```

---

## 🗂️ Dataset

**SIIM-ISIC Melanoma Classification (Kaggle 2020)**

- Dermoscopic JPEG images of skin lesions
- Binary classification: `0` = Benign, `1` = Malignant
- Patient metadata: `age_approx`, `sex`, `anatom_site_general_challenge`
- Available on [Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)

> **Note:** The dataset is not included in this repo. Download it from Kaggle and place it at `/kaggle/input/siim-isic-melanoma-classification/` (or update `DATA_PATH` in the notebook).

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn tqdm pillow
```

### Run the Notebook

```bash
jupyter notebook melanoma-final.ipynb
```

Or open directly on Kaggle via the badge above (GPU recommended).

> **Hardware:** Training was performed on a CUDA-enabled GPU. Adjust `BATCH_SIZE` if running on CPU or limited VRAM.

---

## 🔬 Key Techniques

- **Multimodal fusion** — combines image and structured patient metadata
- **Anatomically stratified balancing** — tackles severe class imbalance without naive oversampling
- **Two-phase transfer learning** — head warm-up followed by full fine-tuning at a lower LR
- **CosineAnnealingLR** — smooth learning rate decay for stable convergence
- **Early Stopping** — prevents overfitting with patience-based monitoring
- **RandomErasing** — simulates real-world image artefacts during training

---

## 👤 Author

**Soumyaranjan Sahoo**
BTech, Electronics & Telecommunication Engineering — VSSUT Burla

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/soumyaranjan--sahoo)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat-square&logo=firefox&logoColor=white)](https://explore-soumya-portfolio.vercel.app/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/SOUMYA-soumya)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/soumyaranjansahoo33)

---

## ⚠️ Disclaimer

This project is for academic and research purposes only. It is not intended for clinical use or medical diagnosis.

---

*If you find this project useful, feel free to ⭐ star the repo!*
