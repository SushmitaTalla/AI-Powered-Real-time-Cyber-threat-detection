# Network Intrusion Detection System (NIDS)
### Hybrid Anomaly Detection & Multi-Class Attack Classification on CICIDS2017

<br>

> **2.83M records → 98.24% classification accuracy** — a production-grade, two-phase ML pipeline that detects and classifies 14 network attack types in real time, with CTGAN-driven class balancing and live concept drift monitoring.

<br>

---

## Why This Project Exists

Network intrusion datasets are notoriously imbalanced. Rare attacks like Heartbleed and SQL Injection represent < 0.01% of real-world traffic — meaning naïve classifiers simply ignore them. This project solves that with a **two-phase hybrid architecture**: an unsupervised autoencoder gates all traffic first, and a Random Forest only activates on flagged flows, achieving high sensitivity without sacrificing classification precision.

---

## Results at a Glance

| Phase | Task | Accuracy | Recall | F1-Score |
|---|---|---|---|---|
| Phase 1 | Binary anomaly detection | 59% | 72.83% | 0.41 |
| Phase 2 | 14-class attack classification | **98.24%** | 94.00% (macro) | **0.87** (macro) |

> Phase 1 is intentionally aggressive — a 55th-percentile reconstruction-error threshold maximises sensitivity. Phase 2 then handles precision at scale.

---

## Architecture

```
Raw Traffic (79 features)
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  MinMaxScaler (fit on benign only) · PCA → 95% variance
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Phase 1          │  Autoencoder · 0.5 encoding ratio · sigmoid output
│  Anomaly Filter   │  Threshold @ 55th percentile of benign reconstruction error
└────────┬──────────┘
         │ flagged anomalies
         ▼
┌───────────────────┐
│  Phase 2          │  Random Forest · balanced class weights
│  Classifier       │  14 attack classes · 98.24% accuracy
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Drift Monitor    │  ADWIN (River) · streaming concept drift detection
└───────────────────┘
```

---

## Dataset & Class Balancing

**Source:** CICIDS2017 — 8 CSV files, ~2.83M rows, 79 features, 15 labels (1 BENIGN + 14 attack types).

The raw dataset is severely imbalanced. A three-tier augmentation strategy was applied:

| Tier | Condition | Method | Target |
|---|---|---|---|
| A | Class ≥ 5,000 samples | CTGAN conditional generation | 230,124 |
| B | Class < 5,000 samples | Duplication | 230,124 |
| C | Rare classes (Heartbleed, SQLi, Infiltration, XSS) | Capped to avoid synthetic noise | 10,000 |

**CTGAN training:** 100 epochs · L4 GPU · ~4 hours · saved to Drive  
**Final balanced dataset:** 2.57M rows · 15 classes

> **Data integrity note:** The MinMaxScaler was fitted exclusively on benign training samples — no attack-pattern leakage into the feature space. The train/test split precedes all augmentation.

---

## Technical Stack

| Layer | Technology |
|---|---|
| Data processing | Python · Pandas · NumPy |
| Dimensionality reduction | Scikit-learn PCA (95% variance retained) |
| Anomaly detector | TensorFlow/Keras Autoencoder |
| Classifier | Scikit-learn Random Forest (balanced class weights) |
| Synthetic data | CTGAN (SDV library) |
| Drift detection | River · ADWIN algorithm |
| Compute | Google Colab · NVIDIA L4 GPU |

---

## Repository Structure

```
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # Merge, clean, dtype optimisation
│   ├── 02_ctgan_training.ipynb         # CTGAN training on GPU
│   ├── 03_synthetic_sampling.ipynb     # Two-tier balancing strategy
│   ├── 04_model_training.ipynb         # Autoencoder + Random Forest
│   └── 05_evaluation.ipynb            # Metrics, confusion matrices, drift
│
├── src/
│   ├── inference.py                    # Production inference pipeline
│   ├── drift_monitor.py                # ADWIN streaming monitor
│   └── preprocessing.py               # Feature engineering & scaling
│
├── artifacts/
│   ├── autoencoder.h5                  # Trained anomaly detector
│   ├── random_forest.pkl               # Trained classifier
│   ├── scaler.pkl                      # MinMaxScaler (benign-fitted)
│   └── pca_transformer.pkl             # PCA model
│
└── config.yaml                         # Centralised hyperparameters & paths
```

---

## Reproducing the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks in order (01 → 05)
#    or use the end-to-end inference script directly

# 3. Run inference on new traffic data
python src/inference.py --input data/new_traffic.csv --output results/predictions.csv

# 4. Start drift monitor
python src/drift_monitor.py --stream data/live_feed.csv
```

---

## Design Decisions & Trade-offs

**Why an autoencoder as the anomaly filter?**  
Autoencoders learn the compressed representation of *normal* traffic. Anything that reconstructs poorly is, by definition, anomalous — no labelled attack data required for Phase 1. This makes the system extensible to zero-day attacks.

**Why the 55th percentile threshold?**  
Tuned to maximise recall at the cost of precision in Phase 1. False positives are cheap (they just trigger Phase 2); false negatives (missed attacks) are not. The Random Forest absorbs the false positive cost.

**Why CTGAN over SMOTE for oversampling?**  
SMOTE interpolates between existing samples in feature space — adequate for smooth distributions but brittle for network flows, which have discrete ports, protocol flags, and multi-modal rate features. CTGAN learns the conditional joint distribution and generates structurally plausible flows.

**Why cap rare classes at 10,000?**  
With < 50 real Heartbleed samples in CICIDS2017, CTGAN is extrapolating from noise beyond a certain volume. Capping limits the injection of low-fidelity synthetic signal that could otherwise mislead the classifier.

---

## Known Limitations

- Phase 1 binary accuracy (59%) reflects a deliberate sensitivity-precision trade-off, not a model failure.
- Rare capped classes (Heartbleed, XSS, SQLi, Infiltration) may show inflated recall from duplication; per-class confusion matrix analysis is recommended before production deployment.
- CICIDS2017 is a lab-generated dataset — real-world traffic may exhibit distribution shift that ADWIN should flag within a live deployment window.

---

## Skills Demonstrated

`Unsupervised Learning` · `Supervised Classification` · `Generative Models (GAN)` · `Imbalanced Data Handling` · `Streaming ML` · `Feature Engineering` · `Dimensionality Reduction` · `Neural Networks` · `Ensemble Methods` · `MLOps / Production Pipelines`
