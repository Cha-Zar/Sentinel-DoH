<div align="center">

# 🛡️ SENTINEL-DoH

**Behavioural classification of DNS-over-HTTPS traffic via hybrid ML/DL spatio-temporal analysis.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![Challenge](https://img.shields.io/badge/Cyber%20Horizon-2.0-dc3545?style=flat)

</div>

---

Detecting **C2 activity and data exfiltration** hidden inside encrypted DoH tunnels — without decrypting a single byte. SENTINEL-DoH analyzes **packet sizes and inter-arrival times** to classify malicious traffic while remaining fully GDPR-compliant.

## Models

| Model | Framework | Type | Imbalance Handling |
|---|---|---|---|
| **XGBoost** | XGBoost | Statistical ML | `scale_pos_weight` + SMOTE |
| **1D-CNN** | PyTorch | Deep Learning | Weighted `BCEWithLogitsLoss` |

Explainability via **SHAP** (XGBoost) and **Grad-CAM 1D** (CNN). Adversarial stress tests with **Jittering** (IAT noise) and **Padding** (packet-size perturbation).

## Dataset

[CIRA-CIC-DoHBrw-2020](https://www.unb.ca/cic/datasets/dohbrw-2020.html) — Benign (Chrome/Firefox) vs. Malicious (dns2tcp, DNScat2, Iodine).

Download the CSV files and place them in `data/`.

## Quickstart

```bash
git clone https://github.com/Cha-Zar/Sentinel-DoH.git
cd Sentinel-DoH

# Docker (recommended)
docker build -t sentinel-doh .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs sentinel-doh

# Or locally
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python main.py
```

### CLI Options

| Flag | Description |
|---|---|
| `--data-dir PATH` | Override data directory (default: `data/`) |
| `--chrono` | Use chronological split instead of stratified |
| `--no-smote` | Disable SMOTE oversampling |
| `--skip-dl` | Skip CNN training (XGBoost only) |
| `--skip-robustness` | Skip adversarial perturbation tests |
| `--skip-shap` | Skip SHAP interpretability (saves time) |

## Project Structure

```
├── main.py                 # Pipeline orchestrator (CLI entry point)
├── src/
│   ├── config.py           # Central configuration & hyperparameters
│   ├── utils.py            # Logging, seed reproducibility
│   ├── preprocessing.py    # Data loading, cleaning, splitting, SMOTE
│   ├── models_ml.py        # XGBoost classifier
│   ├── models_dl.py        # 1D-CNN (PyTorch)
│   ├── evaluation.py       # Metrics, confusion matrices, ROC/PR curves
│   ├── robustness.py       # Adversarial jitter & padding tests
│   └── interpretability.py # SHAP beeswarm + Grad-CAM 1D heatmaps
├── data/                   # Place CIRA-CIC-DoHBrw-2020 CSVs here
├── outputs/                # Generated models, plots, metrics
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Outputs

After a full run, `outputs/` will contain:

- `xgboost_model.json` / `cnn_model.pt` — Trained models
- `metrics.json` — All evaluation metrics
- `confusion_matrices.png` — Side-by-side confusion matrices
- `roc_pr_curves.png` — ROC and Precision-Recall curves
- `confidence_dist_*.png` — Prediction confidence distributions
- `cnn_training_history.png` — Loss/AUC curves over epochs
- `robustness_summary.png` — F1 degradation under adversarial noise
- `adversarial_results.json` — Detailed robustness metrics
- `shap_beeswarm.png` — SHAP feature importance (if not skipped)
- `gradcam_1d.png` — Grad-CAM attention heatmaps
