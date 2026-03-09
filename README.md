<div align="center">

# 🛡️ SENTINEL-DoH

**Behavioural classification of DNS over HTTPS traffic via hybrid ML/DL spatio-temporal analysis.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![Challenge](https://img.shields.io/badge/Cyber%20Horizon-2.0-dc3545?style=flat)

</div>

---

Detecting **C2 activity and data exfiltration** hidden inside encrypted DoH tunnels — without decrypting a single byte. SENTINEL-DoH analyzes **packet sizes and inter-arrival times** to classify malicious traffic while remaining fully GDPR-compliant.

## Models

| Model | Type | Imbalance Handling |
|---|---|---|
| **XGBoost** | Statistical ML | `scale_pos_weight` + SMOTE |
| **1D-CNN** | Deep Learning | Weighted Binary Cross-Entropy |

Explainability via **SHAP** (XGBoost) and **Grad-CAM** (CNN). Adversarial stress tests with **Jittering** and **Padding**.

## Dataset

[CIRA-CIC-DoHBrw-2020](https://www.unb.ca/cic/datasets/dohbrw-2020.html) — Benign (Chrome/Firefox) vs. Malicious (dns2tcp, DNScat, Iodine).

## Quickstart

```bash
git clone https://github.com/Cha-Zar/Sentinel-DoH.git
cd Sentinel-DoH

# Docker (recommended)
docker build -t sentinel-doh .
docker run --rm -v $(pwd)/outputs:/app/outputs sentinel-doh

# Or locally
pip install -r requirements.txt && python main.py
```

## Structure

```
├── src/
│   ├── preprocessing.py   # Feature extraction
│   ├── models_ml.py       # XGBoost
│   ├── models_dl.py       # 1D-CNN
│   └── robustness.py      # Adversarial tests
├── outputs/               # Metrics, SHAP plots
└── Dockerfile
```
