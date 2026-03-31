# Sentinel-DoH Run Instructions

This file explains how to run the project in two ways:

- local Python execution (`main.py`)
- Docker / Docker Compose execution

## 1. Prerequisites

Install the following:

- Python 3.10+
- Git
- Docker Desktop (for Docker runs)

Project root in examples:

`C:\Users\user\Documents\GitHub\Sentinel-DoH`

## 2. Data Setup

Place dataset CSV files in the `data/` directory.

At minimum, this project expects Layer-2 files like:

- `l2-benign.csv`
- `l2-malicious.csv`

The loader can also search recursively under `data/`.

## 3. Run Locally (main.py)

Open PowerShell in project root.

### 3.1 Create and activate virtual environment

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

### 3.2 Install dependencies

```powershell
pip install -r requirements.txt
```

### 3.3 Run full pipeline

```powershell
python .\main.py
```

### 3.4 Optional faster runs

Skip deep learning:

```powershell
python .\main.py --skip-dl
```

Skip robustness and SHAP:

```powershell
python .\main.py --skip-robustness --skip-shap
```

Chronological split (drift-style validation):

```powershell
python .\main.py --chrono
```

## 4. Run with Docker (Dockerfile)

Make sure Docker Desktop is running.

From project root:

### 4.1 Build image

```powershell
docker build -t sentinel-doh .
```

### 4.2 Run container

```powershell
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/outputs:/app/outputs sentinel-doh
```

If `${PWD}` bind format causes issues on your shell, use absolute Windows paths:

```powershell
docker run --rm -v C:/Users/user/Documents/GitHub/Sentinel-DoH/data:/app/data -v C:/Users/user/Documents/GitHub/Sentinel-DoH/outputs:/app/outputs sentinel-doh
```

## 5. Run with Docker Compose

This project includes `docker-compose.yml`.

### 5.1 Build and run

```powershell
docker compose up --build
```

### 5.2 Run one-off command with flags

```powershell
docker compose run --rm sentinel-doh python main.py --skip-dl --skip-shap --skip-robustness
```

## 6. Expected Output Files

After successful execution, files are written to `outputs/`, such as:

- `xgboost_model.json`
- `cnn_model.pt` (if DL is enabled)
- `metrics.json`
- `confusion_matrices.png`
- `roc_pr_curves.png`
- `adversarial_results.json` (if robustness is enabled)
- `shap_summary.png` and `shap_bar.png` (if SHAP is enabled)
 i
