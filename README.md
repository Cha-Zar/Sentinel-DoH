# **SENTINEL-DoH 🛡️**

**Behavioral Classification of DNS over HTTPS (DoH) Traffic via Hybrid ML/DL Spatio-Temporal Analysis.**

## **🌐 Overview**

**SENTINEL-DoH** is a research-driven AI system designed to detect malicious activities (Command & Control, Data Exfiltration) hidden within encrypted **DNS over HTTPS (DoH)** tunnels.  
As encryption becomes the standard (TLS 1.3, DoH), traditional Deep Packet Inspection (DPI) is becoming obsolete. SENTINEL-DoH addresses this "blind spot" by analyzing the **spatio-temporal shape** of traffic—packet sizes and timing—rather than its content. By combining Machine Learning (XGBoost) and Deep Learning (1D-CNN), the system provides a robust, interpretable, and privacy-preserving defense mechanism.

## **🚀 Key Features**

* **Hybrid AI Architecture:** Comparison between statistical ML (Global Flow Features) and sequential DL (Packet Sequences).  
* **Privacy-First:** Operates entirely on encrypted metadata without SSL/TLS termination.  
* **Scientifically Robust:** Built-in evaluations for **Data Drift** and **Adversarial Perturbations** (Jittering/Padding).  
* **Human-Centric:** Integrated explainability using **SHAP** and **Grad-CAM** to assist SOC analysts.  
* **Ready for Reproducibility:** Fully containerized pipeline via Docker.

## **📊 Research Methodology**

### **Problem Statement**

How to robustly classify malicious DoH traffic in real-world conditions: high class imbalance, evolving temporal distributions, and active adversarial evasion?

### **Dataset**

We utilize the **CIRA-CIC-DoHBrw-2020** dataset from the Canadian Institute for Cybersecurity. It features:

* **Benign:** Standard DoH traffic from Chrome and Firefox.  
* **Malicious:** DNS Tunneling via *dns2tcp*, *DNScat*, and *Iodine*.

### **Model Comparison**

1. **XGBoost (ML Baseline):** Analyzes global flow statistics (variance, entropy, duration).  
2. **1D-CNN (Deep Learning):** Learns local patterns in the sequence of the first $N$ packets (packet size \+ inter-arrival time).

## **🛠️ Installation & Reproduction**

### **Prerequisites**

* Docker (Recommended)  
* Python 3.10+ (if running locally)

### **Quick Start (The "3-Command" Rule)**

1. **Clone the Repository:**  
   git clone \[https://github.com/Cha-Zar/Sentinel-DoH.git\]
   cd sentinel-doh

2. **Build the Environment:**  
   docker build \-t sentinel-doh .

3. **Run the Full Pipeline:**  
   docker run \--rm \-v $(pwd)/outputs:/app/outputs sentinel-doh

   *This will trigger: Data Preprocessing → Training → Robustness Testing → Figures Generation.*

## **📈 Project Structure**

├── data/               \# Raw and processed datasets  
├── notebooks/          \# Exploratory Data Analysis  
├── src/                \# Source code  
│   ├── preprocessing.py \# Feature extraction & cleaning  
│   ├── models\_ml.py     \# XGBoost implementation  
│   ├── models\_dl.py     \# 1D-CNN PyTorch/TF implementation  
│   └── evaluate.py      \# Metrics & Robustness tests  
├── paper/              \# Scientific article (Draft PDF)  
├── outputs/            \# Generated metrics, plots, and SHAP reports  
├── Dockerfile          \# Containerization script  
└── requirements.txt    \# Python dependencies

## **⚖️ Limitations & Ethics**

* **Environment:** Trained on lab-generated data; performance may vary in heterogeneous enterprise networks.  
* **Privacy:** SENTINEL-DoH complies with GDPR principles as it does not inspect PII or payload content.  
* **Biais:** The model may be sensitive to the specific tunneling tools used in the training set.

## 
