# **🛡️ SENTINEL-DoH**

**Classification Comportementale du Trafic DNS over HTTPS (DoH) par Analyse Spatio-Temporelle Hybride ML/DL.**

## **🌐 Aperçu du Projet**

**SENTINEL-DoH** est un projet de recherche de pointe développé pour le challenge **Cyber Horizon 2.0**. Son objectif est de détecter les activités de *Command & Control* (C2) et d'exfiltration de données dissimulées dans des tunnels **DNS over HTTPS (DoH)**.  
À l'heure où le chiffrement devient la norme (**TLS 1.3**, **DoH**), l'inspection profonde de paquets (DPI) traditionnelle perd son efficacité. **SENTINEL-DoH** contourne cette limite en analysant les **métadonnées spatio-temporelles** (taille des paquets et intervalles de temps) pour identifier les menaces sans jamais déchiffrer le contenu, garantissant ainsi sécurité et confidentialité.

## **🚀 Piliers de Recherche**

* **Architecture IA Hybride :** Comparaison rigoureuse entre le Machine Learning statistique (**XGBoost**) et le Deep Learning séquentiel (**1D-CNN**).  
* **Privacy-by-Design :** Aucune terminaison SSL/TLS requise ; fonctionne exclusivement sur les métadonnées de flux chiffrés.  
* **Robustesse Adversariale :** Tests de stress intégrés pour les techniques d'évasion telles que le **Jittering** (bruit temporel) et le **Padding** (bruit de taille).  
* **IA Explicable (XAI) :** Audit des décisions via **SHAP** et **Grad-CAM** pour fournir des insights actionnables aux analystes SOC.  
* **Reproductibilité :** Conteneurisation complète via **Docker** pour garantir une réplication expérimentale en un clic.

## **📊 Méthodologie & Dataset**

### **🧠 Le Défi Scientifique**

Classifier le trafic DoH malveillant dans un environnement réaliste caractérisé par :

1. **Déséquilibre Extrême des Classes** (Bénin vs Malveillant).  
2. **Dérive Temporelle** (Évolution des patterns de trafic au fil du temps).  
3. **Évasion Adversariale** (Malwares imitant le comportement humain).

### **📂 Source des Données**

Nous utilisons le dataset **CIRA-CIC-DoHBrw-2020** :

* **Classe Bénigne :** Trafic DoH standard (Chrome, Firefox).  
* **Classe Malveillante :** Outils de tunneling DNS (*dns2tcp*, *DNScat*, *Iodine*).

## **🛠️ Installation & Reproduction**

### **1\. Cloner le Dépôt**

git clone \[https://github.com/Cha-Zar/Sentinel-DoH.git\]  
cd Sentinel-DoH

### **2\. Configuration Environnement (Docker)**

La méthode la plus fiable pour reproduire nos résultats :  
\# Construire l'image  
docker build \-t sentinel-doh .

\# Exécuter le pipeline complet (Preprocessing \-\> Training \-\> Evaluation)  
docker run \--rm \-v $(pwd)/outputs:/app/outputs sentinel-doh

### **3\. Installation Locale (Optionnel)**

pip install \-r requirements.txt  
python main.py

## **📈 Structure du Répertoire**

├── data/                \# Stockage des données (Brutes/Traitées)  
├── src/                 \# Logique Coeur  
│   ├── preprocessing.py \# Extraction de features (Spatio-Temporel)  
│   ├── models\_ml.py     \# XGBoost & Baselines Statistiques  
│   ├── models\_dl.py     \# Architecture Séquentielle 1D-CNN  
│   └── robustness.py    \# Tests de Dérive & Adversariaux  
├── paper/               \# Draft Scientifique (PDF/LaTeX)  
├── outputs/             \# Métriques, Matrices de Confusion, SHAP Plots  
├── Dockerfile           \# Environnement reproductible  
└── requirements.txt     \# Versions figées des dépendances

## **⚖️ Éthique & Limites**

* **Confidentialité :** Cet outil n'inspecte pas les données personnelles (PII) ni le contenu chiffré, conformément au **RGPD**.  
* **Intégrité :** Les performances sont rapportées via **F1-Macro** et **AUC-PR** pour éviter les biais liés au déséquilibre.  
* **Limites :** Bien que robuste, le modèle est entraîné sur des signatures de laboratoire ; les performances réelles peuvent varier selon la topologie réseau.

## 