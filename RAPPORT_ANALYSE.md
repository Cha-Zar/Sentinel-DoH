# Rapport d'Analyse — SENTINEL-DoH

## Classification Comportementale du Trafic DNS-over-HTTPS

**Date :** 9 mars 2026  
**Projet :** SENTINEL-DoH — Cyber Horizon 2.0  
**Pipeline :** XGBoost (ML) + 1D-CNN (DL) avec tests adversariaux  
**Durée d'exécution :** ~14 minutes (CPU)

---

## Table des Matières

1. [Contexte et Objectifs](#1-contexte-et-objectifs)
2. [Jeu de Données](#2-jeu-de-données)
3. [Prétraitement](#3-prétraitement)
4. [Résultats — XGBoost](#4-résultats--xgboost)
5. [Résultats — 1D-CNN](#5-résultats--1d-cnn)
6. [Analyse Comparative](#6-analyse-comparative)
7. [Robustesse Adversariale](#7-robustesse-adversariale)
8. [Interprétabilité](#8-interprétabilité)
9. [Limites et Biais Potentiels](#9-limites-et-biais-potentiels)
10. [Conclusions et Recommandations](#10-conclusions-et-recommandations)

---

## 1. Contexte et Objectifs

Le protocole **DNS-over-HTTPS (DoH)** chiffre les requêtes DNS dans un tunnel HTTPS (port 443), empêchant toute inspection du contenu par les pare-feux traditionnels. Si ce mécanisme protège la vie privée des utilisateurs légitimes, il est aussi exploité par des acteurs malveillants pour :

- **Exfiltration de données** : encoder des données volées dans des flux DoH
- **Commande & Contrôle (C2)** : communiquer avec des serveurs de contrôle via des tunnels DoH (dns2tcp, DNScat2, Iodine)

**Objectif de SENTINEL-DoH** : classifier le trafic DoH comme **Bénin** (navigation normale) ou **Malveillant** (tunneling C2/exfiltration) sans déchiffrer aucun paquet, en se basant uniquement sur les **caractéristiques statistiques des flux** (tailles de paquets, temps inter-arrivées).

---

## 2. Jeu de Données

### Source
**CIRA-CIC-DoHBrw-2020** — publié par le Canadian Institute for Cybersecurity (UNB).

### Composition

| Classe | Échantillons | Pourcentage | Sources |
|---|---|---|---|
| **Bénin (0)** | 19 807 | 7,3% | Chrome, Firefox (navigation web normale) |
| **Malveillant (1)** | 249 836 | 92,7% | dns2tcp, DNScat2, Iodine (tunneling) |
| **Total** | **269 643** | 100% | — |

### Déséquilibre des classes

Le jeu de données est **fortement déséquilibré** avec un ratio de ~1:12,6 (bénin/malveillant). Ce déséquilibre est une caractéristique réaliste du trafic réseau (les attaques sont minoritaires dans la réalité, mais ici le jeu de données sur-représente le malveillant car il contient les captures de 3 outils de tunneling).

### Features (29 caractéristiques numériques)

| Groupe | Nb | Exemples |
|---|---|---|
| **Flow** | 5 | Duration, FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate |
| **PacketLength** | 8 | Variance, StdDev, Mean, Median, Mode, SkewFromMedian, SkewFromMode, CoefficientOfVariation |
| **PacketTime (IAT)** | 8 | Variance, StdDev, Mean, Median, Mode, SkewFromMedian, SkewFromMode, CoefficientOfVariation |
| **ResponseTime** | 8 | Variance, StdDev, Mean, Median, Mode, SkewFromMedian, SkewFromMode, CoefficientOfVariation |

Ces features sont extraites par **DoHLyzer** au niveau du flux (flow-level), ce qui garantit la conformité RGPD (aucune donnée personnelle ni contenu des requêtes DNS).

---

## 3. Prétraitement

### Pipeline de prétraitement

1. **Chargement** : Concaténation des fichiers `l2-benign.csv` et `l2-malicious.csv`
2. **Nettoyage** : Suppression des colonnes métadonnées (IP, ports), remplacement des `Inf` par `NaN`, imputation par la médiane
3. **Découpage** : Split stratifié 70/10/20 (train/validation/test)
4. **Standardisation** : `StandardScaler` ajusté sur le train uniquement
5. **SMOTE** : Sur-échantillonnage synthétique de la classe minoritaire (train uniquement)

### Distribution après split et SMOTE

| Ensemble | Total | Bénin (0) | Malveillant (1) |
|---|---|---|---|
| **Train** (avant SMOTE) | 188 749 | 13 865 (7,3%) | 174 884 (92,7%) |
| **Train** (après SMOTE) | 349 768 | 174 884 (50%) | 174 884 (50%) |
| **Validation** | 26 965 | 1 981 (7,3%) | 24 984 (92,7%) |
| **Test** | 53 929 | 3 961 (7,3%) | 49 968 (92,7%) |

> **Note importante** : Le SMOTE est appliqué **uniquement** sur l'ensemble d'entraînement pour éviter toute fuite de données. La validation et le test restent avec la distribution originale.

---

## 4. Résultats — XGBoost

### Configuration
- 300 estimateurs, profondeur maximale = 6
- `scale_pos_weight` = 1.00 (après SMOTE, les classes sont équilibrées)
- Temps d'entraînement : ~4 secondes

### Métriques sur le jeu de test

| Métrique | Valeur |
|---|---|
| **F1-macro** | **0.9997** |
| **AUC-ROC** | **1.0000** |
| **AUC-PR** | **1.0000** |
| Précision (Bénin) | 1.00 |
| Rappel (Bénin) | 1.00 |
| Précision (Malveillant) | 1.00 |
| Rappel (Malveillant) | 1.00 |
| **Accuracy** | **1.00** |

### Top-5 Features par importance (gain)

| Rang | Feature | Gain |
|---|---|---|
| 1 | **PacketTimeStandardDeviation** | 0.2194 |
| 2 | **PacketLengthMode** | 0.1939 |
| 3 | **Duration** | 0.0974 |
| 4 | **PacketTimeVariance** | 0.0876 |
| 5 | **PacketLengthMean** | 0.0832 |

> **Observation** : Les features temporelles (PacketTime) et spatiales (PacketLength) dominent, confirmant l'hypothèse que le trafic DoH malveillant est distinguable par son profil spatio-temporel.

### Analyse de surconfiance
- **Faux Positifs à haute confiance (>95%)** : 0 sur 1 FP total
- **Faux Négatifs à haute confiance (>95%)** : 1 sur 3 FN total

Le modèle XGBoost ne commet quasiment aucune erreur et n'est pas surconfiant dans ses rares erreurs.

---

## 5. Résultats — 1D-CNN

### Architecture
```
Conv1d(1→64, k=3, padding=1) → BatchNorm → ReLU → MaxPool(2)
→ Conv1d(64→128, k=3, padding=1) → GlobalAvgPool
→ Dense(128→64) → ReLU → Dropout(0.3) → Dense(64→1, sigmoid)
```

### Configuration
- Optimiseur : Adam (lr=1e-3, ReduceLROnPlateau)
- Loss : BCEWithLogitsLoss (poids de classes équilibrés après SMOTE)
- Epochs : 50 (pas d'early stopping atteint)
- Batch size : 256
- Temps d'entraînement : ~14 minutes (CPU)

### Métriques sur le jeu de test

| Métrique | Valeur |
|---|---|
| **F1-macro** | **0.9720** |
| **AUC-ROC** | **0.9991** |
| **AUC-PR** | **0.9999** |
| Précision (Bénin) | 0.91 |
| Rappel (Bénin) | 0.99 |
| Précision (Malveillant) | 1.00 |
| Rappel (Malveillant) | 0.99 |
| **Accuracy** | **0.99** |

### Courbe d'apprentissage
- Le modèle converge progressivement sur 50 epochs
- La loss d'entraînement diminue de 0.2934 → 0.0258
- La loss de validation suit une tendance décroissante : 0.1469 → 0.0237
- Le val_auc progresse de 0.9706 → 0.9997
- Le learning rate est réduit automatiquement : 1e-3 → 7.81e-6

### Analyse de surconfiance
- **Faux Positifs à haute confiance (>95%)** : 6 sur 28 FP total
- **Faux Négatifs à haute confiance (>95%)** : 106 sur 401 FN total

Le CNN commet plus d'erreurs que XGBoost et ~26% de ses FN sont à haute confiance, ce qui est un point d'attention.

---

## 6. Analyse Comparative

### Tableau récapitulatif

| Métrique | XGBoost | 1D-CNN | Écart |
|---|---|---|---|
| F1-macro | **0.9997** | 0.9720 | +0.0277 |
| AUC-ROC | **1.0000** | 0.9991 | +0.0009 |
| AUC-PR | **1.0000** | 0.9999 | +0.0001 |
| Précision macro | **0.9996** | 0.9535 | +0.0461 |
| Rappel macro | **0.9998** | 0.9925 | +0.0073 |
| Faux Positifs | **1** | 28 | ×28 |
| Faux Négatifs | **3** | 401 | ×134 |
| Temps d'entraînement | **~4s** | ~14 min | ×210 |

### Interprétation

1. **XGBoost domine nettement** sur toutes les métriques de classification. Son F1-macro de 0.9997 est quasi-parfait.

2. **Le 1D-CNN atteint d'excellentes performances** (F1=0.972, AUC-ROC=0.999) mais reste en retrait par rapport à XGBoost. Cet écart s'explique par :
   - Les features sont déjà des statistiques agrégées (29 valeurs), ce qui est un espace de faible dimensionnalité mieux adapté aux arbres de décision
   - Un CNN est plus pertinent sur des séquences brutes (time series de paquets), pas sur des statistiques pré-calculées
   - Le SMOTE génère des échantillons synthétiques linéairement interpolés, ce qui avantage les modèles à frontière de décision angulaire (arbres) vs. les réseaux de neurones

3. **Rapport coût-bénéfice** : XGBoost est ~210× plus rapide à entraîner pour de meilleurs résultats. Dans un déploiement SOC temps réel, XGBoost est clairement préférable.

---

## 7. Robustesse Adversariale

### Protocole

Deux types de perturbations simulent un attaquant qui tente d'échapper à la détection :

- **Jitter (bruit gaussien)** : ajout de bruit σ = {5%, 10%, 20%} sur les features temporelles (PacketTime*), simulant un attaquant qui randomise ses délais inter-paquets
- **Padding (bourrage)** : perturbation de ±10% sur les features de taille de paquets (PacketLength*), simulant un attaquant qui ajoute du padding aléatoire

### Résultats détaillés

#### XGBoost — Robustesse au Jitter

| Perturbation | F1-macro | AUC-ROC | Δ F1 vs Clean |
|---|---|---|---|
| Clean | 0.9997 | 1.0000 | — |
| Jitter σ=5% | 0.9987 | 1.0000 | **-0.0010** |
| Jitter σ=10% | 0.9990 | 1.0000 | **-0.0007** |
| Jitter σ=20% | 0.9990 | 1.0000 | **-0.0007** |

> **XGBoost est extrêmement résistant au jitter temporel.** Même avec 20% de bruit sur les IAT, le F1 ne chute que de 0.07 points de pourcentage. L'AUC-ROC reste à 1.0000.

#### XGBoost — Robustesse au Padding

| Perturbation | F1-macro | AUC-ROC | Δ F1 vs Clean |
|---|---|---|---|
| Clean | 0.9997 | 1.0000 | — |
| Padding ±10% | 0.4587 | 0.9598 | **-0.5410** |

> **⚠️ Le padding est dévastateur pour XGBoost.** Le F1 chute de 99.97% à 45.87%. Le rappel de la classe malveillante tombe à 0.508 (la moitié des attaques passent inaperçues). Cela révèle une **forte dépendance aux features de taille de paquets** — un attaquant ajoutant du padding aléatoire pourrait contourner ce modèle.

#### 1D-CNN — Robustesse au Jitter

| Perturbation | F1-macro | AUC-ROC | Δ F1 vs Clean |
|---|---|---|---|
| Clean | 0.9720 | 0.9991 | — |
| Jitter σ=5% | 0.9652 | 0.9988 | **-0.0068** |
| Jitter σ=10% | 0.9431 | 0.9981 | **-0.0289** |
| Jitter σ=20% | 0.8511 | 0.9918 | **-0.1209** |

> Le CNN est **plus sensible au jitter** que XGBoost. À σ=20%, le F1 chute de 12 points. Cela dit, le modèle reste utilisable (F1=0.85) et l'AUC-ROC se maintient à 0.992.

#### 1D-CNN — Robustesse au Padding

| Perturbation | F1-macro | AUC-ROC | Δ F1 vs Clean |
|---|---|---|---|
| Clean | 0.9720 | 0.9991 | — |
| Padding ±10% | 0.9682 | 0.9991 | **-0.0038** |

> **Le CNN est très résistant au padding** (Δ F1 = -0.004). Contrairement à XGBoost, le CNN ne sur-dépend pas des features de taille de paquets et généralise mieux face à cette perturbation.

### Synthèse de robustesse

| Modèle | Jitter Résistance | Padding Résistance | Profil |
|---|---|---|---|
| **XGBoost** | ✅ Excellente | ❌ Très faible | Vulnérable au padding |
| **1D-CNN** | ⚠️ Modérée (dégrade à 20%) | ✅ Excellente | Plus généraliste |

> **Conclusion clé** : Les deux modèles ont des profils de robustesse **complémentaires**. Un ensemble (vote majoritaire) des deux modèles offrirait la meilleure défense contre différentes stratégies d'évasion.

---

## 8. Interprétabilité

### SHAP (XGBoost)

L'analyse SHAP (TreeExplainer) révèle les features les plus influentes dans les décisions de classification :

**Features les plus discriminantes (par gain XGBoost)** :
1. `PacketTimeStandardDeviation` (21.9%) — La variabilité des temps inter-arrivées est le signal le plus fort
2. `PacketLengthMode` (19.4%) — Le mode des tailles de paquets différencie fortement bénin/malveillant  
3. `Duration` (9.7%) — La durée du flux
4. `PacketTimeVariance` (8.8%) — Renforce le signal temporel
5. `PacketLengthMean` (8.3%) — La taille moyenne des paquets

**Interprétation** : Le trafic de tunneling (dns2tcp, DNScat2, Iodine) produit des signatures statistiques distinctes :
- **Temporellement** : Les tunnels ont des patterns d'arrivée plus réguliers (faible variance) car ils transmettent des données en continu, contrairement à la navigation web qui est sporadique
- **Spatialement** : Les paquets de tunneling ont des tailles plus uniformes (encodage de données), alors que la navigation web produit une diversité de tailles (pages, images, CSS, JS)

### Grad-CAM 1D (CNN)

Les heatmaps Grad-CAM montrent les régions du vecteur de features sur lesquelles le CNN concentre son attention. Cela confirme le focus sur les features temporelles (indices 13-21, groupe PacketTime) et les features de taille (indices 5-12, groupe PacketLength).

Les visualisations sont disponibles dans :
- `outputs/shap_summary.png` — Beeswarm plot SHAP
- `outputs/shap_bar.png` — Importance moyenne SHAP
- `outputs/shap_force_plot.html` — Force plots interactifs
- `outputs/gradcam_1d.png` — Heatmaps Grad-CAM

---

## 9. Limites et Biais Potentiels

### 1. Performances « trop parfaites » de XGBoost

Un F1 de 0.9997 et un AUC-ROC de 1.0000 sur 54K échantillons de test sont **exceptionnellement élevés**. Plusieurs facteurs peuvent expliquer cela :

- **Séparation forte des distributions** : Les outils de tunneling (dns2tcp, DNScat2, Iodine) produisent des signatures statistiques très distinctes du trafic web normal. Cela ne signifie pas qu'un nouvel outil de tunneling inconnu serait détecté aussi facilement.
- **Risque de leak temporel** : Bien que le split soit stratifié, des flux d'une même session pourraient se retrouver dans le train et le test (pas de split par session/IP).
- **Biais de dataset** : Le dataset capture les comportements spécifiques de 3 outils. En production, les attaquants utiliseront des outils différents ou modifiés.

### 2. Déséquilibre inversé

Le dataset contient **plus de malveillant que de bénin** (92.7% vs 7.3%), ce qui est inhabuel dans un scénario réel. En production, la proportion serait inversée (>99% bénin). Nos métriques pourraient donc surestimer les performances, particulièrement pour la classe bénigne.

### 3. Vulnérabilité au padding (XGBoost)

La chute catastrophique de XGBoost sous padding (F1 : 0.9997 → 0.4587) est un **risque majeur** en déploiement. Un attaquant averti pourrait simplement ajouter du bourrage aléatoire à ses paquets pour contourner la détection.

### 4. Features pré-calculées

Les 29 features sont des statistiques agrégées par flux. Un CNN sur des **séquences brutes** de paquets (séries temporelles de tailles/temps) pourrait capturer des patterns plus fins et être plus difficile à contourner.

---

## 10. Conclusions et Recommandations

### Conclusions principales

1. **La classification comportementale du trafic DoH est réalisable** avec d'excellentes performances. Les signatures statistiques spatio-temporelles suffisent à distinguer le tunneling malveillant de la navigation légitime.

2. **XGBoost est le modèle le plus performant** (F1=0.9997) et le plus rapide (~4s d'entraînement). Il est le choix optimal pour un déploiement en production dans un SOC.

3. **Le 1D-CNN atteint F1=0.972** et offre une robustesse complémentaire, particulièrement face aux perturbations de taille de paquets.

4. **L'analyse adversariale révèle une vulnérabilité critique** : XGBoost est très vulnérable au padding (F1 chute à 0.46), tandis que le CNN est plus vulnérable au jitter (F1 chute à 0.85 à σ=20%).

5. **L'approche hybride (ML+DL) est justifiée** : les profils de robustesse complémentaires des deux modèles suggèrent qu'un système de vote ou d'ensemble améliorerait la résilience globale.

### Recommandations

| # | Recommandation | Priorité |
|---|---|---|
| 1 | **Déployer XGBoost comme détecteur principal** avec le CNN comme détecteur secondaire (vote majoritaire) | Haute |
| 2 | **Ajouter un adversarial training** : ré-entraîner les modèles sur des données perturbées (jitter + padding) pour améliorer la robustesse | Haute |
| 3 | **Valider sur de nouveaux outils de tunneling** (e.g., dnscat3, sliver-DoH) pour évaluer la généralisation | Haute |
| 4 | **Implémenter un split par session/IP** pour éliminer tout risque de leak entre train/test | Moyenne |
| 5 | **Explorer un CNN sur séquences brutes** (time series de paquets) pour capturer des patterns temporels plus fins | Moyenne |
| 6 | **Monitorer le drift en production** : la composante temporelle (--chrono) du pipeline permet de détecter la dégradation des performances au fil du temps | Moyenne |
| 7 | **Tester avec un ratio bénin/malveillant réaliste** (>99% bénin) pour valider le comportement en conditions réelles | Basse |

### Fichiers générés

| Fichier | Description |
|---|---|
| `outputs/xgboost_model.json` | Modèle XGBoost sérialisé |
| `outputs/cnn_model.pt` | Modèle 1D-CNN PyTorch |
| `outputs/metrics.json` | Métriques détaillées (JSON) |
| `outputs/confusion_matrices.png` | Matrices de confusion comparatives |
| `outputs/roc_pr_curves.png` | Courbes ROC et Precision-Recall |
| `outputs/confidence_dist_xgboost.png` | Distribution de confiance XGBoost |
| `outputs/confidence_dist_1d-cnn.png` | Distribution de confiance CNN |
| `outputs/cnn_training_history.png` | Courbes d'entraînement du CNN |
| `outputs/robustness_summary.png` | Synthèse visuelle de robustesse |
| `outputs/adversarial_results.json` | Métriques adversariales détaillées |
| `outputs/shap_summary.png` | SHAP beeswarm plot |
| `outputs/shap_bar.png` | SHAP bar chart (importance globale) |
| `outputs/shap_force_plot.html` | SHAP force plots interactifs |
| `outputs/gradcam_1d.png` | Grad-CAM 1D heatmaps |

---

*Rapport généré automatiquement par le pipeline SENTINEL-DoH.*  
*Dataset : CIRA-CIC-DoHBrw-2020 — 269 643 échantillons — 29 features — Split stratifié 70/10/20.*
