# **SENTINEL-DoH : SPÉCIFICATIONS TECHNIQUES ET PROTOCOLE DE RECHERCHE**

**Projet :** Classification Comportementale du Trafic DNS over HTTPS par Analyse Spatio-Temporelle Hybride ML/DL  
**Problème scientifique :** Comment classifier de manière robuste et interprétable le trafic DoH malveillant (tunneling) face au déséquilibre des classes, à la dérive temporelle et aux perturbations adversariales, sans jamais inspecter le contenu chiffré ?

## **I. CONTEXTE ET POSITIONNEMENT SCIENTIFIQUE**

### **1.1. Le Problème de Fond**

Le protocole DNS over HTTPS (DoH) encapsule les requêtes DNS dans du trafic HTTPS standard, rendant la distinction entre résolution DNS légitime et tunneling malveillant opaque aux outils d'inspection traditionnels. Les attaquants exploitent ce canal chiffré pour l'exfiltration de données et la communication Command & Control (C2), précisément parce qu'il contourne les firewalls et les sondes DPI.  
Le défi scientifique n'est pas la détection en soi — c'est la détection robuste, explicable et évaluée honnêtement dans des conditions réalistes : classes déséquilibrées, distribution temporelle non stationnaire, et adversaire actif.

### **1.2. Hypothèse Centrale**

Un flux DoH malveillant, même parfaitement chiffré, produit une signature comportementale distincte dans deux dimensions observables sans déchiffrement : la distribution des tailles de paquets (**dimension spatiale**) et le rythme des inter-arrival times (**dimension temporelle**). Cette signature peut être apprise et généralisée par des modèles supervisés.

### **1.3. Contribution Scientifique**

Ce travail apporte trois contributions originales :

* **C1** — Comparaison rigoureuse ML vs. DL sur un protocole expérimental strictement contrôlé, avec gestion explicite du déséquilibre à chaque étape.  
* **C2** — Évaluation de robustesse documentée incluant dérive temporelle chronologique et perturbations adversariales quantifiées.  
* **C3** — Interprétabilité opérationnelle par SHAP (XGBoost) et Grad-CAM 1D (CNN), rendant les décisions auditables par un analyste humain non-expert en IA.

## **II. DATASET ET FORMALISATION DU PROBLÈME**

### **2.1. Dataset : CIRA-CIC-DoHBrw-2020**

* **Source :** Canadian Institute for Cybersecurity, Université de New Brunswick — accès public.  
* **Contenu :** Captures de trafic DoH réelles, prétraitées en features de flux (format CSV). Deux catégories : trafic DoH légitime (navigateurs standard) et trafic DoH malveillant (outils de tunneling : dns2tcp, DNScat, Iodine).  
* **Caractéristiques clés :**  
  * Déséquilibre naturel des classes (\~80% légitime / \~20% malveillant).  
  * Horodatage des flux permettant une analyse de dérive chronologique.  
  * Features pré-extraites permettant une reproductibilité immédiate.

### **2.2. Formalisation**

Le problème est formalisé comme une classification supervisée binaire :

* **Entrée :** Vecteur de métadonnées de flux DoH (aucun payload, aucune donnée personnelle).  
* **Sortie :** $\\{0 \= \\text{légitime}, 1 \= \\text{tunneling malveillant}\\}$.  
* **Contrainte :** Le ratio de classes déséquilibré doit être préservé dans le set de test pour évaluer les performances en conditions réelles.

## **III. ARCHITECTURE DES MODÈLES**

### **3.1. Baseline ML — XGBoost sur Vecteur Statistique Global**

**Pourquoi XGBoost ?** C'est l'algorithme de référence pour les données tabulaires déséquilibrées. Son paramètre scale\_pos\_weight offre un mécanisme natif de compensation du déséquilibre.

* **Features d'entrée :** Statistiques agrégées (Moyenne, Médiane, Écart-type, Max/Min) sur la taille des paquets et l'IAT, ainsi que des features globales (durée, entropie, ratio upload/download).  
* **Gestion du déséquilibre :** scale\_pos\_weight calculé sur le ratio négatifs/positifs \+ SMOTE appliqué uniquement sur le set d'entraînement.

### **3.2. Modèle DL — 1D-CNN sur Séquence Temporelle**

**Pourquoi 1D-CNN ?** Les patterns discriminants sont locaux et répétitifs (ex: alternance \[Petit-Petit-Grand\] du beaconing). Le 1D-CNN détecte ces motifs efficacement sans l'instabilité des LSTM.

* **Entrée séquentielle :** Matrice $(N \\times 2)$ des $N$ premiers paquets : $\[taille\\\_paquet, IAT\]$. $N$ est déterminé empiriquement par courbe d'apprentissage sur $N \\in \\{10, 20, 30, 50\\}$.  
* **Architecture :** \* Conv1D (64 filtres) → BatchNorm → ReLU → MaxPooling  
  * Conv1D (128 filtres) → GlobalAveragePooling  
  * Dense (64) → Dropout (0.3) → Dense (1, sigmoid)  
* **Gestion du déséquilibre :** Pondération de la fonction de perte (Weighted Binary Cross-Entropy).

## **IV. PROTOCOLE EXPÉRIMENTAL**

### **4.1. Splits et Validation**

| Split | Proportion | Stratégie |
| :---- | :---- | :---- |
| **Entraînement** | 70% | Stratifié, SMOTE appliqué ici uniquement |
| **Validation** | 10% | Stratifié, non augmenté |
| **Test** | 20% | Stratifié, set figé pour l'évaluation finale |

### **4.2. Métriques d'Évaluation**

* **Principales :** Precision, Recall, F1-score (macro-moyenne), AUC-ROC et AUC-PR.  
* **Justification :** L'accuracy est proscrite car trompeuse sous déséquilibre. Le F1-score macro pénalise équitablement les échecs sur la classe minoritaire.  
* **Analyse :** Matrices de confusion comparées et distribution des scores de confiance.

## **V. ROBUSTESSE ET TESTS ADVERSARIAUX**

### **5.1. Analyse de Dérive Temporelle (Data Drift)**

Le dataset est partitionné chronologiquement pour simuler un déploiement réel. On mesure la dégradation du F1-score entre un split aléatoire et un split chronologique (données passées vs futur).

### **5.2. Perturbations Adversariales Contrôlées**

Simulation d'un attaquant conscient de l'IA via deux stratégies :

1. **Jittering (Temps) :** Bruit gaussien $\\sigma \\in \\{5\\%, 10\\%, 20\\%\\}$ sur les IAT (randomisation du rythme).  
2. **Padding (Espace) :** Augmentation de la taille des paquets de $\\pm 10\\%$ pour masquer les signatures de taille fixe.

## **VI. INTERPRÉTABILITÉ ET ANALYSE HUMANO-CENTRÉE**

* **SHAP pour XGBoost :** "Summary Plot" pour l'importance globale et "Force Plot" pour expliquer une décision individuelle à un analyste SOC.  
* **Grad-CAM 1D pour le CNN :** Visualisation des segments de la séquence de paquets ayant le plus contribué à l'alerte.  
* **Analyse de la Surconfiance :** Examen des "High Confidence Failures" (Faux Positifs avec score $\> 0.95$).

## **VII. LIMITES ET BIAIS — HONNÊTETÉ SCIENTIFIQUE**

* **Généralisation :** Le dataset de laboratoire peut ne pas refléter la diversité des OS et VPN en production réelle.  
* **Horizon Temporel :** Une attaque "slow-burn" pourrait rester sous le seuil des $N$ premiers paquets.  
* **Biais d'Outils :** Le modèle risque d'apprendre la signature des outils spécifiques (*dns2tcp*, etc.) plutôt que le tunneling en général. Ce biais est quantifié par une analyse de performance par outil.

## **VIII. REPRODUCTIBILITÉ ET LIVRABLES**

### **8.1. Environnement Docker**

Le pipeline est encapsulé dans une image python:3.10-slim incluant xgboost, tensorflow/pytorch, shap et imbalanced-learn.  
Commande unique : docker run sentinel-doh

### **8.2. Structure des Livrables**

* SENTINEL\_AI\_article.pdf : Article scientifique au format IMRaD.  
* SENTINEL\_AI\_code.zip : Code source annoté.  
* SENTINEL\_AI\_environment.zip : Dockerfile \+ requirements.  
* SENTINEL\_AI\_Dataset.txt : Lien direct et checksum du dataset.

*Ce document constitue le socle de la recherche SENTINEL-DoH pour le challenge Cyber Horizon 2.0.*