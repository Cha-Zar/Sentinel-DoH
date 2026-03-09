# SENTINEL-DoH: When XGBoost Collapses Under Padding — A Robustness-First Comparison of ML and DL for DNS-over-HTTPS Tunnel Detection

---

**Abstract** — DNS over HTTPS (DoH) encrypts DNS queries inside standard HTTPS sessions, giving attackers a covert channel for data exfiltration and command-and-control (C2) communication that traditional firewalls cannot inspect. This paper tackles the binary classification of DoH traffic as benign or malicious (tunneling) using only statistical flow-level features — no payload decryption involved. We train and compare an XGBoost classifier against a 1D Convolutional Neural Network (1D-CNN) on the CIRA-CIC-DoHBrw-2020 dataset (269,643 flows, 29 features). Both models reach high baseline accuracy (F1-macro of 0.9997 and 0.9720, respectively), but the real story emerges from our adversarial robustness analysis: XGBoost, despite near-perfect clean performance, suffers a catastrophic F1 drop to 0.46 under a simple ±10% packet-size padding attack, while the CNN remains stable at 0.97. The opposite pattern appears with temporal jitter — XGBoost barely flinches, while the CNN degrades to 0.85 at 20% noise. These complementary vulnerability profiles suggest that no single model is "robust enough" on its own. We further analyze model behavior through SHAP (XGBoost) and Grad-CAM (CNN), identify high-confidence failure cases, and discuss the practical limits of lab-based evaluations. All code, Docker environment, and dataset links are publicly available for full reproducibility.

**Keywords** — DNS over HTTPS, traffic classification, XGBoost, 1D-CNN, adversarial robustness, SHAP, Grad-CAM, CIRA-CIC-DoHBrw-2020

---

## I. Introduction

The adoption of DNS over HTTPS has accelerated since its standardization in RFC 8484 (2018) [1]. Major browsers — Firefox (default since February 2020), Chrome (since May 2020), Edge, Brave — now route DNS queries through encrypted HTTPS tunnels, making DNS traffic indistinguishable from regular web browsing at the network level. This is good for user privacy. It is also good for attackers.

Tools like dns2tcp [2], DNScat2 [3], and Iodine [4] exploit the DoH channel to build covert tunnels. An attacker encodes stolen data or C2 commands within DNS queries that travel over HTTPS, blending in with legitimate traffic on port 443. Traditional deep packet inspection (DPI) is useless here — the payload is encrypted. The NSA warned enterprises about this exact risk in January 2021, recommending tighter control of DoH resolvers [5].

The classification problem is straightforward on paper: given a network flow, decide if it is *benign* (normal browsing) or *malicious* (tunneling). In practice, three complications arise that most prior work tends to underplay:

1. **Class imbalance.** In any real network, the vast majority of DoH traffic is legitimate. Models that optimize raw accuracy will learn to always predict "benign" and still score 90%+.

2. **Adversarial evasion.** An attacker aware of the detection system can alter their traffic patterns — add random delays between packets, pad packet sizes — to escape classification. If a model relies on a few sharp decision boundaries, simple perturbations can break it.

3. **Overconfidence.** A model that is wrong and *sure about it* is worse than one that is wrong and uncertain, because an analyst trusts a high-confidence alert differently than a borderline one.

This paper addresses all three. We build two classifiers (Section III), evaluate them honestly on the standard CIRA-CIC-DoHBrw-2020 dataset (Section IV), then stress-test them with controlled adversarial perturbations and analyze their failures through interpretability methods (Sections V–VI). Our central finding is that XGBoost and the 1D-CNN have *opposite* robustness profiles — a result that has direct implications for deployment in a Security Operations Center (SOC).

**Contributions.** We make three specific contributions:

- **C1.** A controlled side-by-side comparison of XGBoost and a 1D-CNN on the same preprocessing pipeline, dataset splits, and metrics — eliminating methodological differences that often muddy ML vs. DL comparisons.
- **C2.** A documented adversarial robustness evaluation showing that XGBoost collapses under simple padding perturbations (F1: 0.9997 → 0.4587) while the CNN holds, and conversely, the CNN degrades under temporal jitter while XGBoost barely changes.
- **C3.** An interpretability analysis using SHAP and Grad-CAM 1D, combined with a study of high-confidence failure cases, to explain *why* each model breaks where it does.

---

## II. Related Work

### A. DoH Traffic Classification

MontazeriShatoori et al. [6] introduced the CIRA-CIC-DoHBrw-2020 dataset and proposed the DoHLyzer tool for feature extraction. Their original work used time-series classification of encrypted traffic to detect DoH tunnels, achieving strong results with classical statistical classifiers. This dataset has since become a standard benchmark for DoH detection research, cited in over 300 subsequent studies.

Banadaki and Robert [7] applied six machine learning classifiers to the same dataset for distinguishing benign from malicious DoH traffic, reporting accuracy above 99% for several models — a result consistent with ours but, as we will argue, incomplete without robustness testing. Singh and Roy [8] confirmed these findings using Random Forest and Gradient-Boosted Trees, reporting near-perfect F1 scores on clean test data.

Behnke et al. [9] compared 10 different ML classifiers with a focus on feature engineering and showed that even simple models perform well on this task. Vekshin et al. [10] proposed the DoH Insight framework, achieving over 99% detection accuracy with Random Forest and showing that flow-level statistics are sufficient to identify DoH traffic. More recently, Abu Al-Haija et al. [11] introduced a lightweight double-stage scheme combining Random Forest and deep learning, confirming the hybrid approach for DoH classification.

Casanova and Lin [12] took the deep learning route, applying CNN architectures to generalize DoH classification across different traffic scenarios. Li et al. [13] focused specifically on hyperparameter optimization for learning-based DoH classifiers, showing that proper tuning matters more than model choice in some cases.

### B. Adversarial Robustness in Network Intrusion Detection

The vulnerability of ML-based network intrusion detection systems (NIDS) to adversarial examples is well-documented. He et al. [14] provide a comprehensive survey of adversarial attacks against NIDS, covering evasion techniques, threat models, and defense strategies. Han et al. [15] showed that even state-of-the-art NIDS like Kitsune can be evaded with over 97% success in half the tested scenarios.

Zhang and Costa-Perez [16] demonstrated five types of adversarial attacks against deep learning NIDS, finding that standard defenses like adversarial training offer limited protection. Merzouk et al. [17] investigated the practicality of these attacks in realistic conditions, showing that perturbations must respect network protocol constraints to remain valid.

What is notably absent in the DoH-specific literature is a robustness evaluation of this kind. Most DoH classification papers report clean-data performance and stop there. Elaoumari [18] recently addressed evasion-resilient DoH detection with a modified DoHLyzer, but focused on infrastructure changes rather than model-level robustness. Our work fills this gap by testing both ML and DL models against controlled adversarial perturbations *within the feature space*, measuring exactly where each model breaks.

### C. Class Imbalance in Cybersecurity

Handling imbalanced datasets is a recurring challenge in security applications. SMOTE (Synthetic Minority Over-sampling Technique) [19] remains the most widely used resampling method. The combination of SMOTE with cost-sensitive learning (e.g., XGBoost's `scale_pos_weight`) has been shown effective in network intrusion detection [20]. We adopt this dual approach in our pipeline.

### D. Interpretability

SHAP (SHapley Additive exPlanations) [21] has become the go-to method for explaining tree-based models. For CNNs, Grad-CAM [22] — originally designed for image classification — can be adapted to 1D signals by computing gradient-weighted activations over convolutional layers. In the cybersecurity context, interpretability is not a luxury but an operational requirement: human analysts need to understand *why* an alert was raised before taking action [23].

---

## III. Methodology

### A. Dataset

We use the **CIRA-CIC-DoHBrw-2020** dataset [6], specifically the Layer-2 files that classify DoH traffic as benign or malicious at the application level.

**Benign traffic** was captured from Chrome and Firefox browsers performing normal web browsing over DoH resolvers (Cloudflare, Google). **Malicious traffic** was generated using three tunneling tools: dns2tcp (TCP-over-DNS tunneling), DNScat2 (interactive C2 over DNS), and Iodine (IP-over-DNS tunneling). Each tool represents a different tunneling strategy, providing diversity in the malicious class.

The dataset contains **269,643 flows** with the following class distribution:

| Class | Count | Ratio |
|-------|-------|-------|
| Benign | 19,807 | 7.3% |
| Malicious | 249,836 | 92.7% |

Each flow is described by **29 numerical features** extracted by DoHLyzer [6], grouped into four categories:

- **Flow features** (5): Duration, FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate
- **PacketLength statistics** (8): Variance, StdDev, Mean, Median, Mode, SkewFromMedian, SkewFromMode, CoefficientOfVariation
- **PacketTime (IAT) statistics** (8): same eight statistical measures applied to inter-arrival times
- **ResponseTime statistics** (8): same measures applied to DNS response times

No IP addresses, domain names, or payload content are used — only behavioral metadata. This makes the approach inherently GDPR-compliant.

### B. Preprocessing Pipeline

The preprocessing follows a strict protocol to avoid data leakage:

1. **Loading.** The two Layer-2 CSV files (`l2-benign.csv`, `l2-malicious.csv`) are concatenated. String labels ("Benign"/"Malicious") are encoded as 0/1.

2. **Cleaning.** Metadata columns (IP addresses, ports) are dropped. Infinite values are replaced by NaN, then imputed with the column median. Remaining NaN rows are dropped.

3. **Splitting.** A stratified 70/10/20 split partitions the data into training, validation, and test sets. The class ratio is preserved in each split. The test set (53,929 flows) is frozen — no hyperparameter decision is made on it.

4. **Scaling.** A StandardScaler is fitted on the training set only, then applied to validation and test sets.

5. **SMOTE.** Applied to the training set only after scaling. This brings the minority class (Benign) from 13,865 to 174,884 samples, matching the majority class for a balanced training set of 349,768 flows.

| Set | Before SMOTE | After SMOTE | Benign | Malicious |
|-----|-------------|-------------|--------|-----------|
| Train | 188,749 | 349,768 | 174,884 | 174,884 |
| Validation | 26,965 | — | 1,981 | 24,984 |
| Test | 53,929 | — | 3,961 | 49,968 |

### C. XGBoost Classifier

XGBoost [24] is our ML baseline. It is the standard choice for tabular classification tasks and handles imbalanced data natively through the `scale_pos_weight` parameter.

**Configuration:**
- 300 estimators, max depth 6, learning rate 0.1
- Subsample: 0.8, colsample_bytree: 0.8
- Regularization: gamma 0.1, L1 (alpha) 0.1, L2 (lambda) 1.0
- After SMOTE, `scale_pos_weight` is set to 1.0 (classes are balanced in training)

No grid search was performed — these are reasonable defaults for this dataset size. We deliberately avoid extensive tuning to show that even a standard configuration achieves strong results, which is itself a finding about the dataset's separability.

### D. 1D-CNN Architecture

Our deep learning model is a 1D Convolutional Neural Network built in PyTorch. The 29-dimensional feature vector is treated as a 1D signal of length 29 with a single input channel.

**Architecture:**

```
Input (1 × 29)
   → Conv1D(1→64, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool(2)
   → Conv1D(64→128, kernel=3, padding=1) → GlobalAveragePooling
   → Dense(128→64) → ReLU → Dropout(0.3)
   → Dense(64→1) → Sigmoid
```

**Training:**
- Loss: BCEWithLogitsLoss with pos_weight set to the class ratio
- Optimizer: Adam (initial lr = 1e-3)
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 3)
- Early stopping: patience 7 on validation AUC-ROC
- Batch size: 256, max epochs: 50

The architecture is intentionally compact — two convolutional layers with global average pooling rather than a fully connected stack. This reflects the fact that 29 features is a short signal; deeper architectures would risk overfitting.

### E. Evaluation Metrics

We report:
- **F1-macro**: the arithmetic mean of per-class F1 scores, giving equal weight to both classes regardless of their size in the test set. This is the primary metric.
- **AUC-ROC** and **AUC-PR**: threshold-independent measures of ranking quality. AUC-PR is more informative than AUC-ROC under class imbalance [25].
- **Precision and Recall per class**: to understand the nature of errors (false positives vs. false negatives).
- **High-Confidence Failures (HCF)**: the number of misclassifications where the model predicted with confidence > 0.95. A model that fails while being overconfident is more dangerous than one that fails with low certainty.

Accuracy is reported for completeness but is *not* a decision metric, as a trivial majority-class classifier would score 92.7%.

### F. Robustness Testing

We test both models against two types of perturbations that simulate a realistic attacker:

**1. Temporal Jitter.** Gaussian noise is added to all PacketTime features (8 features) with σ ∈ {5%, 10%, 20%} of each feature's standard deviation. This simulates an attacker who randomizes the timing between packets to break temporal signatures. This is cheap to implement — a simple `sleep(random)` in the tunneling tool.

**2. Spatial Padding.** Uniform random noise of ±10% is added to all PacketLength features (8 features). This simulates an attacker who pads packets with random bytes to obscure size-based signatures. Again, trivial to implement.

The perturbations are applied to the *test set only* — models are not retrained. We measure the same metrics on perturbed data and report the F1 delta from clean performance.

### G. Interpretability

**SHAP (XGBoost).** We use TreeExplainer [21] on a sample of 500 test instances to compute Shapley values, generate beeswarm plots (global importance) and force plots (individual decisions).

**Grad-CAM 1D (CNN).** We adapt the Grad-CAM method [22] to our 1D architecture. Gradients of the output with respect to the second convolutional layer activations are computed, averaged over the spatial dimension, and used to weight the activation maps. The result is a heatmap over the 29 input features showing which regions the CNN focuses on for each prediction.

---

## IV. Results

### A. Clean Performance

Both models achieve strong performance on the unperturbed test set:

| Metric | XGBoost | 1D-CNN |
|--------|---------|--------|
| **F1-macro** | **0.9997** | 0.9720 |
| AUC-ROC | 1.0000 | 0.9991 |
| AUC-PR | 1.0000 | 0.9999 |
| Precision (Benign) | 1.00 | 0.91 |
| Recall (Benign) | 1.00 | 0.99 |
| Precision (Malicious) | 1.00 | 1.00 |
| Recall (Malicious) | 1.00 | 0.99 |

**XGBoost** misclassifies just 4 out of 53,929 test flows (1 false positive, 3 false negatives). Its F1-macro of 0.9997 and AUC-ROC of 1.0000 indicate near-perfect separation between classes.

**The 1D-CNN** commits 429 errors (28 false positives, 401 false negatives). Its F1-macro of 0.972 is excellent but measurably below XGBoost. Training took 50 epochs (~14 minutes on CPU), with the learning rate decaying from 1e-3 to 7.81e-6 via ReduceLROnPlateau. Validation AUC-ROC climbed from 0.971 at epoch 1 to 0.9997 by epoch 50, showing steady convergence.

**Top-5 XGBoost features by information gain:**

| Rank | Feature | Gain |
|------|---------|------|
| 1 | PacketTimeStandardDeviation | 0.219 |
| 2 | PacketLengthMode | 0.194 |
| 3 | Duration | 0.097 |
| 4 | PacketTimeVariance | 0.088 |
| 5 | PacketLengthMean | 0.083 |

This ranking is revealing. The two most important features are one temporal (PacketTimeStandardDeviation) and one spatial (PacketLengthMode). Tunneling traffic has more regular inter-arrival times (lower variance) because it transmits data continuously, unlike web browsing which is bursty. Packet sizes in tunneling flows cluster around fixed encoding sizes (high mode concentration), while browsing produces diverse content sizes.

### B. High-Confidence Failure Analysis

| Model | Total FP | HC-FP (>0.95) | Total FN | HC-FN (>0.95) |
|-------|----------|---------------|----------|---------------|
| XGBoost | 1 | 0 | 3 | 1 |
| 1D-CNN | 28 | 6 (21%) | 401 | 106 (26%) |

XGBoost makes almost no errors and is not overconfident in its rare mistakes. The CNN, however, is overconfident in 26% of its false negatives — it lets 106 malicious flows pass while being >95% sure they are benign. In a SOC environment, this means missed attacks with no flag for human review.

### C. Discussion of Clean Performance

The near-perfect XGBoost results deserve scrutiny rather than celebration. An F1 of 0.9997 on a test set of 54K samples with 29 features suggests one of three things: (a) the task is genuinely easy on this dataset, (b) there is information leakage, or (c) the model memorized tool-specific signatures rather than learning the general concept of "tunneling."

We believe (a) and (c) are both partially true. The three tunneling tools (dns2tcp, DNScat2, Iodine) produce statistically distinct traffic patterns that differ sharply from browser-generated DoH. An XGBoost model with 300 trees has more than enough capacity to learn these distinct signatures. This does not mean a *new* tunneling tool (e.g., dnscat3, DoH-specific QUIC tunnels) would be detected equally well. We return to this in Section VII.

Regarding (b), we verified that there is no direct leakage (no IP/port features, no temporal ordering). However, flows from the same capture session may share ambient characteristics (e.g., network conditions), and our stratified random split does not prevent same-session contamination. A per-session split would provide a stricter evaluation, though the dataset metadata does not cleanly support this.

---

## V. Adversarial Robustness

This section contains the most important findings of the paper.

### A. Temporal Jitter (IAT Noise)

| Model | Clean F1 | σ=5% | σ=10% | σ=20% |
|-------|----------|------|-------|-------|
| XGBoost | 0.9997 | 0.9987 (−0.001) | 0.9990 (−0.001) | 0.9990 (−0.001) |
| 1D-CNN | 0.9720 | 0.9652 (−0.007) | 0.9431 (−0.029) | 0.8511 (−0.121) |

**XGBoost is nearly immune to temporal jitter.** Even at σ=20%, its F1 drops by only 0.07 percentage points. The AUC-ROC remains at 1.0000. This stability comes from XGBoost's tree-based decision boundaries, which are piecewise-constant and robust to small continuous perturbations within each partition.

**The CNN degrades progressively.** At σ=20%, its F1 drops from 0.972 to 0.851 — a 12-point loss. The precision for the benign class falls from 0.91 to 0.59, meaning many benign flows are falsely flagged. The CNN's convolutional filters learn smooth patterns across the feature vector; noise in the temporal features disrupts these patterns more severely than it disrupts tree splits.

### B. Spatial Padding (Packet-Size Noise)

| Model | Clean F1 | Padding ±10% |
|-------|----------|-----------|
| XGBoost | 0.9997 | **0.4587** (−0.541) |
| 1D-CNN | 0.9720 | **0.9682** (−0.004) |

**This is the central finding.** Padding ±10% on packet-size features *destroys* XGBoost's performance. Its F1 drops from 0.9997 to 0.4587 — a catastrophic 54-point collapse. Looking at the class-level breakdown, the recall for the malicious class drops to 0.508: the model lets through *half* of all attacks. The precision for benign drops to 0.14: nearly all "benign" predictions are actually malicious.

Why? XGBoost's second-most-important feature is `PacketLengthMode` (19.4% of total gain). Tunneling tools produce packets of fixed, characteristic sizes (e.g., DNS query encoding overhead). A sharp split on `PacketLengthMode ≤ threshold` can perfectly separate the classes in clean data. But adding ±10% noise blurs these clean boundaries. The tree's hard thresholds become meaningless when the feature distribution shifts.

**The CNN barely notices the same perturbation.** Its F1 drops by just 0.4 percentage points. The CNN's convolutional layers learn distributed representations that combine multiple features, making it inherently more resistant to perturbation of any single feature group. Global average pooling further smooths out local perturbations.

### C. Robustness Summary

| | Jitter Resistance | Padding Resistance |
|---|---|---|
| **XGBoost** | Excellent (ΔF1 < 0.001) | **Very poor** (ΔF1 = −0.541) |
| **1D-CNN** | Moderate (ΔF1 = −0.121 at 20%) | Excellent (ΔF1 = −0.004) |

The two models have **complementary vulnerability profiles.** XGBoost relies heavily on sharp splits in the packet-size space but is insensitive to temporal noise. The CNN learns distributed, smooth representations that handle spatial perturbations well but are more sensitive to temporal noise that shifts the learned convolutional patterns.

This finding has a practical implication: **a simple majority-vote ensemble of XGBoost and the CNN would be resilient to both attack vectors**, because neither perturbation degrades both models simultaneously. We leave the formal evaluation of such an ensemble to future work.

---

## VI. Interpretability

### A. SHAP Analysis (XGBoost)

The SHAP beeswarm plot reveals a clear pattern: temporal and spatial features dominate the model's decisions, but in distinct ways.

**PacketTimeStandardDeviation** is the single most influential feature. High values (diverse inter-arrival times, typical of bursty web browsing) push the prediction toward benign; low values (regular timing, typical of continuous data tunneling) push toward malicious. This aligns with domain knowledge: dns2tcp and Iodine maintain relatively steady data streams, while human browsing is inherently sporadic.

**PacketLengthMode** shows a bimodal influence. Certain mode values are strong indicators of specific tunneling tools (e.g., dns2tcp uses fixed-size encoding), while browser DoH produces diverse packet sizes with no dominant mode. This explains why XGBoost is so vulnerable to padding attacks: perturbing the mode value undermines one of its two strongest decision criteria.

**Duration** (third in importance) acts as a secondary signal — tunneling sessions tend to be longer and more sustained than typical DNS resolution sequences.

### B. Grad-CAM 1D Analysis (CNN)

The Grad-CAM heatmaps show that the CNN attends primarily to two regions of the feature vector:

1. **Features 10–17** (PacketTime group): consistent high activation, confirming that temporal behavior is the primary discriminant for the CNN too.
2. **Features 5–12** (PacketLength group): moderate but distributed activation, without the sharp dependence on any single feature that XGBoost exhibits.

The key difference: the CNN spreads its attention across several related features rather than relying on one or two decisive splits. This distributed attention is precisely what makes it more robust to targeted perturbation of any single feature, and less robust to broad perturbation of an entire feature group (as in the 20% jitter case, where all 8 temporal features are degraded simultaneously).

---

## VII. Limitations and Threats to Validity

We identify five limitations that must be acknowledged:

**1. Tool-specific signatures.** The malicious class contains traffic from only three tools (dns2tcp, DNScat2, Iodine). The models may have learned *tool fingerprints* rather than the general concept of DNS tunneling. A new tunneling tool with different encoding (e.g., DoH-specific QUIC-based tunnels) might not trigger detection. Cross-tool validation — training on two tools and testing on the third — would clarify this, but the available metadata does not cleanly support it.

**2. Inverted class ratio.** The dataset contains 92.7% malicious traffic, which is the opposite of any real network (where >99% of traffic is benign). Although our metrics (F1-macro, AUC-PR) partially account for this, the model's calibration — the correspondence between predicted probabilities and actual frequencies — has not been tested under realistic priors.

**3. No per-session split.** Our stratified random split may allow flows from the same capture session to appear in both training and test sets. This could inflate performance if ambient network conditions create session-level correlations. A stricter evaluation would split by source IP or session ID.

**4. Feature-space vs. traffic-space perturbations.** Our adversarial tests perturb the *feature* values directly, not the underlying network traffic. A real attacker cannot choose arbitrary feature values — they must generate valid network packets that *produce* those feature values after extraction by DoHLyzer. Some of our perturbation scenarios (e.g., negative packet sizes) are technically impossible. A more realistic adversarial evaluation would generate perturbed traffic at the packet level.

**5. Single dataset.** All results are specific to CIRA-CIC-DoHBrw-2020. Generalization to other DoH datasets (e.g., from different ISPs, operating systems, or resolver providers) is unknown.

---

## VIII. Conclusion

We have presented a side-by-side evaluation of XGBoost and a 1D-CNN for detecting malicious DNS-over-HTTPS traffic. On clean test data, both models perform well, with XGBoost achieving near-perfect F1 of 0.9997. But the real value of this study lies in what happens *after* the standard evaluation.

Our adversarial analysis uncovered a striking result: **XGBoost's near-perfect accuracy masks a critical vulnerability.** A trivial ±10% padding perturbation on packet sizes — something an attacker could implement in minutes — collapses XGBoost's F1 from 0.9997 to 0.4587. The CNN, in contrast, remains at 0.9682 under the same attack. The inverse holds for temporal jitter, where XGBoost is rock-solid and the CNN weakens.

This complementary fragility pattern carries a clear operational message: **deploying either model alone creates a known, exploitable weakness.** An ensemble combining both models would cover each other's blind spots. We also showed through SHAP and Grad-CAM that the vulnerability is structurally explainable — XGBoost overfits hard decision boundaries on a few dominant features, while the CNN distributes its attention more evenly but depends on all temporal features simultaneously.

For the cybersecurity community, the takeaway is that **clean-data F1 alone is insufficient for evaluating a detection model.** The gap between 0.9997 and 0.4587 is a gap between a lab benchmark and reality. We encourage future work on DoH classification to systematically include adversarial robustness testing and to explore traffic-level (rather than feature-level) perturbations.

---

## References

[1] P. Hoffman and P. McManus, "DNS Queries over HTTPS (DoH)," RFC 8484, IETF, October 2018.

[2] O. Dembour, "dns2tcp — a tool for relaying TCP connections over DNS," 2010. [Online]. Available: https://github.com/alex-sector/dns2tcp

[3] R. Bowes, "dnscat2 — Command-and-Control over DNS," 2019. [Online]. Available: https://github.com/iagox86/dnscat2

[4] E. Ekman, "Iodine — IPv4 over DNS tunnel," 2019. [Online]. Available: https://github.com/yarrick/iodine

[5] NSA, "Adopting Encrypted DNS in Enterprise Environments," Cybersecurity Information Sheet, January 2021.

[6] M. MontazeriShatoori, L. Davidson, G. Kaur, and A. H. Lashkari, "Detection of DoH Tunnels using Time-Series Classification of Encrypted Traffic," in Proc. IEEE Intl. Conf. on Dependable, Autonomic and Secure Computing (DASC), 2020, pp. 63-70.

[7] Y. M. Banadaki and S. Robert, "Detecting Malicious DNS over HTTPS Traffic in Domain Name System using Machine Learning Classifiers," Journal of Computer Sciences and Applications, vol. 8, no. 2, pp. 46-55, 2020.

[8] S. K. Singh and P. K. Roy, "Detecting Malicious DNS over HTTPS Traffic Using Machine Learning," in Proc. Intl. Conf. on Innovation and Intelligence for Informatics, Computing and Technologies, 2020.

[9] M. Behnke, N. Briner, D. Cullen, K. Schwerdtfeger, et al., "Feature Engineering and Machine Learning Model Comparison for Malicious Activity Detection in the DNS-over-HTTPS Protocol," IEEE Access, vol. 9, pp. 129902-129916, 2021.

[10] D. Vekshin, K. Hynek, and T. Čejka, "DoH Insight: Detecting DNS over HTTPS by Machine Learning," in Proc. 15th Intl. Conf. on Availability, Reliability and Security (ARES), 2020.

[11] Q. Abu Al-Haija, M. Alohaly, and A. Odeh, "A Lightweight Double-Stage Scheme to Identify Malicious DNS over HTTPS Traffic Using a Hybrid Learning Approach," Sensors, vol. 23, no. 7, p. 3489, 2023.

[12] L. F. G. Casanova and P. C. Lin, "Generalized Classification of DNS over HTTPS Traffic with Deep Learning," in Proc. Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), 2021, pp. 1903-1910.

[13] Y. Li, A. Dandoush, and J. Liu, "Evaluation and Optimization of Learning-based DNS over HTTPS Traffic Classification," in Proc. Intl. Symposium on Networks, Computers and Communications (ISNCC), 2021.

[14] K. He, D. D. Kim, and M. R. Asghar, "Adversarial Machine Learning for Network Intrusion Detection Systems: A Comprehensive Survey," IEEE Communications Surveys & Tutorials, vol. 25, no. 1, pp. 538-566, 2023.

[15] D. Han, Z. Wang, Y. Zhong, W. Chen, et al., "Evaluating and Improving Adversarial Robustness of Machine Learning-Based Network Intrusion Detectors," IEEE Journal on Selected Areas in Communications, vol. 39, no. 8, pp. 2632-2647, 2021.

[16] C. Zhang and X. Costa-Perez, "Adversarial Attacks Against Deep Learning-Based Network Intrusion Detection Systems and Defense Mechanisms," IEEE/ACM Transactions on Networking, vol. 30, no. 3, pp. 1294-1311, 2022.

[17] M. A. Merzouk, F. Cuppens, N. Boulahia-Cuppens, et al., "Investigating the Practicality of Adversarial Evasion Attacks on Network Intrusion Detection," Annals of Telecommunications, vol. 77, pp. 607-624, 2022.

[18] A. Elaoumari, "Evasion-Resilient Detection of DNS-over-HTTPS Data Exfiltration: A Practical Evaluation and Toolkit," arXiv preprint arXiv:2512.20423, 2025.

[19] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.

[20] A. Fernández, S. García, F. Herrera, and N. V. Chawla, "SMOTE for Learning from Imbalanced Data: Progress and Challenges," Journal of Artificial Intelligence Research, vol. 61, pp. 863-905, 2018.

[21] S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.

[22] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in Proc. IEEE Intl. Conf. on Computer Vision (ICCV), 2017, pp. 618-626.

[23] A. Adadi and M. Berrada, "Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)," IEEE Access, vol. 6, pp. 52138-52160, 2018.

[24] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proc. 22nd ACM SIGKDD Intl. Conf. on Knowledge Discovery and Data Mining, 2016, pp. 785-794.

[25] J. Davis and M. Goadrich, "The Relationship Between Precision-Recall and ROC Curves," in Proc. 23rd Intl. Conf. on Machine Learning (ICML), 2006, pp. 233-240.

---

## Appendix A: Reproducibility

All source code is available at: https://github.com/Cha-Zar/Sentinel-DoH

**To reproduce all results:**

```bash
# Option 1: Docker (recommended)
docker build -t sentinel-doh .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs sentinel-doh

# Option 2: Local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Dataset:** CIRA-CIC-DoHBrw-2020, available at https://www.unb.ca/cic/datasets/dohbrw-2020.html

**Environment:** Python 3.10+, PyTorch ≥ 2.0, XGBoost ≥ 1.7, scikit-learn ≥ 1.2, SHAP ≥ 0.42. Full list in `requirements.txt`.

**Hardware:** All experiments were run on a standard CPU (no GPU required). Full pipeline execution: ~14 minutes.

## Appendix B: Model Card

**Model Name:** SENTINEL-DoH (XGBoost + 1D-CNN)

**Task:** Binary classification of DNS-over-HTTPS traffic (Benign vs. Malicious)

**Training Data:** CIRA-CIC-DoHBrw-2020 Layer-2, 269,643 flows (19,807 benign / 249,836 malicious)

**Input:** 29 numerical flow-level features (packet sizes, inter-arrival times, flow statistics)

**Output:** Binary prediction (0 = Benign, 1 = Malicious) with confidence score

**Performance (test set, 53,929 flows):**
- XGBoost: F1-macro = 0.9997, AUC-ROC = 1.0000
- 1D-CNN: F1-macro = 0.9720, AUC-ROC = 0.9991

**Known Limitations:**
- Trained on 3 tunneling tools only (dns2tcp, DNScat2, Iodine). May not detect novel tools.
- XGBoost is critically vulnerable to packet-size padding attacks (F1 drops to 0.46).
- CNN is sensitive to temporal jitter at 20% noise (F1 drops to 0.85).
- Dataset class ratio (7.3% benign) does not match real-world traffic distributions.
- No per-session train/test split — potential for session-level leakage.

**Ethical Considerations:**
- No personal user data is processed (only statistical metadata).
- False positives could block legitimate traffic; false negatives miss attacks.
- Should be deployed as an assistive tool for human analysts, not as an autonomous blocking system.

**Intended Use:** Research benchmark and SOC alert triage. Not intended for autonomous traffic blocking without human review.
