# Technical Report: XRan - Explainable Deep Learning-based Ransomware Detection Using Dynamic Analysis

## 1. Problem Statement

Ransomware attacks have surged in both frequency and sophistication, posing critical threats to individuals and organizations. Traditional signature-based detection systems can identify known threats but fail against novel, zero-day ransomware variants that employ code obfuscation, encryption, and packing techniques to evade static analysis.

This project implements and evaluates **XRan**, a deep learning-based ransomware detection system proposed by Gulmez et al. (2024). XRan addresses two key limitations of existing methods:

1. **Single-perspective feature analysis**: Most prior work uses only API call sequences or frequency-based feature combinations, potentially ignoring cross-feature correlations.
2. **Black-box models**: Deep learning classifiers provide no insight into their decision-making process, limiting trust and actionability for security professionals.

XRan solves these by (a) combining API call, DLL, and Mutex sequences into a joint representation, and (b) integrating LIME and SHAP explainability models to provide local and global interpretations.

## 2. Related Work

### 2.1 Ransomware Detection

The literature on ransomware detection falls into two categories:

**Real-time monitoring approaches:**
- **ShieldFS** (Continella et al., 2016): Uses low-level filesystem monitoring to build a normal behavioral profile and detects deviations.
- **UNVEIL** (Kharaz et al., 2016): Detects file encryption and desktop locking through I/O request monitoring with Shannon entropy.
- **REDFISH** (Morato et al., 2018): Identifies ransomware via network traffic monitoring.

**Analysis-based approaches:**
- **Sgandurra et al. (2016)**: Dynamic analysis using Windows API calls, registry keys, and filesystem operations with Regularized Logistic Regression. Achieved 96.34% TPR on VirusShare data.
- **Hasan and Rahman (2017)**: Hybrid analysis with mutual information feature selection and SVM. Achieved 96.1% accuracy.
- **Hwang et al. (2020)**: Two-stage model combining Markov model with Random Forest on API calls and registry keys. Achieved 97.28% accuracy.
- **Qin et al. (2020)**: TextCNN on API call sequences, achieving 95.9% accuracy.
- **Jethva et al. (2020)**: Multi-layer technique combining API calls, DLLs, and registry keys with SVM/RF/LR. Achieved 100% TPR but 13.3% FPR.

### 2.2 XAI for Malware Detection

- **Drebin** (Arp et al., 2014): SVM-based Android malware detection reporting the most influential features by classifier weight.
- **Fan et al. (2020)**: Applied five XAI methods (LIME, Anchor, LORE, SHAP, LEMNA) to Android malware detection.
- **Feichtner and Gruber (2020)**: CNN with LIME for permission-description correlation analysis.
- **XMal** (Alani et al., 2023): Lightweight memory-based malware detector with SHAP global explanations.

XRan distinguishes itself as the first study to integrate XAI models specifically for ransomware detection.

## 3. Methodology

### 3.1 Feature Extraction

XRan uses **Cuckoo Sandbox** for dynamic analysis, executing samples in a Windows 7 virtual environment with a 60-minute analysis timeout. Three feature types are extracted:

1. **API Calls**: Windows API function calls that capture the behavioral footprint of the executable (e.g., `NtAllocateVirtualMemory`, `CryptEncrypt`, `RegOpenKeyExW`).
2. **DLLs**: Dynamic Link Libraries loaded during execution (e.g., `Advapi32.dll`, `Crypt32.dll`, `Kernel32.dll`).
3. **Mutexes**: Mutual exclusion objects used for process synchronization (e.g., `ZonesCounterMutex`, `ZonesLockedCacheCounterMutex`).

### 3.2 Sequence Construction

Features are chronologically ordered and concatenated into a combined sequence:

$$S = A_a \| D_d \| M_m$$

Where:
- $A$: API call sequence, length $a = 500$
- $D$: DLL sequence, length $d = 10$
- $M$: Mutex sequence, length $m = 10$
- Total sequence length: $520$

These parameters were determined through geometric mean analysis of the dataset feature distributions and validated through parameter analysis (Figures 4-5 in the paper).

### 3.3 CNN Architecture (XRan)

XRan employs a **2-layered CNN** architecture:

```
Input (520) -> Embedding ->
  Conv1D(128 filters, k=5) -> Sigmoid -> MaxPool(2) -> Dropout(0.3) ->
  Conv1D(64 filters, k=3) -> Sigmoid -> MaxPool(2) -> Dropout(0.3) ->
  Flatten -> Dense(128) -> Sigmoid -> Dense(1) -> Sigmoid
```

- **Loss function**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 10 (to prevent overfitting, per Fig. 6 in paper)
- **Activation**: Sigmoid throughout (suitable for binary classification)

### 3.4 Baseline Models

The following baselines are implemented for comparison:
- **Machine Learning**: Decision Tree, Random Forest, Naive Bayes, KNN (k=3), KNN (k=5)
- **Deep Learning**: Single-layer CNN, 2-layer LSTM

### 3.5 Explainability

**LIME (Local Interpretable Model-Agnostic Explanations):**
- Provides per-prediction explanations
- Identifies which features most influenced a specific classification
- Reveals both supporting and contradicting features

**SHAP (SHapley Additive exPlanations):**
- Provides global feature importance
- Calculates Shapley values representing average marginal contributions
- Generates location-based importance maps showing the significance of each position in the combined sequence

## 4. Dataset

### 4.1 Original Paper Datasets

| Dataset | Source | Type | Samples |
|---------|--------|------|---------|
| RD1 | VirusShare | Ransomware | 6,263 |
| RD2 | Sorel-20M | Ransomware | 7,703 |
| RD3 | ISOT | Ransomware | 668 |
| MD | VX Heaven | Malware | 6,263 |
| BD | Various | Benign | 14,797 |

### 4.2 Implementation Dataset

Our implementation supports two modes:

1. **Real Data Mode**: Processes Cuckoo Sandbox JSON reports from VirusShare or ISOT datasets. Users provide directories of reports organized by class.

2. **Synthetic Data Mode**: Generates a dataset that mimics the statistical properties of real ransomware/benign/malware samples based on the paper's LIME/SHAP analysis findings. Feature distributions are calibrated using the important features identified in the paper's explainability analysis (Section 4.4).

### 4.3 Data Split

Following the paper, data is split as:
- **Training**: 80%
- **Validation**: 10%
- **Testing**: 10%

10-fold stratified cross-validation is used for robust evaluation.

## 5. Experimental Results

### 5.1 Paper's Reported Results (Ransomware/Benign Classification)

| Method | Accuracy | TPR | FPR | F-Score |
|--------|----------|-----|-----|---------|
| Decision Tree | 0.912 | 0.943 | 0.115 | 0.913 |
| Random Forest | 0.964 | 0.963 | 0.036 | 0.964 |
| Naive Bayes | 0.680 | 0.700 | 0.325 | 0.687 |
| KNN (k=3) | 0.869 | 0.906 | 0.156 | 0.874 |
| KNN (k=5) | 0.865 | 0.903 | 0.159 | 0.871 |
| LSTM | 0.976 | 0.989 | 0.035 | 0.977 |
| CNN (1-layer) | 0.976 | 0.988 | 0.035 | 0.976 |
| **XRan (2L-CNN)** | **0.990** | **0.994** | **0.010** | **0.992** |

*Results for combined RD1+RD2+RD3+BD dataset*

### 5.2 Our Experimental Results

*(Results are populated after running `python main.py`)*

Results are saved to `results/results_table.csv` after execution, including:
- Per-model accuracy, TPR, FPR, F1-score, precision
- Cross-validation standard deviations
- Confusion matrices and ROC curves

### 5.3 Key Observations from Paper

1. **XRan outperforms all baselines** on the combined dataset, achieving 99.4% TPR with only 1.0% FPR.

2. **Sequence combination is crucial**: Using only API calls yields 85-88% accuracy. Adding DLLs and Mutexes in sequence form (not frequency) improves accuracy to 98-100%.

3. **Deep learning > ML for complex datasets**: For RD1 (most similar to benign), CNN/LSTM significantly outperform ML methods. For RD2/RD3 (more distinct), even Random Forest achieves near-perfect scores.

4. **DLLs and Mutexes are disproportionately important**: Despite occupying only 20 of 520 positions, DLL and Mutex features contribute more to the model's decisions than the 500 API call positions (per SHAP analysis).

5. **Ransomware vs. Malware is harder than Ransomware vs. Benign**: The malicious behavior overlap between ransomware and general malware makes discrimination more challenging.

## 6. Observations and Conclusions

### 6.1 Strengths of XRan

- **Multi-perspective feature fusion**: Combining API calls, DLLs, and Mutexes as concatenated sequences preserves both individual feature patterns and cross-feature correlations.
- **Hierarchical feature extraction**: The 2-layer CNN captures both low-level and high-level patterns through hierarchical convolution.
- **Explainability**: LIME and SHAP integration provides actionable insights for security analysts, identifying specific APIs, DLLs, and Mutexes that characterize ransomware behavior.

### 6.2 Limitations

- **Dynamic analysis overhead**: Cuckoo Sandbox analysis averages 155.3 seconds per sample, making real-time deployment challenging.
- **Evasion potential**: Sophisticated ransomware may detect sandbox environments and modify behavior.
- **Dataset dependency**: Performance varies across datasets (RD1 is harder to classify than RD2/RD3), suggesting sensitivity to the feature space overlap between classes.
- **Fixed sequence lengths**: The truncation of API calls to 500 may lose late-stage behavioral information for long-running samples.

### 6.3 Conclusions

XRan demonstrates that combining multiple dynamic analysis features in sequence form with a deep CNN architecture achieves state-of-the-art ransomware detection. The integration of XAI models provides valuable insights:

- Ransomware-indicative features: `NtAllocateVirtualMemory`, `RegOpenKeyExW`, `Advapi32.dll`, `Crypt32.dll`, `ZonesCounterMutex`
- Benign-indicative features: `LoadStringA/W`, `Kernel32.dll`, `Version.dll`
- DLLs and Mutexes provide stronger discriminative signals per feature than API calls

The study validates that multi-view dynamic analysis combined with deep learning and explainability is a promising direction for next-generation ransomware detection systems.

## References

1. Gulmez, S., Gorgulu Kakisim, A., & Sogukpinar, I. (2024). XRan: Explainable deep learning-based ransomware detection using dynamic analysis. *Computers & Security*, 139, 103703.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
4. Sgandurra, D., et al. (2016). Automated dynamic analysis of ransomware. *Journal in Computer Virology and Hacking Techniques*.
5. Jethva, B., et al. (2020). Multilayer ransomware detection using grouped registry key operations, file entropy and file signature monitoring. *Journal of Computer Security*.
