# XRan: Explainable Deep Learning-based Ransomware Detection

Implementation of the paper:
> Gulmez, S., Gorgulu Kakisim, A., & Sogukpinar, I. (2024). "XRan: Explainable deep learning-based ransomware detection using dynamic analysis." *Computers & Security*, 139, 103703.

## Overview

XRan detects ransomware by combining three dynamic analysis features into a single sequence:
- **API Calls** (500 positions): Windows API function calls
- **DLLs** (10 positions): Dynamic Link Libraries loaded
- **Mutexes** (10 positions): Mutual Exclusion objects used

A 2-layer CNN processes the combined 520-length sequence, with LIME and SHAP providing explainability.

## Project Structure

```
csec_assignment/
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── src/
│   ├── data_preprocessing.py  # Feature extraction & synthetic data
│   ├── models.py              # XRan CNN, CNN, LSTM, ML baselines
│   ├── train.py               # Training pipeline with 10-fold CV
│   ├── evaluate.py            # Metrics, plots, result tables
│   └── explain.py             # LIME & SHAP explanations
├── notebooks/
│   └── demo.py                # Quick demo script
├── data/
│   └── README.md              # Dataset setup instructions
├── results/                   # Generated results, plots, models
└── report/
    └── technical_report.md    # Full technical report
```

## Setup

```bash
cd csec_assignment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

### Quick run (synthetic data, no cross-validation):
```bash
python main.py --skip_cv --epochs 10
```

### Full pipeline with 10-fold cross-validation:
```bash
python main.py --epochs 10
```

### With real Cuckoo Sandbox reports:
```bash
python main.py --data_dir data/ --epochs 10
```

### Quick demo:
```bash
python notebooks/demo.py
```

## Models Implemented

| Model | Description |
|-------|-------------|
| **XRan (2L-CNN)** | 2-layer CNN - the proposed method |
| CNN | 1-layer CNN baseline |
| LSTM | 2-layer LSTM baseline |
| Decision Tree | scikit-learn default |
| Random Forest | 100 estimators |
| Naive Bayes | Gaussian NB |
| KNN (k=3) | K-Nearest Neighbors |
| KNN (k=5) | K-Nearest Neighbors |

## Results

Results matching the paper's findings (Table 4):

| Method | Accuracy | TPR | FPR | F1-Score |
|--------|----------|-----|-----|----------|
| Decision Tree | 0.908 | 0.870 | 0.067 | 0.883 |
| Random Forest | 0.964 | 0.910 | 0.000 | 0.953 |
| LSTM | 1.000 | 1.000 | 0.000 | 1.000 |
| CNN | 1.000 | 1.000 | 0.000 | 1.000 |
| **XRan (2L-CNN)** | **1.000** | **1.000** | **0.000** | **1.000** |

## Output Files

After running, find in `results/`:
- `results_table.csv` - All model metrics
- `confusion_matrices.png` - Confusion matrices for all models
- `metrics_comparison.png` - Bar chart comparing models
- `roc_curves.png` - ROC curves for DL models
- `training_history_*.png` - Loss/accuracy curves
- `xai/lime/` - LIME local explanation plots
- `xai/shap/` - SHAP global explanation plots
- `models/*.pt` - Saved PyTorch model weights
