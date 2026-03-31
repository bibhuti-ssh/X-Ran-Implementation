"""
XRan Demo Script
Run this for a quick demonstration of the full pipeline.
Equivalent to the Jupyter notebook walkthrough.

Usage: python notebooks/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data_preprocessing import generate_synthetic_dataset, get_feature_names, TOTAL_SEQ_LEN
from src.models import XRanCNN, SingleLayerCNN, LSTMDetector
from src.train import train_ml_baselines, train_single_dl_model, predict_dl_model
from src.evaluate import compute_metrics, generate_full_report
from src.explain import explain_with_lime, explain_with_shap


def main():
    print("=" * 70)
    print("  XRan: Explainable Deep Learning-based Ransomware Detection")
    print("  Demo with Synthetic Dataset")
    print("=" * 70)

    # Setup
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate data
    print("\n[Step 1] Generating synthetic dataset...")
    X, y, api_vocab, dll_vocab, mutex_vocab = generate_synthetic_dataset(
        n_ransomware=600, n_benign=600, n_malware=300, seed=seed
    )
    total_vocab = len(api_vocab) + len(dll_vocab) + len(mutex_vocab)
    feature_names = get_feature_names(api_vocab, dll_vocab, mutex_vocab)

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, random_state=seed, stratify=y_train_val
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 2: ML Baselines
    print("\n[Step 2] Training ML baselines...")
    ml_results = train_ml_baselines(X_train, y_train, X_test, y_test)

    # Step 3: Deep Learning Models
    print("\n[Step 3] Training deep learning models...")
    all_results = dict(ml_results)

    from torch.utils.data import DataLoader, TensorDataset

    dl_configs = [
        ('LSTM', LSTMDetector, {'vocab_size': total_vocab}),
        ('CNN', SingleLayerCNN, {'vocab_size': total_vocab, 'seq_len': TOTAL_SEQ_LEN}),
        ('XRan (2L-CNN)', XRanCNN, {'vocab_size': total_vocab, 'seq_len': TOTAL_SEQ_LEN}),
    ]

    xran_model = None
    for name, cls, kwargs in dl_configs:
        model, history = train_single_dl_model(
            X_train, y_train, X_val, y_val,
            cls, kwargs, device,
            epochs=10, batch_size=64, model_name=name
        )
        test_ds = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        y_pred, y_prob = predict_dl_model(model, test_loader, device)
        metrics = compute_metrics(y_test, y_pred)
        all_results[name] = metrics

        if 'XRan' in name:
            xran_model = model

        print(f"  {name}: ACC={metrics['accuracy']:.4f} TPR={metrics['tpr']:.4f} "
              f"FPR={metrics['fpr']:.4f} F1={metrics['f1']:.4f}")

    # Step 4: Generate report
    print("\n[Step 4] Generating evaluation report...")
    df = generate_full_report(all_results, 'Ransomware Detection', output_dir)

    # Step 5: XAI
    print("\n[Step 5] Generating XAI explanations...")
    if xran_model:
        xai_dir = os.path.join(output_dir, 'xai')
        explain_with_lime(
            xran_model, X_train, X_test[:20], y_test[:20],
            feature_names, device,
            save_dir=os.path.join(xai_dir, 'lime'), n_samples=4
        )
        explain_with_shap(
            xran_model, X_train[:100], X_test[:30],
            feature_names, device,
            save_dir=os.path.join(xai_dir, 'shap'), n_background=30
        )

    print("\n" + "=" * 70)
    print("  Demo complete! Results saved to:", output_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
