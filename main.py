"""
XRan: Explainable Deep Learning-based Ransomware Detection
Main entry point for training, evaluation, and explanation generation.

Paper: "XRan: Explainable deep learning-based ransomware detection using dynamic analysis"
       Gulmez et al., Computers & Security, 2024

Usage:
    python main.py                          # Full pipeline with synthetic data
    python main.py --mode train             # Train all models
    python main.py --mode evaluate          # Evaluate trained models
    python main.py --mode explain           # Generate LIME/SHAP explanations
    python main.py --data_dir /path/to/data # Use real Cuckoo reports
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    generate_synthetic_dataset,
    process_dataset_from_reports,
    get_feature_names,
    TOTAL_SEQ_LEN
)
from src.models import XRanCNN, SingleLayerCNN, LSTMDetector
from src.train import (
    run_cross_validation,
    train_ml_baselines,
    train_single_dl_model,
    predict_dl_model
)
from src.evaluate import (
    compute_metrics,
    create_results_table,
    plot_confusion_matrices,
    plot_training_history,
    plot_metrics_comparison,
    plot_roc_curves,
    generate_full_report
)
from src.explain import explain_with_lime, explain_with_shap


def parse_args():
    parser = argparse.ArgumentParser(description='XRan Ransomware Detection')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'train', 'evaluate', 'explain'],
                        help='Pipeline mode')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with Cuckoo JSON reports. '
                             'Expected subdirs: ransomware/, benign/, malware/')
    parser.add_argument('--n_ransomware', type=int, default=1000,
                        help='Number of synthetic ransomware samples')
    parser.add_argument('--n_benign', type=int, default=1000,
                        help='Number of synthetic benign samples')
    parser.add_argument('--n_malware', type=int, default=500,
                        help='Number of synthetic malware samples')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs (paper uses 10)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of CV folds')
    parser.add_argument('--skip_cv', action='store_true',
                        help='Skip cross-validation (faster, single split)')
    parser.add_argument('--skip_xai', action='store_true',
                        help='Skip XAI explanations')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def setup_device():
    """Set up computation device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_or_generate_data(args):
    """Load real data from Cuckoo reports or generate synthetic data."""
    cache_path = os.path.join(args.output_dir, 'processed_data.pkl')

    if args.data_dir and os.path.isdir(args.data_dir):
        print(f"\nLoading data from Cuckoo reports in {args.data_dir}...")
        report_dirs = {}
        labels = {}

        ransomware_dir = os.path.join(args.data_dir, 'ransomware')
        benign_dir = os.path.join(args.data_dir, 'benign')
        malware_dir = os.path.join(args.data_dir, 'malware')

        if os.path.isdir(ransomware_dir):
            report_dirs['ransomware'] = ransomware_dir
            labels['ransomware'] = 1
        if os.path.isdir(benign_dir):
            report_dirs['benign'] = benign_dir
            labels['benign'] = 0
        if os.path.isdir(malware_dir):
            report_dirs['malware'] = malware_dir
            labels['malware'] = 0

        X, y, api_vocab, dll_vocab, mutex_vocab = process_dataset_from_reports(
            report_dirs, labels, save_path=cache_path
        )
    elif os.path.exists(cache_path):
        print(f"\nLoading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        X, y = data['X'], data['y']
        api_vocab, dll_vocab, mutex_vocab = data['api_vocab'], data['dll_vocab'], data['mutex_vocab']
    else:
        print("\nGenerating synthetic dataset...")
        print("(To use real data, provide --data_dir with Cuckoo report subdirectories)")
        X, y, api_vocab, dll_vocab, mutex_vocab = generate_synthetic_dataset(
            n_ransomware=args.n_ransomware,
            n_benign=args.n_benign,
            n_malware=args.n_malware,
            seed=args.seed
        )

    return X, y, api_vocab, dll_vocab, mutex_vocab


def run_full_pipeline(args):
    """Run the complete XRan pipeline: data -> train -> evaluate -> explain."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================
    # Step 1: Load/Generate Data
    # =========================================================
    X, y, api_vocab, dll_vocab, mutex_vocab = load_or_generate_data(args)
    total_vocab_size = len(api_vocab) + len(dll_vocab) + len(mutex_vocab)
    feature_names = get_feature_names(api_vocab, dll_vocab, mutex_vocab)

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: Ransomware={np.sum(y==1)}, Non-ransomware={np.sum(y==0)}")
    print(f"Total vocabulary size: {total_vocab_size}")

    # Split: 80% train, 10% val, 10% test (as per paper)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111,  # 10% of total
        random_state=args.seed, stratify=y_train_val
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    all_results = {}
    all_probs = {}

    # =========================================================
    # Step 2: Train and Evaluate ML Baselines
    # =========================================================
    print("\n" + "="*60)
    print("TRAINING MACHINE LEARNING BASELINES")
    print("="*60)

    ml_results = train_ml_baselines(X_train, y_train, X_test, y_test)
    all_results.update(ml_results)

    # =========================================================
    # Step 3: Train and Evaluate Deep Learning Models
    # =========================================================
    print("\n" + "="*60)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*60)

    dl_configs = [
        ('LSTM', LSTMDetector, {'vocab_size': total_vocab_size}),
        ('CNN', SingleLayerCNN, {'vocab_size': total_vocab_size, 'seq_len': TOTAL_SEQ_LEN}),
        ('XRan (2L-CNN)', XRanCNN, {'vocab_size': total_vocab_size, 'seq_len': TOTAL_SEQ_LEN}),
    ]

    trained_models = {}

    if not args.skip_cv:
        # 10-fold cross-validation
        for name, model_class, kwargs in dl_configs:
            cv_results = run_cross_validation(
                X, y, model_class, kwargs, device,
                n_folds=args.n_folds, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                model_name=name
            )
            all_results[name] = cv_results

    # Train final models on the train set for evaluation and XAI
    for name, model_class, kwargs in dl_configs:
        model, history = train_single_dl_model(
            X_train, y_train, X_val, y_val,
            model_class, kwargs, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, model_name=name
        )
        trained_models[name] = model

        # Plot training history
        plot_training_history(
            history, model_name=name,
            save_path=os.path.join(args.output_dir, f'training_history_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
        )

        # Evaluate on test set
        from torch.utils.data import DataLoader, TensorDataset
        test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        y_pred, y_prob = predict_dl_model(model, test_loader, device)
        metrics = compute_metrics(y_test, y_pred)

        if args.skip_cv:
            all_results[name] = metrics

        all_probs[name] = y_prob

        print(f"\n{name} Test Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  TPR: {metrics['tpr']:.4f}, FPR: {metrics['fpr']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}")

    # Save models
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    for name, model in trained_models.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        torch.save(model.state_dict(), os.path.join(models_dir, f'{safe_name}.pt'))
    print(f"\nModels saved to {models_dir}")

    # =========================================================
    # Step 4: Generate Evaluation Report
    # =========================================================
    print("\n" + "="*60)
    print("GENERATING EVALUATION REPORT")
    print("="*60)

    df = generate_full_report(all_results, 'Ransomware vs Benign+Malware', args.output_dir)

    # ROC curves for DL models
    if all_probs:
        plot_roc_curves(all_probs, y_test,
                        save_path=os.path.join(args.output_dir, 'roc_curves.png'))

    # =========================================================
    # Step 5: XAI Explanations
    # =========================================================
    if not args.skip_xai and 'XRan (2L-CNN)' in trained_models:
        print("\n" + "="*60)
        print("GENERATING XAI EXPLANATIONS")
        print("="*60)

        xran_model = trained_models['XRan (2L-CNN)']
        xai_dir = os.path.join(args.output_dir, 'xai')

        # LIME explanations (local)
        explain_with_lime(
            xran_model, X_train, X_test, y_test,
            feature_names, device,
            save_dir=os.path.join(xai_dir, 'lime'),
            n_samples=6
        )

        # SHAP explanations (global)
        explain_with_shap(
            xran_model, X_train, X_test,
            feature_names, device,
            save_dir=os.path.join(xai_dir, 'shap'),
            n_background=50
        )

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {os.path.abspath(args.output_dir)}/")
    print(f"  - results_table.csv")
    print(f"  - confusion_matrices.png")
    print(f"  - metrics_comparison.png")
    print(f"  - roc_curves.png")
    print(f"  - training_history_*.png")
    if not args.skip_xai:
        print(f"  - xai/lime/lime_case_*.png")
        print(f"  - xai/shap/shap_*.png")
    print(f"  - models/*.pt")

    return all_results


if __name__ == '__main__':
    args = parse_args()
    results = run_full_pipeline(args)
