"""
Evaluation Module for XRan
Computes accuracy, TPR, FPR, precision, recall, F1-score and generates
comparison tables and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


def compute_metrics(y_true, y_pred):
    """
    Compute all evaluation metrics used in the paper.

    Args:
        y_true: Ground truth labels (0=benign/malware, 1=ransomware).
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, TPR, FPR, precision, recall, F1-score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }


def create_results_table(all_results, dataset_name='Dataset'):
    """
    Create a results comparison table matching Table 4 from the paper.

    Args:
        all_results: Dictionary of model_name -> metrics dict.
        dataset_name: Name for the dataset column.

    Returns:
        pandas DataFrame with results.
    """
    rows = []
    for model_name, metrics in all_results.items():
        rows.append({
            'Dataset': dataset_name,
            'Detection Technique': model_name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'TPR': f"{metrics['tpr']:.3f}",
            'FPR': f"{metrics['fpr']:.3f}",
            'F-Score': f"{metrics['f1']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
        })

    df = pd.DataFrame(rows)
    return df


def plot_confusion_matrices(all_results, save_path=None):
    """Plot confusion matrices for all models."""
    n_models = len(all_results)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_name, metrics) in enumerate(all_results.items()):
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Ransomware'],
                    yticklabels=['Benign', 'Ransomware'],
                    ax=axes[idx])
        axes[idx].set_title(model_name, fontsize=10)
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrices to {save_path}")
    plt.close()


def plot_training_history(history, model_name='Model', save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    plt.close()


def plot_metrics_comparison(all_results, save_path=None):
    """Create bar chart comparing metrics across all models."""
    models = list(all_results.keys())
    metrics_to_plot = ['accuracy', 'tpr', 'fpr', 'f1']
    metric_labels = ['Accuracy', 'TPR', 'FPR', 'F1-Score']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(models))

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [all_results[m][metric] for m in models]
        bars = axes[idx].bar(range(len(models)), values, color=colors)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylim(0, 1.05)
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    plt.close()


def plot_roc_curves(all_results_with_probs, y_true, save_path=None):
    """
    Plot ROC curves for models that provide probability outputs.

    Args:
        all_results_with_probs: Dict of model_name -> y_prob array.
        y_true: True labels.
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("husl", len(all_results_with_probs))

    for idx, (model_name, y_prob) in enumerate(all_results_with_probs.items()):
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr_vals, tpr_vals)
        ax.plot(fpr_vals, tpr_vals, color=colors[idx],
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    plt.close()


def generate_full_report(all_results, dataset_name, save_dir):
    """Generate all evaluation plots and tables."""
    os.makedirs(save_dir, exist_ok=True)

    # Results table
    df = create_results_table(all_results, dataset_name)
    table_path = os.path.join(save_dir, 'results_table.csv')
    df.to_csv(table_path, index=False)
    print(f"\nResults Table:\n{df.to_string(index=False)}\n")

    # Confusion matrices
    plot_confusion_matrices(
        all_results,
        save_path=os.path.join(save_dir, 'confusion_matrices.png')
    )

    # Metrics comparison
    plot_metrics_comparison(
        all_results,
        save_path=os.path.join(save_dir, 'metrics_comparison.png')
    )

    return df
