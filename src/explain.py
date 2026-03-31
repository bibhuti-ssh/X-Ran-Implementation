"""
Explainability Module for XRan
Implements LIME and SHAP explanations as described in Sections 3.3 and 4.4.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ModelWrapper:
    """Wraps a PyTorch model to provide a sklearn-like predict_proba interface."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_proba(self, X):
        """Return probability estimates for both classes."""
        self.model.eval()
        if isinstance(X, np.ndarray):
            X = torch.LongTensor(X.astype(np.int64))
        with torch.no_grad():
            X = X.to(self.device)
            probs = self.model(X).cpu().numpy()
        # Return [P(benign), P(ransomware)]
        return np.column_stack([1 - probs, probs])


def explain_with_lime(model, X_train, X_explain, y_explain,
                      feature_names, device, save_dir, n_samples=5):
    """
    Generate LIME explanations for randomly selected samples.
    LIME provides per-prediction (local) explanations as described in Section 4.4.1.

    Args:
        model: Trained PyTorch model.
        X_train: Training data for building LIME explainer.
        X_explain: Samples to explain.
        y_explain: True labels for the samples.
        feature_names: List of feature position names.
        device: torch device.
        save_dir: Directory to save explanation plots.
        n_samples: Number of samples to explain.
    """
    if not HAS_LIME:
        print("LIME not installed. Skipping LIME explanations.")
        return

    os.makedirs(save_dir, exist_ok=True)
    wrapper = ModelWrapper(model, device)

    # Build LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.astype(np.float64),
        feature_names=feature_names,
        class_names=['Benign', 'Ransomware'],
        mode='classification',
        discretize_continuous=False
    )

    # Select samples to explain (mix of correct and incorrect predictions)
    preds_proba = wrapper.predict_proba(X_explain)
    preds = (preds_proba[:, 1] >= 0.5).astype(int)

    # Find correctly and incorrectly classified samples
    correct_mask = preds == y_explain
    incorrect_mask = ~correct_mask

    sample_indices = []
    # Add correct predictions
    correct_idx = np.where(correct_mask)[0]
    if len(correct_idx) > 0:
        n_correct = min(n_samples // 2 + 1, len(correct_idx))
        sample_indices.extend(np.random.choice(correct_idx, n_correct, replace=False))

    # Add incorrect predictions
    incorrect_idx = np.where(incorrect_mask)[0]
    if len(incorrect_idx) > 0:
        n_incorrect = min(n_samples - len(sample_indices), len(incorrect_idx))
        sample_indices.extend(np.random.choice(incorrect_idx, n_incorrect, replace=False))

    # Fill remaining with random samples
    while len(sample_indices) < n_samples and len(sample_indices) < len(X_explain):
        idx = np.random.randint(len(X_explain))
        if idx not in sample_indices:
            sample_indices.append(idx)

    print(f"\nGenerating LIME explanations for {len(sample_indices)} samples...")

    for i, idx in enumerate(sample_indices):
        sample = X_explain[idx].astype(np.float64)
        true_label = y_explain[idx]
        pred_label = preds[idx]
        confidence = preds_proba[idx, pred_label]
        is_correct = "Correct" if true_label == pred_label else "Misclassified"

        exp = explainer.explain_instance(
            sample,
            wrapper.predict_proba,
            num_features=15,
            num_samples=500
        )

        # Save explanation plot
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(12, 6)
        title = (f"LIME Explanation - Case {i+1} ({is_correct})\n"
                 f"True: {'Ransomware' if true_label == 1 else 'Benign'}, "
                 f"Predicted: {'Ransomware' if pred_label == 1 else 'Benign'} "
                 f"(Confidence: {confidence:.3f})")
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'lime_case_{i+1}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Case {i+1}: True={true_label}, Pred={pred_label} "
              f"({is_correct}, conf={confidence:.3f})")

    print(f"LIME explanations saved to {save_dir}")


def explain_with_shap(model, X_train, X_explain, feature_names,
                      device, save_dir, n_background=100):
    """
    Generate SHAP explanations for global feature importance.
    SHAP provides global explanations as described in Section 4.4.2.

    Args:
        model: Trained PyTorch model.
        X_train: Training data for background distribution.
        X_explain: Samples to explain.
        feature_names: List of feature position names.
        device: torch device.
        save_dir: Directory to save explanation plots.
        n_background: Number of background samples for KernelExplainer.
    """
    if not HAS_SHAP:
        print("SHAP not installed. Skipping SHAP explanations.")
        return

    os.makedirs(save_dir, exist_ok=True)
    wrapper = ModelWrapper(model, device)

    # Use a subset as background for efficiency
    bg_indices = np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)
    background = X_train[bg_indices].astype(np.float64)

    # Use KernelExplainer (model-agnostic)
    predict_fn = lambda x: wrapper.predict_proba(x)[:, 1]

    print(f"\nComputing SHAP values (this may take a few minutes)...")
    explainer = shap.KernelExplainer(predict_fn, background)

    # Explain a subset of samples
    n_explain = min(50, len(X_explain))
    explain_indices = np.random.choice(len(X_explain), n_explain, replace=False)
    X_subset = X_explain[explain_indices].astype(np.float64)

    shap_values = explainer.shap_values(X_subset, nsamples=100)

    # 1. Global feature importance (bar plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Get top 20 most important features
    top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    ax.barh(range(len(top_names)), top_values[::-1], color='steelblue')
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Global Feature Importance (SHAP)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_global_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Location-based importance (as in Fig. 11 & 12 of the paper)
    fig, ax = plt.subplots(figsize=(14, 5))
    position_importance = np.abs(shap_values).mean(axis=0)

    # Color-code by feature type: API (blue), DLL (orange), Mutex (green)
    colors = ['#4472C4'] * 500 + ['#ED7D31'] * 10 + ['#70AD47'] * 10
    ax.bar(range(len(position_importance)), position_importance, color=colors, width=1.0)
    ax.axvline(x=500, color='red', linestyle='--', alpha=0.7, label='DLL start')
    ax.axvline(x=510, color='green', linestyle='--', alpha=0.7, label='Mutex start')
    ax.set_xlabel('Feature Position in Combined Sequence')
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title('Location-based Global Explanation (SHAP)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_location_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Summary: importance by feature type
    api_importance = position_importance[:500].mean()
    dll_importance = position_importance[500:510].mean()
    mutex_importance = position_importance[510:520].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    feature_types = ['API Calls\n(positions 0-499)', 'DLLs\n(positions 500-509)',
                     'Mutexes\n(positions 510-519)']
    importances = [api_importance, dll_importance, mutex_importance]
    bars = ax.bar(feature_types, importances, color=['#4472C4', '#ED7D31', '#70AD47'])
    for bar, val in zip(bars, importances):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Mean |SHAP value| per position')
    ax.set_title('Average Feature Importance by Type')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_feature_type_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSHAP Feature Type Importance:")
    print(f"  API Calls (avg): {api_importance:.4f}")
    print(f"  DLLs (avg):      {dll_importance:.4f}")
    print(f"  Mutexes (avg):   {mutex_importance:.4f}")
    print(f"SHAP explanations saved to {save_dir}")

    return shap_values
