"""
Training Pipeline for XRan
Implements training for deep learning models with 10-fold cross-validation,
and training for ML baselines, as described in Section 4.1 of the paper.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .models import XRanCNN, SingleLayerCNN, LSTMDetector, get_ml_models
from .evaluate import compute_metrics


def train_dl_model(model, train_loader, val_loader, device,
                   epochs=10, lr=0.001, model_name='model'):
    """
    Train a deep learning model using Binary CrossEntropy and Adam optimizer.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: torch device.
        epochs: Number of training epochs (default 10 as per paper).
        lr: Learning rate.
        model_name: Name for logging.

    Returns:
        Dictionary with training history (loss, accuracy per epoch).
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = (outputs >= 0.5).long()
            train_correct += (preds == y_batch.long()).sum().item()
            train_total += X_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                preds = (outputs >= 0.5).long()
                val_correct += (preds == y_batch.long()).sum().item()
                val_total += X_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    return history


def predict_dl_model(model, data_loader, device):
    """Get predictions from a deep learning model."""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy()
            preds = (outputs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_probs)


def run_cross_validation(X, y, model_class, model_kwargs, device,
                         n_folds=10, epochs=10, batch_size=64, lr=0.001,
                         model_name='model'):
    """
    Run 10-fold cross-validation for a deep learning model.
    Dataset split: 80% train, 10% val, 10% test (per fold).

    Args:
        X: Input sequences, shape (n_samples, seq_len).
        y: Labels, shape (n_samples,).
        model_class: Class of the model to instantiate.
        model_kwargs: Keyword arguments for model constructor.
        device: torch device.
        n_folds: Number of CV folds.
        epochs: Training epochs per fold.
        batch_size: Batch size.
        lr: Learning rate.
        model_name: Name for logging.

    Returns:
        Dictionary with per-fold and averaged metrics.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n{'='*60}")
    print(f"Running {n_folds}-fold CV for {model_name}")
    print(f"{'='*60}")

    dataset = TensorDataset(torch.LongTensor(X), torch.LongTensor(y))

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")

        # Split train_val into train (80%) and val (10%)
        n_train = int(len(train_val_idx) * 0.889)  # 80/90 of train_val = 80% of total
        train_idx = train_val_idx[:n_train]
        val_idx = train_val_idx[n_train:]

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx),
                                batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_idx),
                                 batch_size=batch_size, shuffle=False)

        # Initialize fresh model for each fold
        model = model_class(**model_kwargs).to(device)

        # Train
        history = train_dl_model(model, train_loader, val_loader, device,
                                 epochs=epochs, lr=lr, model_name=model_name)

        # Test
        y_test = y[test_idx]
        y_pred, y_prob = predict_dl_model(model, test_loader, device)

        metrics = compute_metrics(y_test, y_pred)
        metrics['history'] = history
        fold_results.append(metrics)

        print(f"  Fold {fold+1} Test -> "
              f"ACC: {metrics['accuracy']:.4f} TPR: {metrics['tpr']:.4f} "
              f"FPR: {metrics['fpr']:.4f} F1: {metrics['f1']:.4f}")

    # Average results across folds
    avg_metrics = {}
    metric_keys = ['accuracy', 'tpr', 'fpr', 'f1', 'precision', 'recall']
    for key in metric_keys:
        values = [r[key] for r in fold_results]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)

    avg_metrics['fold_results'] = fold_results

    print(f"\n{model_name} Average Results:")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f} (+/- {avg_metrics['accuracy_std']:.4f})")
    print(f"  TPR:      {avg_metrics['tpr']:.4f} (+/- {avg_metrics['tpr_std']:.4f})")
    print(f"  FPR:      {avg_metrics['fpr']:.4f} (+/- {avg_metrics['fpr_std']:.4f})")
    print(f"  F1:       {avg_metrics['f1']:.4f} (+/- {avg_metrics['f1_std']:.4f})")

    return avg_metrics


def train_ml_baselines(X_train, y_train, X_test, y_test):
    """
    Train and evaluate scikit-learn baseline models.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        Dictionary of model_name -> metrics.
    """
    models = get_ml_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        elapsed = time.time() - start_time
        metrics = compute_metrics(y_test, y_pred)
        metrics['train_time'] = elapsed

        results[name] = metrics
        print(f"  {name} -> ACC: {metrics['accuracy']:.4f} "
              f"TPR: {metrics['tpr']:.4f} FPR: {metrics['fpr']:.4f} "
              f"F1: {metrics['f1']:.4f} (Time: {elapsed:.2f}s)")

    return results


def train_single_dl_model(X_train, y_train, X_val, y_val,
                          model_class, model_kwargs, device,
                          epochs=10, batch_size=64, lr=0.001,
                          model_name='model'):
    """
    Train a single deep learning model (no cross-validation).
    Used for final model training and XAI explanation generation.

    Returns:
        Trained model, training history.
    """
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model_class(**model_kwargs).to(device)

    print(f"\nTraining {model_name}...")
    history = train_dl_model(model, train_loader, val_loader, device,
                             epochs=epochs, lr=lr, model_name=model_name)

    return model, history
