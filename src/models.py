"""
Model Definitions for XRan
Implements the 2-layer CNN (XRan), 1-layer CNN, LSTM, and ML baselines.
Architecture follows Section 3.2 and Section 4.1 of the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


class XRanCNN(nn.Module):
    """
    XRan: 2-layered CNN architecture for ransomware detection.

    Architecture (from paper Section 4.1):
    - Embedding layer
    - Conv1D -> ReLU -> MaxPool -> Dropout (Layer 1)
    - Conv1D -> ReLU -> MaxPool -> Dropout (Layer 2)
    - Flatten -> Dense -> ReLU -> Dense -> Sigmoid output

    Uses Binary CrossEntropy loss with Adam optimizer.
    Sigmoid activation at the output layer for binary classification (Section 3.2).
    """

    def __init__(self, vocab_size, embedding_dim=64, seq_len=520,
                 num_filters1=128, num_filters2=64,
                 kernel_size1=5, kernel_size2=3,
                 pool_size=2, dropout_rate=0.3):
        super(XRanCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Layer 1: Conv1D -> ReLU -> MaxPool -> Dropout
        self.conv1 = nn.Conv1d(embedding_dim, num_filters1, kernel_size1, padding=kernel_size1 // 2)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2: Conv1D -> ReLU -> MaxPool -> Dropout
        self.conv2 = nn.Conv1d(num_filters1, num_filters2, kernel_size2, padding=kernel_size2 // 2)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened size
        conv_out_len = seq_len // (pool_size * pool_size)
        self.flat_size = num_filters2 * conv_out_len

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len) for Conv1d

        # Layer 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid output for binary classification

        return x.squeeze(-1)


class SingleLayerCNN(nn.Module):
    """
    1-layered CNN baseline for comparison.
    Same as XRan but with only one convolutional layer.
    """

    def __init__(self, vocab_size, embedding_dim=64, seq_len=520,
                 num_filters=128, kernel_size=5,
                 pool_size=2, dropout_rate=0.3):
        super(SingleLayerCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        conv_out_len = seq_len // pool_size
        self.flat_size = num_filters * conv_out_len

        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x.squeeze(-1)


class LSTMDetector(nn.Module):
    """
    LSTM-based ransomware detector for comparison.
    2-layered LSTM with the same activation and optimizer as CNN.
    """

    def __init__(self, vocab_size, embedding_dim=64,
                 hidden_dim=128, num_layers=2, dropout_rate=0.3):
        super(LSTMDetector, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out.squeeze(-1)


def get_ml_models():
    """
    Return dictionary of scikit-learn baseline models as defined in the paper.
    Uses default parameters as specified in Section 4.1.
    """
    return {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    }


def get_dl_models(vocab_size, seq_len=520, device='cpu'):
    """
    Return dictionary of deep learning models.

    Args:
        vocab_size: Total vocabulary size (API + DLL + Mutex).
        seq_len: Input sequence length (default 520).
        device: torch device.

    Returns:
        Dictionary of model_name -> model instance.
    """
    models = {
        'XRan (2L-CNN)': XRanCNN(vocab_size, seq_len=seq_len).to(device),
        'CNN': SingleLayerCNN(vocab_size, seq_len=seq_len).to(device),
        'LSTM': LSTMDetector(vocab_size).to(device),
    }
    return models
