"""
Data Preprocessing Module for XRan
Extracts API calls, DLLs, and Mutexes from Cuckoo Sandbox JSON reports
and builds combined feature sequences as described in the paper.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional


# Sequence lengths as defined in the paper (Section 4.3)
API_SEQ_LEN = 500
DLL_SEQ_LEN = 10
MUTEX_SEQ_LEN = 10
TOTAL_SEQ_LEN = API_SEQ_LEN + DLL_SEQ_LEN + MUTEX_SEQ_LEN  # 520


def extract_features_from_cuckoo_report(report_path: str) -> Dict[str, List[str]]:
    """
    Extract API calls, DLLs, and Mutexes from a Cuckoo Sandbox JSON report.

    Args:
        report_path: Path to the Cuckoo JSON report file.

    Returns:
        Dictionary with keys 'api_calls', 'dlls', 'mutexes' each mapping
        to a chronologically ordered list of feature strings.
    """
    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        report = json.load(f)

    api_calls = []
    dlls = []
    mutexes = []

    # Extract API calls from behavior -> processes -> calls
    behavior = report.get("behavior", {})
    processes = behavior.get("processes", [])
    for process in processes:
        calls = process.get("calls", [])
        for call in calls:
            api_name = call.get("api", "")
            if api_name:
                api_calls.append(api_name)

    # Extract DLLs from behavior -> processes -> modules or summary
    for process in processes:
        modules = process.get("modules", [])
        for module in modules:
            dll_name = module.get("dll", "") or module.get("name", "")
            if dll_name:
                dlls.append(dll_name)

    # Also check behavior summary for DLLs
    summary = behavior.get("summary", {})
    loaded_dlls = summary.get("dll_loaded", [])
    if loaded_dlls and not dlls:
        dlls = loaded_dlls

    # Extract Mutexes from behavior -> summary -> mutex
    mutex_list = summary.get("mutex", [])
    if mutex_list:
        mutexes = mutex_list

    return {"api_calls": api_calls, "dlls": dlls, "mutexes": mutexes}


def build_vocabulary(
    feature_lists: List[List[str]], min_freq: int = 1
) -> Dict[str, int]:
    """
    Build a vocabulary mapping from feature strings to integer indices.
    Index 0 is reserved for padding.

    Args:
        feature_lists: List of feature sequences from all samples.
        min_freq: Minimum frequency for a feature to be included.

    Returns:
        Dictionary mapping feature string to integer index.
    """
    counter = Counter()
    for features in feature_lists:
        counter.update(features)

    vocab = {"<PAD>": 0}
    idx = 1
    for feature, count in counter.most_common():
        if count >= min_freq:
            vocab[feature] = idx
            idx += 1

    return vocab


def encode_sequence(
    features: List[str], vocab: Dict[str, int], max_len: int
) -> np.ndarray:
    """
    Encode a feature sequence to integer indices and pad/truncate to max_len.

    Args:
        features: List of feature strings.
        vocab: Vocabulary mapping.
        max_len: Target sequence length.

    Returns:
        Numpy array of shape (max_len,) with integer indices.
    """
    encoded = []
    for f in features[:max_len]:
        encoded.append(vocab.get(f, 0))

    # Pad if shorter than max_len
    while len(encoded) < max_len:
        encoded.append(0)

    return np.array(encoded, dtype=np.int64)


def build_combined_sequence(
    api_calls: List[str],
    dlls: List[str],
    mutexes: List[str],
    api_vocab: Dict[str, int],
    dll_vocab: Dict[str, int],
    mutex_vocab: Dict[str, int],
) -> np.ndarray:
    """
    Build the combined sequence A_a || D_d || M_m as described in the paper.
    DLL and Mutex indices are offset so they occupy a separate embedding space.

    Args:
        api_calls, dlls, mutexes: Feature lists for a sample.
        api_vocab, dll_vocab, mutex_vocab: Vocabularies.

    Returns:
        Numpy array of shape (520,) with combined encoded sequence.
    """
    api_seq = encode_sequence(api_calls, api_vocab, API_SEQ_LEN)
    dll_seq = encode_sequence(dlls, dll_vocab, DLL_SEQ_LEN)
    mutex_seq = encode_sequence(mutexes, mutex_vocab, MUTEX_SEQ_LEN)

    # Offset DLL and Mutex indices so they don't collide with API indices
    dll_offset = len(api_vocab)
    mutex_offset = len(api_vocab) + len(dll_vocab)

    dll_seq_offset = np.where(dll_seq > 0, dll_seq + dll_offset, 0)
    mutex_seq_offset = np.where(mutex_seq > 0, mutex_seq + mutex_offset, 0)

    combined = np.concatenate([api_seq, dll_seq_offset, mutex_seq_offset])
    return combined


def process_dataset_from_reports(
    report_dirs: Dict[str, str], labels: Dict[str, int], save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, Dict]:
    """
    Process all Cuckoo reports from directories and build the dataset.

    Args:
        report_dirs: Dict mapping label_name -> directory of JSON reports.
        labels: Dict mapping label_name -> integer label (1=ransomware, 0=benign).
        save_path: Optional path to save processed data.

    Returns:
        X (sequences), y (labels), and the three vocabularies.
    """
    all_api_calls = []
    all_dlls = []
    all_mutexes = []
    all_labels = []
    all_features = []

    for label_name, report_dir in report_dirs.items():
        label = labels[label_name]
        report_files = [f for f in os.listdir(report_dir) if f.endswith(".json")]

        for rf in report_files:
            report_path = os.path.join(report_dir, rf)
            try:
                features = extract_features_from_cuckoo_report(report_path)
                all_features.append(features)
                all_api_calls.append(features["api_calls"])
                all_dlls.append(features["dlls"])
                all_mutexes.append(features["mutexes"])
                all_labels.append(label)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping {report_path}: {e}")
                continue

    # Build vocabularies
    api_vocab = build_vocabulary(all_api_calls)
    dll_vocab = build_vocabulary(all_dlls)
    mutex_vocab = build_vocabulary(all_mutexes)

    # Build combined sequences
    X = np.zeros((len(all_features), TOTAL_SEQ_LEN), dtype=np.int64)
    for i, features in enumerate(all_features):
        X[i] = build_combined_sequence(
            features["api_calls"],
            features["dlls"],
            features["mutexes"],
            api_vocab,
            dll_vocab,
            mutex_vocab,
        )

    y = np.array(all_labels, dtype=np.int64)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            "X": X,
            "y": y,
            "api_vocab": api_vocab,
            "dll_vocab": dll_vocab,
            "mutex_vocab": mutex_vocab,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {save_path}")

    return X, y, api_vocab, dll_vocab, mutex_vocab


def decode_sequence(sequence, api_vocab, dll_vocab, mutex_vocab):
    """Decode a combined integer sequence back to feature names."""
    api_inv = {v: k for k, v in api_vocab.items()}
    dll_offset = len(api_vocab)
    dll_inv = {v: k for k, v in dll_vocab.items()}
    mutex_offset = dll_offset + len(dll_vocab)
    mutex_inv = {v: k for k, v in mutex_vocab.items()}

    decoded = []
    for i, val in enumerate(sequence):
        if val == 0:
            decoded.append("<PAD>")
        elif i < API_SEQ_LEN:
            decoded.append(api_inv.get(val, f"UNK_API_{val}"))
        elif i < API_SEQ_LEN + DLL_SEQ_LEN:
            decoded.append(dll_inv.get(val - dll_offset, f"UNK_DLL_{val}"))
        else:
            decoded.append(mutex_inv.get(val - mutex_offset, f"UNK_MUTEX_{val}"))

    return decoded
