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
    with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
        report = json.load(f)

    api_calls = []
    dlls = []
    mutexes = []

    # Extract API calls from behavior -> processes -> calls
    behavior = report.get('behavior', {})
    processes = behavior.get('processes', [])
    for process in processes:
        calls = process.get('calls', [])
        for call in calls:
            api_name = call.get('api', '')
            if api_name:
                api_calls.append(api_name)

    # Extract DLLs from behavior -> processes -> modules or summary
    for process in processes:
        modules = process.get('modules', [])
        for module in modules:
            dll_name = module.get('dll', '') or module.get('name', '')
            if dll_name:
                dlls.append(dll_name)

    # Also check behavior summary for DLLs
    summary = behavior.get('summary', {})
    loaded_dlls = summary.get('dll_loaded', [])
    if loaded_dlls and not dlls:
        dlls = loaded_dlls

    # Extract Mutexes from behavior -> summary -> mutex
    mutex_list = summary.get('mutex', [])
    if mutex_list:
        mutexes = mutex_list

    return {
        'api_calls': api_calls,
        'dlls': dlls,
        'mutexes': mutexes
    }


def build_vocabulary(feature_lists: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
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

    vocab = {'<PAD>': 0}
    idx = 1
    for feature, count in counter.most_common():
        if count >= min_freq:
            vocab[feature] = idx
            idx += 1

    return vocab


def encode_sequence(features: List[str], vocab: Dict[str, int], max_len: int) -> np.ndarray:
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
    mutex_vocab: Dict[str, int]
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
    report_dirs: Dict[str, str],
    labels: Dict[str, int],
    save_path: Optional[str] = None
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
        report_files = [f for f in os.listdir(report_dir) if f.endswith('.json')]

        for rf in report_files:
            report_path = os.path.join(report_dir, rf)
            try:
                features = extract_features_from_cuckoo_report(report_path)
                all_features.append(features)
                all_api_calls.append(features['api_calls'])
                all_dlls.append(features['dlls'])
                all_mutexes.append(features['mutexes'])
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
            features['api_calls'], features['dlls'], features['mutexes'],
            api_vocab, dll_vocab, mutex_vocab
        )

    y = np.array(all_labels, dtype=np.int64)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            'X': X, 'y': y,
            'api_vocab': api_vocab, 'dll_vocab': dll_vocab, 'mutex_vocab': mutex_vocab
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {save_path}")

    return X, y, api_vocab, dll_vocab, mutex_vocab


def generate_synthetic_dataset(
    n_ransomware: int = 1000,
    n_benign: int = 1000,
    n_malware: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, Dict]:
    """
    Generate a synthetic dataset that mimics the characteristics of the
    VirusShare ransomware dataset for development and testing purposes.

    This creates realistic feature distributions based on the paper's findings:
    - Ransomware: heavy use of NtAllocateVirtualMemory, RegOpenKeyExW,
      Advapi32.dll, encryption-related APIs, ZonesCounterMutex, etc.
    - Benign: LoadStringA/W, Kernel32.dll, Version.dll, standard APIs
    - Malware: Wininet.dll, NtCreateMutant, Sechost.dll, propagation APIs

    Returns:
        X, y, api_vocab, dll_vocab, mutex_vocab
    """
    rng = np.random.RandomState(seed)

    # Define realistic feature vocabularies based on paper's LIME/SHAP analysis
    ransomware_apis = [
        'NtAllocateVirtualMemory', 'RegOpenKeyExW', 'NtOpenProcess',
        'CryptEncrypt', 'CryptGenKey', 'CryptAcquireContextW',
        'NtWriteFile', 'NtReadFile', 'NtCreateFile', 'NtClose',
        'NtDeviceIoControlFile', 'NtQueryInformationFile',
        'NtSetInformationFile', 'NtQueryDirectoryFile',
        'CryptDestroyKey', 'CryptReleaseContext',
        'RegSetValueExW', 'RegQueryValueExW', 'RegCloseKey',
        'NtFreeVirtualMemory', 'NtProtectVirtualMemory',
        'LdrGetProcedureAddress', 'LdrLoadDll', 'NtOpenKey',
        'GetFileAttributesW', 'FindFirstFileExW', 'FindNextFileW',
        'MoveFileWithProgressW', 'DeleteFileW', 'CreateDirectoryW',
        'GetSystemDirectoryA', 'GetSystemDirectoryW',
    ]

    benign_apis = [
        'LoadStringA', 'LoadStringW', 'GetModuleHandleW',
        'GetProcAddress', 'GetLastError', 'SetLastError',
        'GetCurrentThreadId', 'GetCurrentProcessId',
        'OutputDebugStringA', 'IsDebuggerPresent',
        'GetTickCount', 'QueryPerformanceCounter',
        'GetSystemTimeAsFileTime', 'GetCommandLineW',
        'GetModuleFileNameW', 'GetStartupInfoW',
        'InitializeCriticalSection', 'EnterCriticalSection',
        'LeaveCriticalSection', 'DeleteCriticalSection',
        'HeapAlloc', 'HeapFree', 'HeapReAlloc',
        'VirtualAlloc', 'VirtualFree',
        'CreateFileW', 'ReadFile', 'WriteFile', 'CloseHandle',
        'GetFileSize', 'SetFilePointer',
    ]

    malware_apis = [
        'NtCreateMutant', 'InternetOpenA', 'InternetConnectA',
        'HttpOpenRequestA', 'HttpSendRequestA', 'InternetReadFile',
        'URLDownloadToFileW', 'WinExec', 'ShellExecuteW',
        'CreateRemoteThread', 'WriteProcessMemory',
        'OpenProcess', 'VirtualAllocEx', 'NtUnmapViewOfSection',
        'RegCreateKeyExW', 'RegDeleteKeyW',
        'CreateToolhelp32Snapshot', 'Process32FirstW', 'Process32NextW',
        'Thread32First', 'Thread32Next',
        'socket', 'connect', 'send', 'recv',
        'WSAStartup', 'gethostbyname', 'inet_addr',
        'CreateServiceW', 'StartServiceW',
    ]

    shared_apis = [
        'NtClose', 'NtCreateFile', 'NtReadFile', 'NtWriteFile',
        'GetSystemTimeAsFileTime', 'GetTickCount',
        'NtQueryAttributesFile', 'NtOpenFile',
        'GetFileAttributesW', 'RtlInitUnicodeString',
    ]

    all_apis = list(set(ransomware_apis + benign_apis + malware_apis + shared_apis))

    ransomware_dlls = [
        'Advapi32.dll', 'Oleaut32.dll', 'Comctl32.dll', 'Crypt32.dll',
        'Bcrypt.dll', 'Ncrypt.dll', 'ntdll.dll', 'Kernel32.dll',
        'User32.dll', 'Shell32.dll', 'Gdi32.dll', 'Ole32.dll',
    ]

    benign_dlls = [
        'Kernel32.dll', 'Version.dll', 'User32.dll', 'Gdi32.dll',
        'Comctl32.dll', 'Shell32.dll', 'Shlwapi.dll', 'Msvcrt.dll',
        'ntdll.dll', 'Rpcrt4.dll', 'Ws2_32.dll', 'Imm32.dll',
    ]

    malware_dlls = [
        'Wininet.dll', 'Sechost.dll', 'Urlmon.dll', 'Winhttp.dll',
        'ntdll.dll', 'Kernel32.dll', 'Advapi32.dll', 'User32.dll',
        'Crypt32.dll', 'Wtsapi32.dll', 'Netapi32.dll', 'Psapi.dll',
    ]

    all_dlls = list(set(ransomware_dlls + benign_dlls + malware_dlls))

    ransomware_mutexes = [
        'ZonesCounterMutex', 'ZonesLockedCacheCounterMutex',
        'ZonesLockedCacheMutex', 'Global\\CryptoMutex',
        'Local\\RansomLock', 'Global\\EncryptionSync',
        'DBWinMutex', 'ShimCacheMutex',
        '_SHuassist.mtx', 'MSCTF.Asm.MutexDefault1',
    ]

    benign_mutexes = [
        'MSCTF.Asm.MutexDefault1', 'DBWinMutex',
        'ShimCacheMutex', '_SHuassist.mtx',
        'Local\\ZonesCacheCounterMutex',
        'CTF.TimListCache.FMPDefaultS-1-5-21',
        'CTF.Compart.MutexDefaultS-1-5-21',
        'CTF.LBES.MutexDefaultS-1-5-21',
    ]

    malware_mutexes = [
        'Global\\MalwareMutex', 'NtCreateMutant_default',
        'DBWinMutex', 'ShimCacheMutex',
        'Global\\BotSync', 'Local\\PropagationLock',
        'MSCTF.Asm.MutexDefault1', '_SHuassist.mtx',
    ]

    all_mutexes = list(set(ransomware_mutexes + benign_mutexes + malware_mutexes))

    # Build vocabularies
    api_vocab = {'<PAD>': 0}
    for i, api in enumerate(all_apis):
        api_vocab[api] = i + 1

    dll_vocab = {'<PAD>': 0}
    for i, dll in enumerate(all_dlls):
        dll_vocab[dll] = i + 1

    mutex_vocab = {'<PAD>': 0}
    for i, mutex in enumerate(all_mutexes):
        mutex_vocab[mutex] = i + 1

    total_vocab_size = len(api_vocab) + len(dll_vocab) + len(mutex_vocab)

    def generate_sample(api_pool, api_weights, dll_pool, dll_weights,
                        mutex_pool, mutex_weights, noise_ratio=0.15):
        """Generate a single sample's combined sequence."""
        # Generate API call sequence
        n_api = rng.randint(100, API_SEQ_LEN)
        api_probs = np.array(api_weights, dtype=np.float64)
        api_probs /= api_probs.sum()
        api_indices = rng.choice(len(api_pool), size=n_api, p=api_probs)
        apis = [api_pool[i] for i in api_indices]

        # Add noise (shared APIs)
        n_noise = int(n_api * noise_ratio)
        noise_apis = [shared_apis[rng.randint(len(shared_apis))] for _ in range(n_noise)]
        insert_positions = sorted(rng.choice(n_api, size=min(n_noise, n_api), replace=False))
        for pos, noise_api in zip(insert_positions, noise_apis):
            apis[pos] = noise_api

        # Generate DLL sequence
        n_dll = rng.randint(3, DLL_SEQ_LEN + 1)
        dll_probs = np.array(dll_weights, dtype=np.float64)
        dll_probs /= dll_probs.sum()
        dll_indices = rng.choice(len(dll_pool), size=n_dll, p=dll_probs, replace=True)
        dlls = list(dict.fromkeys([dll_pool[i] for i in dll_indices]))  # unique, preserve order

        # Generate Mutex sequence
        n_mutex = rng.randint(1, MUTEX_SEQ_LEN + 1)
        mutex_probs = np.array(mutex_weights, dtype=np.float64)
        mutex_probs /= mutex_probs.sum()
        mutex_indices = rng.choice(len(mutex_pool), size=n_mutex, p=mutex_probs, replace=True)
        mutexes = list(dict.fromkeys([mutex_pool[i] for i in mutex_indices]))

        return build_combined_sequence(
            apis, dlls, mutexes,
            api_vocab, dll_vocab, mutex_vocab
        )

    # Generate samples
    n_total = n_ransomware + n_benign + n_malware
    X = np.zeros((n_total, TOTAL_SEQ_LEN), dtype=np.int64)
    y = np.zeros(n_total, dtype=np.int64)

    idx = 0

    # Ransomware samples (label=1)
    for i in range(n_ransomware):
        api_pool = ransomware_apis + shared_apis
        api_weights = [3.0] * len(ransomware_apis) + [1.0] * len(shared_apis)
        dll_pool = ransomware_dlls
        dll_weights = [2.0] * len(ransomware_dlls)
        # Boost Advapi32, Crypt32
        for j, d in enumerate(dll_pool):
            if d in ['Advapi32.dll', 'Crypt32.dll', 'Oleaut32.dll']:
                dll_weights[j] = 5.0
        mutex_pool = ransomware_mutexes
        mutex_weights = [2.0] * len(ransomware_mutexes)
        for j, m in enumerate(mutex_pool):
            if 'Zones' in m or 'Crypto' in m or 'Encrypt' in m:
                mutex_weights[j] = 5.0

        X[idx] = generate_sample(api_pool, api_weights, dll_pool, dll_weights,
                                 mutex_pool, mutex_weights, noise_ratio=0.1)
        y[idx] = 1
        idx += 1

    # Benign samples (label=0)
    for i in range(n_benign):
        api_pool = benign_apis + shared_apis
        api_weights = [3.0] * len(benign_apis) + [1.0] * len(shared_apis)
        dll_pool = benign_dlls
        dll_weights = [2.0] * len(benign_dlls)
        for j, d in enumerate(dll_pool):
            if d in ['Kernel32.dll', 'Version.dll']:
                dll_weights[j] = 5.0
        mutex_pool = benign_mutexes
        mutex_weights = [2.0] * len(mutex_pool)

        X[idx] = generate_sample(api_pool, api_weights, dll_pool, dll_weights,
                                 mutex_pool, mutex_weights, noise_ratio=0.2)
        y[idx] = 0
        idx += 1

    # Malware (non-ransomware) samples (label=0 for binary; separate for multi-class)
    for i in range(n_malware):
        api_pool = malware_apis + shared_apis
        api_weights = [3.0] * len(malware_apis) + [1.0] * len(shared_apis)
        dll_pool = malware_dlls
        dll_weights = [2.0] * len(malware_dlls)
        for j, d in enumerate(dll_pool):
            if d in ['Wininet.dll', 'Sechost.dll']:
                dll_weights[j] = 5.0
        mutex_pool = malware_mutexes
        mutex_weights = [2.0] * len(mutex_pool)

        X[idx] = generate_sample(api_pool, api_weights, dll_pool, dll_weights,
                                 mutex_pool, mutex_weights, noise_ratio=0.15)
        y[idx] = 0
        idx += 1

    # Shuffle
    shuffle_idx = rng.permutation(n_total)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    print(f"Generated synthetic dataset: {n_total} samples "
          f"({n_ransomware} ransomware, {n_benign} benign, {n_malware} malware)")
    print(f"Vocabulary sizes - API: {len(api_vocab)}, DLL: {len(dll_vocab)}, "
          f"Mutex: {len(mutex_vocab)}, Total: {total_vocab_size}")

    return X, y, api_vocab, dll_vocab, mutex_vocab


def get_feature_names(api_vocab, dll_vocab, mutex_vocab):
    """Get ordered feature names for the combined sequence for XAI explanations."""
    api_inv = {v: k for k, v in api_vocab.items()}
    dll_inv = {v: k for k, v in dll_vocab.items()}
    mutex_inv = {v: k for k, v in mutex_vocab.items()}

    feature_names = []
    for i in range(API_SEQ_LEN):
        feature_names.append(f'API_{i}')
    for i in range(DLL_SEQ_LEN):
        feature_names.append(f'DLL_{i}')
    for i in range(MUTEX_SEQ_LEN):
        feature_names.append(f'Mutex_{i}')

    return feature_names


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
            decoded.append('<PAD>')
        elif i < API_SEQ_LEN:
            decoded.append(api_inv.get(val, f'UNK_API_{val}'))
        elif i < API_SEQ_LEN + DLL_SEQ_LEN:
            decoded.append(dll_inv.get(val - dll_offset, f'UNK_DLL_{val}'))
        else:
            decoded.append(mutex_inv.get(val - mutex_offset, f'UNK_MUTEX_{val}'))

    return decoded
