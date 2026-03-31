# Dataset Setup

## Using Real Cuckoo Sandbox Reports (VirusShare Dataset)

The paper uses samples from VirusShare analyzed through Cuckoo Sandbox.
To use real data:

### 1. Obtain Ransomware Samples
- Request access from [VirusShare](https://virusshare.com/) (free registration)
- Download PE (.exe) ransomware samples

### 2. Obtain Benign Samples
- Windows system files (C:\Windows\System32\*.exe)
- Download.com / Software Informer / PortableFreeware.com

### 3. Run Dynamic Analysis with Cuckoo Sandbox
- Install [Cuckoo Sandbox](https://cuckoosandbox.org/)
- Set up a Windows 7 VM as the analysis environment
- Submit each sample for analysis (timeout: 60 minutes)
- Cuckoo generates JSON reports with API calls, DLLs, and Mutexes

### 4. Organize Reports
Place the JSON reports in this directory structure:
```
data/
  ransomware/
    sample1.json
    sample2.json
    ...
  benign/
    sample1.json
    sample2.json
    ...
  malware/          (optional, for ransomware vs malware classification)
    sample1.json
    ...
```

### 5. Run with Real Data
```bash
python main.py --data_dir data/
```

## Using the ISOT Dataset (Pre-extracted Reports)

The ISOT ransomware dataset contains pre-extracted Cuckoo JSON reports:
- Download from the ISOT research lab
- Place JSON files under `data/ransomware/`

## Using Synthetic Data (Default)

If no `--data_dir` is provided, the system generates a synthetic dataset
that mimics the distributions found in real VirusShare ransomware samples.
This is suitable for development, testing, and demonstration purposes.

```bash
python main.py  # Uses synthetic data by default
```
