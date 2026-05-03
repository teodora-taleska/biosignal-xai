# Data Loading

## Overview
ECG waveforms in this dataset use the WFDB format. Each record is stored as a pair of files: a binary data file and a text header file. Use the wfdb Python library to read records — you do not need to manipulate these files directly.

## WFDB file pair
For a record named `00001_lr` you will see:
- `00001_lr.dat` — binary raw signal samples.
- `00001_lr.hea` — human-readable header (sampling rate, lead names, units, patient info).

The wfdb library reads the pair together with a single call, for example:
```python
path = '../data/'
record = wfdb.rdsamp(path + 'records100/00000/00001_lr')
```

---

## CSV Metadata Files

### ptbxl_database.csv — Dataset Overview

The main metadata file contains one row per ECG record (21,799 rows total) with 28 columns.

**Key columns:**
- `filename_lr` — path to the low-res record relative to `data/` (e.g., `records100/00000/00001_lr`).
- `scp_codes` — diagnosis labels stored as a stringified dict; parse with `ast.literal_eval`.
- `age`, `sex` — patient demographic info. Sex is encoded as `0 = Male`, `1 = Female`. Age `300` is a placeholder for unknown age.
- `strat_fold` — integer 1–10 for the stratified split (folds 1–8 train, 9 val, 10 test).

**Visualization & Exploration:**

```python
import pandas as pd
import matplotlib.pyplot as plt

path = '../data/'

# Load the database
db = pd.read_csv(path + 'ptbxl_database.csv')

# Dataset shape and info
print(f"Dataset shape: {db.shape}")
print(f"Total records: {len(db)}")
print(f"\nFirst few rows:")
print(db.head())

print(f"\nColumn names:")
print(db.columns.tolist())

print(f"\nData types:")
print(db.dtypes)

# Summary statistics
print(f"\nAge distribution:")
print(db['age'].describe())
print(f"\nRecords with age=300 (missing): {(db['age'] == 300).sum()}")

# Sex distribution
print(f"\nSex distribution (0=Male, 1=Female):")
print(db['sex'].value_counts())

# Stratified fold distribution (train/val/test splits)
print(f"\nStratified fold distribution:")
print(db['strat_fold'].value_counts().sort_index())

# Visualize fold distribution
db['strat_fold'].value_counts().sort_index().plot(kind='bar', figsize=(10, 4))
plt.xlabel('Stratified Fold')
plt.ylabel('Number of Records')
plt.title('Distribution of Records Across Stratified Folds (1-10)')
plt.tight_layout()
plt.show()

# Age and sex distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Filter out age=300 (placeholder for missing age)
db[db['age'] < 300]['age'].hist(bins=30, ax=axes[0])
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Count')
axes[0].set_title('Age Distribution')

# Map 0/1 to Male/Female
db['sex'].map({0: 'Male', 1: 'Female'}).value_counts().plot(kind='bar', ax=axes[1], rot=0)
axes[1].set_xlabel('Sex')
axes[1].set_ylabel('Count')
axes[1].set_title('Sex Distribution')

plt.tight_layout()
plt.show()
```

**What to observe:**
- The dataset is large and balanced across folds for reproducible train/val/test splits.
- Age range and sex distribution give context for the population studied.
- Each record's `scp_codes` field is a dictionary; you will need to parse it using `scp_statements.csv`.

---

### scp_statements.csv — Diagnosis Code Lookup Table

This file maps individual ECG diagnosis codes to their full descriptions and diagnostic classes (5 main categories). The index is the short SCP code (e.g. `NORM`, `IMI`, `LBBB`).

**Key columns:**
- `description` — full text description of the code.
- `diagnostic_class` — one of 5 high-level diagnostic categories:
  - `NORM` — Normal ECG
  - `MI` — Myocardial Infarction
  - `STTC` — ST/T Change
  - `CD` — Conduction Disturbance
  - `HYP` — Hypertrophy
- `diagnostic_subclass` — finer-grained grouping within the class.

> **Note:** `MI` is a class name, not an individual code. Individual MI codes are things like `IMI` (inferior), `ASMI` (anteroseptal), etc.

**Visualization & Exploration:**

```python
# Load the SCP statements lookup table
scp = pd.read_csv(path + 'scp_statements.csv', index_col=0)

print(f"Total unique diagnosis codes: {len(scp)}")
print(f"\nFirst few rows:")
print(scp.head(10))

print(f"\nDiagnostic class distribution:")
print(scp['diagnostic_class'].value_counts())

# Visualize diagnostic class distribution
scp['diagnostic_class'].value_counts().plot(kind='bar', figsize=(10, 4))
plt.xlabel('Diagnostic Class')
plt.ylabel('Number of Codes')
plt.title('Distribution of Diagnosis Codes Across Diagnostic Classes')
plt.tight_layout()
plt.show()

# Example: look up a specific code
print(f"\nExample lookup for 'NORM':")
print(scp.loc['NORM'])

print(f"\nExample lookup for 'IMI' (inferior myocardial infarction, diagnostic_class='MI'):")
print(scp.loc['IMI'])

# Build a mapping from scp_code to diagnostic class for quick reference
code_to_class = scp['diagnostic_class'].to_dict()
print(f"\nCode-to-class mapping (sample):")
for code in ['NORM', 'IMI', 'AFIB', 'LBBB']:
    print(f"  {code} → {code_to_class.get(code, 'N/A')}")
```

**What to observe:**
- There are 71 unique diagnosis codes grouped into 5 diagnostic classes.
- The classes are imbalanced (e.g., more ST/T changes than myocardial infarctions).
- Each record in `ptbxl_database.csv` has one or more codes; use this lookup to expand them.

---

## Signal Data — WFDB Records

Each ECG record contains the actual voltage waveforms sampled over time. Standard ECGs have 12 leads (I, II, III, AVR, AVL, AVF, V1–V6) recorded simultaneously.

**Key properties (low-res `_lr` records):**
- **Sampling rate:** 100 Hz.
- **Duration:** 10 seconds per record → 1,000 samples.
- **Units:** mV.
- **Format:** 2D NumPy array with shape `(1000, 12)`.

### Loading and Inspecting Individual Signals

```python
import wfdb
import numpy as np

# Load a single record
record_name = '00001_lr'
record = wfdb.rdsamp(path + 'records100/00000/' + record_name)

# Unpack the signal and metadata
signal = record[0]  # 2D numpy array (num_samples, num_leads)
meta = record[1]   # metadata dict

print(f"Record: {record_name}")
print(f"Signal shape: {signal.shape}")
print(f"  Samples: {signal.shape[0]}, Leads: {signal.shape[1]}")
print(f"Sampling rate: {meta['fs']} Hz")
print(f"Duration: {signal.shape[0] / meta['fs']:.1f} seconds")
print(f"Lead names: {meta['sig_name']}")
print(f"Units: {meta['units']}")

# Get diagnosis codes for this record
record_row = db[db['filename_lr'].str.endswith(record_name)].iloc[0]
print(f"\nDiagnosis codes: {record_row['scp_codes']}")
print(f"Age: {record_row['age']}, Sex: {record_row['sex']}")
print(f"Stratified fold: {record_row['strat_fold']}")
```

### Visualization: Overview of All Leads

```python
# Plot all leads for one record
fig, axes = plt.subplots(12, 1, figsize=(14, 10))
fig.suptitle(f'ECG Record: {record_name} (All 12 Leads)', fontsize=14, fontweight='bold')

time_axis = np.arange(signal.shape[0]) / meta['fs']

for i, lead_name in enumerate(meta['sig_name']):
    axes[i].plot(time_axis, signal[:, i], linewidth=0.8, color='steelblue')
    axes[i].set_ylabel(lead_name, fontsize=10, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(time_axis[0], time_axis[-1])

axes[-1].set_xlabel('Time (seconds)', fontsize=10)
plt.tight_layout()
plt.show()
```

### Visualization: Individual Lead Detail

```python
# Zoom in on a single lead for detailed inspection
lead_idx = 0  # Lead I
lead_name = meta['sig_name'][lead_idx]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(time_axis, signal[:, lead_idx], linewidth=1, color='darkblue')
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel(f'{lead_name} ({meta["units"][lead_idx]})', fontsize=11)
ax.set_title(f'ECG Record {record_name}: Lead {lead_name}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print signal statistics
print(f"\nLead {lead_name} statistics:")
print(f"  Min: {signal[:, lead_idx].min():.2f}")
print(f"  Max: {signal[:, lead_idx].max():.2f}")
print(f"  Mean: {signal[:, lead_idx].mean():.2f}")
print(f"  Std Dev: {signal[:, lead_idx].std():.2f}")
```

### Comparison: Multiple Records with Different Diagnoses

```python
# Load a few records with different diagnoses
normal_record = db[db['scp_codes'].str.contains('NORM', na=False)].iloc[0]['filename_lr']
abnormal_record = db[db['scp_codes'].str.contains('MI', na=False)].iloc[0]['filename_lr']

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

for idx, (rec_name, title) in enumerate([
    (normal_record, 'Normal ECG (NORM)'),
    (abnormal_record, 'Abnormal ECG (MI - Myocardial Infarction)')
]):
    rec = wfdb.rdsamp(path + rec_name)
    sig = rec[0]
    fs = rec[1]['fs']
    
    time = np.arange(sig.shape[0]) / fs
    lead_II_idx = rec[1]['sig_name'].index('II')  # Plot lead II for comparison
    
    axes[idx].plot(time, sig[:, lead_II_idx], linewidth=0.8, color='steelblue')
    axes[idx].set_ylabel('Voltage (mV)', fontsize=10)
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(time[0], time[-1])

axes[-1].set_xlabel('Time (seconds)', fontsize=10)
plt.tight_layout()
plt.show()

print(f"Normal record: {normal_record}")
print(f"Abnormal record: {abnormal_record}")
```

---

## Practical Workflow

1. **Load metadata:** `db = pd.read_csv(path + 'ptbxl_database.csv')`
2. **Load SCP codes:** `scp = pd.read_csv(path + 'scp_statements.csv', index_col=0)`
3. **For each record in your dataset:**
   - Use `strat_fold` to assign to train/val/test.
   - Use `wfdb.rdsamp(path + row['filename_lr'])` to load the signal.
   - Parse `scp_codes` with `ast.literal_eval` and expand using the `scp` lookup table.
4. **Always normalize/preprocess signals before model training.**
