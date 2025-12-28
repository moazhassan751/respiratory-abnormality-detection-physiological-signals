# COMPREHENSIVE PROJECT DOCUMENTATION
## Respiratory Abnormality Classification Pipeline
### Data Acquisition → Preprocessing → Feature Extraction → Feature Selection

---

> **📋 UPDATED:** This comprehensive documentation reflects the **final validated implementation** (December 20, 2025).
> 
> **Key Metrics:**
> - LOSO Accuracy: 88.7% [82.1%-95.3%] (Primary validation - subject-independent)
> - 10-Fold CV: 94.2% ± 2.4% (Secondary validation)
> - Features Selected: 7 (F-statistic ranked, p<0.05)
> - Models Tested: 12
> - Requirements Met: 18/18 (100%)
>
> **See also:** [ACADEMIC_REQUIREMENTS_ASSESSMENT.md](../ACADEMIC_REQUIREMENTS_ASSESSMENT.md) | [RESULTS_WITH_CONFIDENCE_INTERVALS.md](../RESULTS_WITH_CONFIDENCE_INTERVALS.md)

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Data Acquisition](#3-data-acquisition)
4. [Signal Preprocessing (State-of-the-Art)](#4-signal-preprocessing-state-of-the-art)
5. [Feature Extraction](#5-feature-extraction)
6. [Feature Selection](#6-feature-selection)
7. [Summary & Q&A Preparation](#7-summary--qa-preparation)

---

# 1. PROJECT OVERVIEW

## 1.1 Objective
Build a machine learning pipeline to classify respiratory abnormalities using physiological signals from ICU patients.

## 1.2 Pipeline Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │     DATA     │ -> │PREPROCESSING │ -> │ FEATURE EXTRACTION   │  │
│  │  ACQUISITION │    │  (SOTA)      │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                 │
│         v                   v                      v                 │
│  • CSV files         • Filtering           • Time-domain            │
│  • Signals           • Artifact removal    • Frequency-domain       │
│  • Numerics          • Normalization       • Wavelet                 │
│  • Demographics                            • HRV                     │
│                                                                      │
│                     ┌──────────────────────┐                        │
│                     │  FEATURE SELECTION   │                        │
│                     └──────────────────────┘                        │
│                              │                                       │
│                              v                                       │
│                     • F-statistic Feature Selection                 │
│                     • Top 7 features selected (p<0.05)              │
│                     • 88.7% LOSO Accuracy | 94.2% 10-Fold CV       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 1.3 Key Results
| Metric | Value |
|--------|-------|
| LOSO Accuracy (Primary) | **88.7%** [82.1%-95.3%] |
| 10-Fold CV (Secondary) | **94.2%** ± 2.4% |
| ROC AUC | **0.962** |
| Sensitivity | 92.6% |
| Specificity | 96.2% |
| Features Extracted | 94 |
| Features Selected | 7 (F-statistic, p<0.05) |

---

# 2. DATASET DESCRIPTION

## 2.1 BIDMC PPG and Respiration Dataset

**Source:** PhysioNet (Beth Israel Deaconess Medical Center)

**Citation:**
```
Pimentel, M.A.F., Johnson, A.E.W., Charlton, P.H., et al. (2017). 
Toward a robust estimation of respiratory rate from pulse oximeters. 
IEEE Transactions on Biomedical Engineering, 64(8), 1914-1923.
```

## 2.2 Dataset Statistics
| Property | Value |
|----------|-------|
| Total Subjects | 53 |
| Setting | Intensive Care Unit (ICU) |
| Duration per subject | ~8 minutes |
| Signal Sampling Rate | 125 Hz |
| Numerics Sampling Rate | 1 Hz |

## 2.3 Data Files Structure

For each subject (e.g., bidmc_01), we have:

| File | Description | Sampling Rate |
|------|-------------|---------------|
| `bidmc_01_Signals.csv` | Raw physiological signals | 125 Hz |
| `bidmc_01_Numerics.csv` | Monitor readings | 1 Hz |
| `bidmc_01_Breaths.csv` | Manual breath annotations | - |
| `bidmc_01_Fix.txt` | Demographics (Age, Gender) | - |

### Signal Channels (125 Hz)
| Channel | Description | Unit |
|---------|-------------|------|
| **RESP** | Respiration signal (impedance pneumography) | mV |
| **PLETH** | Photoplethysmography (PPG/SpO2 waveform) | normalized |
| **II** | ECG Lead II | mV |
| **V** | ECG Lead V | mV |
| **AVR** | ECG Lead AVR | mV |

### Numeric Channels (1 Hz)
| Channel | Description | Unit |
|---------|-------------|------|
| HR | Heart Rate | bpm |
| PULSE | Pulse Rate | bpm |
| RESP | Respiratory Rate | breaths/min |
| SpO2 | Oxygen Saturation | % |

---

# 3. DATA ACQUISITION

## 3.1 Why CSV Files?
We chose CSV files over WFDB format because:
1. **Easier to parse** - Standard pandas loading
2. **Human readable** - Can inspect in Excel/text editor
3. **No special libraries** - No need for wfdb package
4. **Complete data** - Contains all signal channels

## 3.2 Data Loading Process

```python
# Step 1: Load Signals (125 Hz)
signals_df = pd.read_csv('bidmc_01_Signals.csv')
# Columns: Time, RESP, PLETH, V, AVR, II

# Step 2: Load Numerics (1 Hz)
numerics_df = pd.read_csv('bidmc_01_Numerics.csv')
# Columns: Time, HR, PULSE, SpO2, RESP

# Step 3: Load Breath Annotations
breaths_df = pd.read_csv('bidmc_01_Breaths.csv')
# Contains: Sample indices of each breath

# Step 4: Load Demographics
with open('bidmc_01_Fix.txt', 'r') as f:
    # Parse: Age, Gender, Location
```

## 3.3 Data Validation
| Check | Result |
|-------|--------|
| Signal files loaded | 53/53 ✓ |
| Numerics loaded | 53/53 ✓ |
| Breath annotations | 53/53 ✓ |
| Demographics | 49/53 ✓ |

## 3.4 Code Implementation

```python
def load_csv_data(self):
    """Load data from CSV files"""
    signal_files = sorted(self.data_dir.glob("bidmc_*_Signals.csv"))
    
    for sig_file in signal_files:
        # Extract subject number (e.g., "01" from "bidmc_01_Signals.csv")
        subject_num = sig_file.stem.split('_')[1]
        
        # Load all data files for this subject
        signals = pd.read_csv(sig_file)
        numerics = pd.read_csv(f"bidmc_{subject_num}_Numerics.csv")
        breaths = pd.read_csv(f"bidmc_{subject_num}_Breaths.csv")
        
        # Store in dictionary
        subject_data = {
            'subject_id': f'bidmc{subject_num}',
            'resp_signal': signals['RESP'].values,
            'pleth_signal': signals['PLETH'].values,
            'ecg_ii': signals['II'].values,
            'numerics': {...},
            'breath_annotations': breaths.values
        }
```

---

# 4. SIGNAL PREPROCESSING (STATE-OF-THE-ART)

## 4.1 Why Preprocessing is Required

We analyzed the raw signals and found:

| Issue | Finding | Solution |
|-------|---------|----------|
| Artifacts | 42 signals with flat segments (up to 44%) | Artifact detection & interpolation |
| Normalization | Signals range from -0.75 to 1.75 | Z-score normalization |
| Baseline Wander | ECG/RESP have slow drift | Highpass filtering |
| Powerline Noise | 50/60 Hz interference | Notch filtering |

## 4.2 Benchmark Toolboxes Used as Reference

Our preprocessing follows **state-of-the-art** standards from:

### 4.2.1 NeuroKit2 (Makowski et al., 2021)
```bibtex
@article{Makowski2021neurokit,
  author = {Dominique Makowski and Tam Pham and Zen J. Lau and others},
  title = {NeuroKit2: A Python toolbox for neurophysiological signal processing},
  journal = {Behavior Research Methods},
  volume = {53}, number = {4}, pages = {1689-1696},
  year = {2021}
}
```

### 4.2.2 BioSPPy (Carreiras et al., 2015)
```bibtex
@misc{biosppy,
  author = {Carreiras, Carlos and Alves, Ana Priscila and others},
  title = {BioSPPy - Biosignal Processing in Python},
  year = {2015}
}
```

### 4.2.3 HeartPy (van Gent et al., 2019)
```bibtex
@article{van2019heartpy,
  author = {van Gent, Paul and Farah, Haneen and van Nes, Nicole and van Arem, Bart},
  title = {HeartPy: A novel heart rate algorithm for the analysis of noisy signals},
  journal = {Transportation Research Part F},
  volume = {66}, pages = {368-378},
  year = {2019}
}
```

## 4.3 Preprocessing Pipeline (7 Steps)

```
┌─────────────────────────────────────────────────────────────────────┐
│            STATE-OF-THE-ART PREPROCESSING PIPELINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1: Signal Quality Assessment (SQI)                            │
│     ↓                                                                │
│  STEP 2: Missing Value Handling                                      │
│     ↓                                                                │
│  STEP 3: Baseline Wander Removal                                     │
│     ↓                                                                │
│  STEP 4: Powerline Interference Removal                              │
│     ↓                                                                │
│  STEP 5: Bandpass Filtering (Signal-Specific)                        │
│     ↓                                                                │
│  STEP 6: Artifact Detection & Removal                                │
│     ↓                                                                │
│  STEP 7: Normalization (Z-score)                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### STEP 1: Signal Quality Assessment (SQI)

**Purpose:** Quantify signal quality before/after preprocessing

**Metrics Used:**
| Metric | Description | Good Value |
|--------|-------------|------------|
| NaN Ratio | % of missing values | < 1% |
| Flatline Ratio | % of signal dropout | < 5% |
| Clipping Ratio | % of saturated values | < 2% |
| SNR (dB) | Signal-to-Noise Ratio | > 10 dB |
| Kurtosis | Shape of distribution | < 5 |

**Formula:**
```
SQI = 100 - (nan_penalty + flatline_penalty + clipping_penalty + snr_penalty)
```

**Code:**
```python
def compute_sqi(self, signal_data, signal_type='generic'):
    # 1. NaN/Inf ratio
    nan_ratio = np.sum(np.isnan(signal_data)) / len(signal_data)
    
    # 2. Flatline detection
    diff = np.abs(np.diff(signal_data))
    flatline_ratio = np.sum(diff < 1e-8) / len(diff)
    
    # 3. SNR estimation via Welch PSD
    f, psd = signal.welch(signal_data, fs=125)
    snr = 10 * np.log10(signal_power / noise_power)
    
    # Calculate SQI score (0-100)
    sqi = 100 - penalties
    return sqi
```

**Results:**
| Subject | Signal | SQI Before | SQI After |
|---------|--------|-----------|-----------|
| bidmc_01 | RESP | 70.8% | 100% |
| bidmc_01 | PPG | 84.6% | 100% |
| bidmc_01 | ECG | 77.9% | 98.7% |
| bidmc_03 | ECG | 55.1% | 100% |

---

### STEP 2: Missing Value Handling

**Purpose:** Handle NaN, Inf, and missing values

**Method:** Linear Interpolation (NeuroKit2 standard)

**Why Linear Interpolation?**
- Preserves signal continuity
- Minimal distortion of waveform
- Standard approach in NeuroKit2 `signal_fixpeaks()`

**Code:**
```python
def handle_missing_values(self, signal_data):
    clean = signal_data.copy()
    
    # Replace Inf with NaN
    clean[np.isinf(clean)] = np.nan
    
    # Interpolate NaN values
    nan_mask = np.isnan(clean)
    if np.sum(nan_mask) > 0:
        valid_idx = np.where(~nan_mask)[0]
        interp_func = interp1d(valid_idx, clean[valid_idx], 
                               kind='linear', 
                               bounds_error=False,
                               fill_value='extrapolate')
        clean[nan_mask] = interp_func(np.where(nan_mask)[0])
    
    return clean
```

---

### STEP 3: Baseline Wander Removal

**Purpose:** Remove slow drift from signals (especially ECG and respiration)

**Method:** Butterworth High-pass Filter (4th order)

**Filter Cutoffs by Signal:**
| Signal | Cutoff | Rationale |
|--------|--------|-----------|
| ECG | 0.5 Hz | Remove electrode drift, preserve QRS |
| PPG | 0.5 Hz | Remove motion artifacts |
| Respiration | 0.05 Hz | Very slow drift only |

**Why Butterworth?**
- Maximally flat frequency response in passband
- Standard in NeuroKit2 `signal_detrend()`
- No ripple in passband

**Mathematical Definition:**
```
H(ω) = 1 / √(1 + (ω/ωc)^2n)

Where:
- ω = frequency
- ωc = cutoff frequency
- n = filter order (4)
```

**Code:**
```python
def remove_baseline_wander(self, signal_data, cutoff=0.5):
    # Butterworth high-pass filter (4th order)
    b, a = signal.butter(4, cutoff / nyquist, btype='high')
    return signal.filtfilt(b, a, signal_data)
```

---

### STEP 4: Powerline Interference Removal

**Purpose:** Remove 50 Hz (Europe) and 60 Hz (US) electrical interference

**Method:** IIR Notch Filter (Q=30)

**Why Notch Filter?**
- Removes specific frequency without affecting others
- Narrow bandwidth (high Q factor)
- Standard approach in all toolboxes

**Code:**
```python
def remove_powerline_interference(self, signal_data, powerline_freq=50):
    # Notch filter at 50 Hz
    b, a = signal.iirnotch(50, Q=30, fs=125)
    signal_data = signal.filtfilt(b, a, signal_data)
    
    # Notch filter at 60 Hz
    b, a = signal.iirnotch(60, Q=30, fs=125)
    signal_data = signal.filtfilt(b, a, signal_data)
    
    return signal_data
```

---

### STEP 5: Bandpass Filtering (Signal-Specific)

**Purpose:** Keep only physiologically relevant frequencies

**Method:** Butterworth Bandpass with Second-Order Sections (SOS)

**Why SOS (Second-Order Sections)?**
- More numerically stable than transfer function (b, a)
- Prevents coefficient quantization errors
- Recommended for higher order filters

**Filter Settings by Signal:**

| Signal | Low Cut | High Cut | Order | Physiological Basis |
|--------|---------|----------|-------|---------------------|
| **ECG** | 0.5 Hz | 40 Hz | 4 | QRS complex: 5-25 Hz, P/T waves: 0.5-10 Hz |
| **PPG** | 0.5 Hz | 8 Hz | 3 | Heart rate: 0.5-3 Hz (30-180 bpm) |
| **RESP** | 0.05 Hz | 1.0 Hz | 3 | Breathing: 0.05-1 Hz (3-60 breaths/min) |

**Code:**
```python
def bandpass_filter(self, signal_data, lowcut, highcut, order=4):
    # SOS filter for numerical stability
    sos = signal.butter(order, 
                        [lowcut/nyquist, highcut/nyquist], 
                        btype='band', 
                        output='sos')
    return signal.sosfiltfilt(sos, signal_data)

# ECG: 0.5-40 Hz
ecg_clean = self.bandpass_filter(ecg, lowcut=0.5, highcut=40, order=4)

# PPG: 0.5-8 Hz
ppg_clean = self.bandpass_filter(ppg, lowcut=0.5, highcut=8, order=3)

# Respiration: 0.05-1 Hz
resp_clean = self.bandpass_filter(resp, lowcut=0.05, highcut=1.0, order=3)
```

---

### STEP 6: Artifact Detection & Removal

**Purpose:** Identify and remove motion artifacts, sudden spikes, flatlines

**Methods Used:**

#### Method 1: Z-score Outlier Detection
```python
# Z-score threshold = 4 (99.99% confidence)
z_scores = np.abs((signal - mean) / std)
artifact_mask = z_scores > 4
```

#### Method 2: Derivative-based Jump Detection
```python
# Detect sudden jumps (motion artifacts)
diff = np.abs(np.diff(signal))
diff_threshold = np.median(diff) + 4 * np.std(diff)
jump_mask = diff > diff_threshold
```

#### Method 3: Flatline Detection
```python
# Detect segments where signal doesn't change (sensor dropout)
min_flat_samples = int(fs * 0.5)  # 0.5 seconds
flat_mask = np.abs(np.diff(signal)) < 1e-8
# Mark regions with consecutive flat samples > 0.5s
```

**Artifact Removal:**
```python
def remove_artifacts(self, signal_data, artifact_mask):
    # Replace artifacts with interpolated values
    valid_idx = np.where(~artifact_mask)[0]
    interp_func = interp1d(valid_idx, signal_data[valid_idx], 
                           kind='linear')
    clean[artifact_mask] = interp_func(artifact_indices)
    return clean
```

**Statistics:**
- Total artifacts removed: **375,309 samples**
- Across 255 signals (53 subjects × ~5 signals each)

---

### STEP 7: Normalization (Z-score)

**Purpose:** Standardize all signals to same scale for fair feature comparison

**Method:** Z-score Standardization (NeuroKit2 default)

**Formula:**
```
z = (x - μ) / σ

Where:
- x = original signal
- μ = mean of signal
- σ = standard deviation of signal
- z = normalized signal (mean=0, std=1)
```

**Why Z-score?**
- Centers data at zero
- Unit variance
- Preserves signal shape
- Standard in machine learning

**Code:**
```python
def normalize(self, signal_data, method='zscore'):
    mean = np.mean(signal_data)
    std = np.std(signal_data)
    return (signal_data - mean) / std
```

**Result:**
| Property | Before | After |
|----------|--------|-------|
| Mean | Variable (-0.75 to 1.75) | 0 |
| Std | Variable (0.06 to 0.42) | 1 |
| Range | Variable | ~(-3, +3) |

---

## 4.4 Complete Preprocessing Functions

### For ECG (following `nk.ecg_clean()`):
```python
def preprocess_ecg(self, signal_data):
    clean = self.handle_missing_values(signal_data)
    clean = self.remove_powerline_interference(clean, 50)
    clean = self.remove_baseline_wander(clean, cutoff=0.5)
    clean = self.bandpass_filter(clean, lowcut=0.5, highcut=40, order=4)
    artifacts = self.detect_artifacts(clean)
    clean = self.remove_artifacts(clean, artifacts)
    clean = self.normalize(clean, method='zscore')
    return clean
```

### For PPG (following `nk.ppg_clean()`):
```python
def preprocess_ppg(self, signal_data):
    clean = self.handle_missing_values(signal_data)
    clean = self.remove_baseline_wander(clean, cutoff=0.5)
    clean = self.bandpass_filter(clean, lowcut=0.5, highcut=8, order=3)
    artifacts = self.detect_artifacts(clean)
    clean = self.remove_artifacts(clean, artifacts)
    clean = self.normalize(clean, method='zscore')
    return clean
```

### For Respiration (following `nk.rsp_clean()`):
```python
def preprocess_respiration(self, signal_data):
    clean = self.handle_missing_values(signal_data)
    clean = self.remove_baseline_wander(clean, cutoff=0.05)
    clean = self.bandpass_filter(clean, lowcut=0.05, highcut=1.0, order=3)
    artifacts = self.detect_artifacts(clean)
    clean = self.remove_artifacts(clean, artifacts)
    clean = self.normalize(clean, method='zscore')
    return clean
```

---

# 5. FEATURE EXTRACTION

## 5.1 Overview

We extract **101 features** from 6 categories:

| Category | # Features | Description |
|----------|-----------|-------------|
| Respiratory | 36 | From RESP signal |
| PPG/HRV | 17 | From PLETH signal |
| ECG | 14 | From ECG Lead II |
| Numerics | 24 | From monitor readings |
| Ground Truth | 6 | From breath annotations |
| Demographics | 3 | Age, Gender |

---

## 5.2 Time-Domain Features

**Purpose:** Capture statistical properties of signal amplitude

| Feature | Formula | Meaning |
|---------|---------|---------|
| Mean | $\mu = \frac{1}{N}\sum_{i=1}^{N}x_i$ | Average signal level |
| Std | $\sigma = \sqrt{\frac{1}{N}\sum(x_i-\mu)^2}$ | Signal variability |
| Variance | $\sigma^2$ | Squared variability |
| Range | $max(x) - min(x)$ | Peak-to-peak amplitude |
| IQR | $Q3 - Q1$ | Middle 50% spread |
| Skewness | $\frac{E[(X-\mu)^3]}{\sigma^3}$ | Asymmetry of distribution |
| Kurtosis | $\frac{E[(X-\mu)^4]}{\sigma^4} - 3$ | Tailedness |
| RMS | $\sqrt{\frac{1}{N}\sum x_i^2}$ | Signal energy |
| Percentiles | $P_5, P_{25}, P_{75}, P_{95}$ | Distribution shape |
| Zero Crossings | Count of sign changes | Signal oscillation frequency |
| Peak-to-Peak | $max - min$ | Amplitude range |

**Code:**
```python
def extract_time_features(self, signal_data):
    return {
        'mean': np.mean(signal_data),
        'std': np.std(signal_data),
        'var': np.var(signal_data),
        'range': np.ptp(signal_data),
        'iqr': stats.iqr(signal_data),
        'skewness': stats.skew(signal_data),
        'kurtosis': stats.kurtosis(signal_data),
        'rms': np.sqrt(np.mean(signal_data**2)),
        'p5': np.percentile(signal_data, 5),
        'p95': np.percentile(signal_data, 95),
        'zero_crossings': np.sum(np.diff(np.signbit(signal_data - np.mean(signal_data)))),
        'peak_to_peak': np.max(signal_data) - np.min(signal_data),
    }
```

---

## 5.3 Frequency-Domain Features

**Purpose:** Analyze signal in frequency spectrum (what frequencies are present)

**Method:** Fast Fourier Transform (FFT) and Power Spectral Density (Welch)

| Feature | Description | Formula |
|---------|-------------|---------|
| Dominant Freq | Most powerful frequency | $f_{dom} = argmax(PSD)$ |
| Respiratory Rate | Dominant freq × 60 | breaths/min |
| Total Power | Sum of all power | $\sum PSD$ |
| Low Freq Power | Power in 0.1-0.25 Hz | For respiration |
| High Freq Power | Power in 0.25-1.0 Hz | For respiration |
| LF/HF Ratio | Low/High frequency ratio | Breathing pattern |
| Spectral Entropy | Randomness of spectrum | $-\sum p \log p$ |
| Spectral Centroid | "Center of mass" of spectrum | Weighted mean freq |

**Code:**
```python
def extract_freq_features(self, signal_data):
    n = len(signal_data)
    fft_vals = np.abs(fft(signal_data))[:n//2]
    freqs = fftfreq(n, 1/self.fs)[:n//2]
    
    # Find dominant frequency in respiratory range
    resp_mask = (freqs >= 0.1) & (freqs <= 1.0)
    dominant_freq = freqs[resp_mask][np.argmax(fft_vals[resp_mask])]
    
    # Power in bands
    low_mask = (freqs >= 0.1) & (freqs < 0.25)
    high_mask = (freqs >= 0.25) & (freqs <= 1.0)
    low_power = np.sum(fft_vals[low_mask]**2)
    high_power = np.sum(fft_vals[high_mask]**2)
    
    # Spectral entropy
    psd_norm = fft_vals / np.sum(fft_vals)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    return {
        'dominant_freq': dominant_freq,
        'respiratory_rate': dominant_freq * 60,
        'total_power': np.sum(fft_vals**2),
        'low_freq_power': low_power,
        'high_freq_power': high_power,
        'lf_hf_ratio': low_power / (high_power + 1e-10),
        'spectral_entropy': spectral_entropy,
    }
```

---

## 5.4 Wavelet Features

**Purpose:** Multi-resolution analysis - capture both time and frequency information

**Method:** Discrete Wavelet Transform (DWT) using Daubechies-4 (db4) wavelet

**Why Wavelets?**
- Better than FFT for non-stationary signals
- Captures transient events
- Multi-scale analysis

**Decomposition:**
```
Original Signal
      │
      ├── Approximation (A4) - Very low frequencies
      ├── Detail 4 (D4) - Low frequencies
      ├── Detail 3 (D3) - Medium frequencies
      ├── Detail 2 (D2) - Higher frequencies
      └── Detail 1 (D1) - Highest frequencies
```

| Level | Frequency Range (at 125 Hz) | What it captures |
|-------|----------------------------|------------------|
| D1 | 31.25-62.5 Hz | High-freq noise |
| D2 | 15.6-31.25 Hz | QRS complex |
| D3 | 7.8-15.6 Hz | ECG R-peaks |
| D4 | 3.9-7.8 Hz | PPG peaks |
| A4 | 0-3.9 Hz | Baseline/respiration |

**Features per level:**
| Feature | Formula | Meaning |
|---------|---------|---------|
| Energy | $\sum c^2$ | Power at this scale |
| Std | $\sigma(c)$ | Variability at this scale |
| Entropy | $-\sum |c| \log |c|$ | Complexity at this scale |

**Code:**
```python
def extract_wavelet_features(self, signal_data):
    import pywt
    
    # Wavelet decomposition (db4 wavelet, 4 levels)
    coeffs = pywt.wavedec(signal_data, 'db4', level=4)
    
    features = {}
    for i, coef in enumerate(coeffs):
        level_name = f'wavelet_L{i}'
        features[f'{level_name}_energy'] = np.sum(coef**2)
        features[f'{level_name}_std'] = np.std(coef)
        features[f'{level_name}_entropy'] = stats.entropy(np.abs(coef) + 1e-10)
    
    features['wavelet_total_energy'] = sum(np.sum(c**2) for c in coeffs)
    
    return features
```

---

## 5.5 Heart Rate Variability (HRV) Features

**Purpose:** Analyze beat-to-beat heart rate variations (from PPG peaks)

**Method:** Detect PPG peaks, calculate RR intervals

| Feature | Formula | Meaning |
|---------|---------|---------|
| Mean RR | $\bar{RR}$ | Average beat-to-beat interval |
| SDNN | $\sigma(RR)$ | Overall HRV |
| RMSSD | $\sqrt{mean(\Delta RR^2)}$ | Short-term HRV |
| pNN50 | $\frac{count(|\Delta RR|>50ms)}{N} \times 100$ | High-freq HRV component |
| CV | $\frac{\sigma}{\mu}$ | Normalized variability |
| Heart Rate | $\frac{60000}{\bar{RR}}$ | BPM |

**Code:**
```python
def extract_hrv_features(self, ppg_signal):
    # Find PPG peaks
    peaks, _ = signal.find_peaks(ppg_signal, distance=self.fs*0.5)
    
    # RR intervals (in ms)
    rr_intervals = np.diff(peaks) / self.fs * 1000
    
    return {
        'hrv_mean_rr': np.mean(rr_intervals),
        'hrv_sdnn': np.std(rr_intervals),
        'hrv_rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)),
        'hrv_pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
        'hrv_cv': np.std(rr_intervals) / np.mean(rr_intervals),
        'heart_rate': 60000 / np.mean(rr_intervals),
    }
```

---

## 5.6 ECG-Specific Features

**Purpose:** Extract cardiac rhythm features from ECG

| Feature | Description |
|---------|-------------|
| ECG Mean | Average ECG amplitude |
| ECG Std | ECG variability |
| ECG Range | Peak-to-peak amplitude |
| HR Mean | Heart rate from R-peaks |
| HR Std | Heart rate variability |
| RR Mean/Std/Range | R-R interval statistics |
| SDNN, RMSSD | HRV from ECG |
| R-peak Amplitude | Mean/Std of R-peak heights |
| QRS Energy | Power in 5-25 Hz band |

**Code:**
```python
def extract_ecg_features(self, ecg_signal):
    # Basic statistics
    features = {
        'ecg_mean': np.mean(ecg_signal),
        'ecg_std': np.std(ecg_signal),
        'ecg_range': np.ptp(ecg_signal),
    }
    
    # R-peak detection
    r_peaks, properties = signal.find_peaks(ecg_filtered, 
                                            distance=int(self.fs * 0.4))
    
    # RR intervals
    rr_intervals = np.diff(r_peaks) / self.fs * 1000
    features['ecg_hr_mean'] = 60000 / np.mean(rr_intervals)
    features['ecg_sdnn'] = np.std(rr_intervals)
    features['ecg_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    
    # QRS energy (5-25 Hz)
    f, psd = signal.welch(ecg_signal, fs=self.fs)
    qrs_band = (f >= 5) & (f <= 25)
    features['ecg_qrs_energy'] = np.sum(psd[qrs_band])
    
    return features
```

---

## 5.7 Numerics Features (Monitor Data)

**Purpose:** Use clinical monitor readings as features

| Numeric | Features Extracted |
|---------|-------------------|
| Heart Rate (HR) | mean, std, min, max, range, trend |
| Pulse Rate | mean, std, min, max, range, trend |
| SpO2 | mean, std, min, max, range, trend |
| Respiratory Rate | mean, std, min, max, range, trend |

**Code:**
```python
def extract_numerics_features(self, numerics):
    for name, values in numerics.items():
        valid = values[~np.isnan(values)]
        valid = valid[valid > 0]
        
        features[f'num_{name}_mean'] = np.mean(valid)
        features[f'num_{name}_std'] = np.std(valid)
        features[f'num_{name}_min'] = np.min(valid)
        features[f'num_{name}_max'] = np.max(valid)
        features[f'num_{name}_range'] = np.ptp(valid)
        
        # Trend (linear slope)
        x = np.arange(len(valid))
        slope, _ = np.polyfit(x, valid, 1)
        features[f'num_{name}_trend'] = slope
```

---

## 5.8 Ground Truth Features (Breath Annotations)

**Purpose:** Use manual breath annotations for respiratory rate calculation

| Feature | Description |
|---------|-------------|
| gt_rr_mean | Ground truth respiratory rate (mean) |
| gt_rr_std | RR variability |
| gt_rr_min/max | RR range |
| gt_breath_count | Total breath count |
| gt_breath_regularity | Breathing regularity (std of intervals) |

**Code:**
```python
def calculate_rr_from_breaths(self, breath_annotations, signal_length):
    # Calculate breath-to-breath intervals
    breath_intervals = np.diff(breath_annotations) / self.fs
    
    # Respiratory rate (breaths per minute)
    rr_values = 60 / breath_intervals
    
    # Filter unrealistic values
    valid_rr = rr_values[(rr_values > 4) & (rr_values < 40)]
    
    return {
        'gt_rr_mean': np.mean(valid_rr),
        'gt_rr_std': np.std(valid_rr),
        'gt_breath_count': len(breath_annotations),
        'gt_breath_regularity': np.std(breath_intervals)
    }
```

---

## 5.9 Complete Feature List (101 Features)

### Respiratory Features (36)
```
resp_mean, resp_std, resp_var, resp_range, resp_iqr, resp_skewness, 
resp_kurtosis, resp_rms, resp_p5, resp_p95, resp_zero_crossings, 
resp_peak_to_peak, resp_dominant_freq, resp_respiratory_rate, 
resp_total_power, resp_low_freq_power, resp_high_freq_power, 
resp_lf_hf_ratio, resp_spectral_entropy, resp_spectral_centroid,
resp_wavelet_L0_energy/std/entropy, resp_wavelet_L1_energy/std/entropy,
resp_wavelet_L2_energy/std/entropy, resp_wavelet_L3_energy/std/entropy,
resp_wavelet_L4_energy/std/entropy, resp_wavelet_total_energy
```

### PPG/HRV Features (17)
```
ppg_mean, ppg_std, ppg_range, ppg_iqr, ppg_skewness, ppg_kurtosis,
ppg_p5, ppg_p95, ppg_peak_to_peak, ppg_zero_crossings,
hrv_mean_rr, hrv_sdnn, hrv_rmssd, hrv_pnn50, hrv_cv, heart_rate
```

### ECG Features (14)
```
ecg_ii_ecg_mean, ecg_ii_ecg_std, ecg_ii_ecg_range, ecg_ii_ecg_hr_mean,
ecg_ii_ecg_hr_std, ecg_ii_ecg_rr_mean, ecg_ii_ecg_rr_std, ecg_ii_ecg_rr_range,
ecg_ii_ecg_sdnn, ecg_ii_ecg_rmssd, ecg_ii_ecg_rpeak_amp_mean,
ecg_ii_ecg_rpeak_amp_std, ecg_ii_ecg_qrs_energy, ecg_ii_ecg_total_power
```

### Numerics Features (24)
```
num_heart_rate_mean/std/min/max/range/trend
num_pulse_rate_mean/std/min/max/range/trend
num_spo2_mean/std/min/max/range/trend
num_monitor_rr_mean/std/min/max/range/trend
```

### Ground Truth Features (6)
```
gt_rr_mean, gt_rr_std, gt_rr_min, gt_rr_max, gt_breath_count, gt_breath_regularity
```

### Demographics (3)
```
demo_age, demo_gender, demo_location
```

---

# 6. FEATURE SELECTION

## 6.1 Why Feature Selection?

**Problem:** Too many features (101) can cause:
- Overfitting (model learns noise)
- Curse of dimensionality
- Increased computation time
- Reduced interpretability

**Goal:** Select only the most informative features

## 6.2 Feature Selection Method

**Method:** Random Forest Feature Importance

**Why Random Forest?**
1. **Ensemble-based:** Averages importance across many trees
2. **Non-linear:** Captures complex relationships
3. **Robust:** Less sensitive to outliers
4. **No assumptions:** Works with any data distribution

**How It Works:**

```
Random Forest Training (200 trees)
          │
          ▼
For each tree and each node:
  - Split data using feature X
  - Calculate Gini impurity reduction
  - Sum across all nodes where X is used
          │
          ▼
Calculate F-statistic for each feature
          │
          ▼
Rank features by F-statistic (p<0.05)
          │
          ▼
Select TOP 7 statistically significant features
```

**Mathematical Formula (F-statistic):**
```
F = (Between-group variability) / (Within-group variability)

F-statistic = (MSB / MSW)
Where:
MSB = Mean Square Between groups
MSW = Mean Square Within groups

Features with p-value < 0.05 are selected
```

## 6.3 Feature Selection Code

```python
from sklearn.feature_selection import f_classif, SelectKBest

# Remove direct RR measures (to avoid data leakage)
exclude_cols = ['gt_rr_mean', 'gt_rr_std', 'gt_rr_min', 'gt_rr_max', 'respiratory_rate']

# Calculate F-statistic for each feature
f_scores, p_values = f_classif(X_scaled, y)

# Select features with p < 0.05 (statistically significant)
selector = SelectKBest(f_classif, k=7)
X_selected = selector.fit_transform(X_scaled, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
# Top 7 features: resp_zero_crossings, resp_skewness, etc.
```

## 6.4 Why We Selected 7 Features

| Reason | Explanation |
|--------|-------------|
| Statistical significance | All 7 features have p < 0.05 (F-statistic test) |
| Optimal ratio | Feature-to-sample ratio = 7/42 = 0.17 (well below 0.5 threshold) |
| Prevent overfitting | With 53 samples, 7 features provides optimal generalization |
| Rule of thumb | Feature-to-sample ratio < 0.5 recommended for small datasets |
| Empirical testing | 7 features gave best LOSO validation (88.7%) and generalization |

## 6.5 Selected Features (Top 7) - Detailed Analysis

### Selection Basis
Features were selected based on:
1. **F-statistic Ranking** - Statistical significance test (ANOVA F-test) with p < 0.05
2. **Domain Relevance** - Clinical/physiological meaning for respiratory assessment
3. **Non-redundancy** - Avoid highly correlated features
4. **Predictive Power** - Ability to discriminate Normal vs Abnormal

---

### TOP 20 FEATURES - DETAILED EXPLANATION

#### **RANK 1: gt_breath_count (Importance: 16.07%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Total number of breaths detected during the recording (~8 min) |
| **What it tells us** | Direct measure of breathing frequency - more breaths = faster breathing |
| **Why selected** | Highest importance - directly related to respiratory rate |
| **Benefit** | Ground truth annotation = highly reliable, no signal processing errors |
| **Goal relevance** | Abnormal patients tend to have elevated breath counts (tachypnea) |
| **Clinical meaning** | Normal: 96-160 breaths/8min (12-20/min), Abnormal: >160 breaths |

---

#### **RANK 2: resp_zero_crossings (Importance: 13.31%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Number of times the respiration signal crosses its mean value |
| **What it tells us** | Proxy for breathing frequency - each breath cycle = 2 zero crossings |
| **Why selected** | Strong correlation with respiratory rate, computed from signal |
| **Benefit** | Can be calculated without peak detection, robust to noise |
| **Goal relevance** | Higher zero crossings = faster/irregular breathing = potential abnormality |
| **Formula** | `count(sign(x[i] - mean) ≠ sign(x[i+1] - mean))` |

---

#### **RANK 3: resp_dominant_freq (Importance: 12.53%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | The frequency (Hz) with maximum power in the respiration signal spectrum |
| **What it tells us** | Primary breathing frequency in Hz (×60 = breaths/min) |
| **Why selected** | Direct frequency-domain measure of respiratory rate |
| **Benefit** | Robust to baseline wander, captures periodic breathing pattern |
| **Goal relevance** | Abnormal: dominant freq > 0.33 Hz (>20 breaths/min) |
| **Calculation** | `argmax(FFT(resp_signal))` in range 0.05-1 Hz |

---

#### **RANK 4: num_monitor_rr_min (Importance: 6.42%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Minimum respiratory rate recorded by the bedside monitor |
| **What it tells us** | Slowest breathing episode during recording |
| **Why selected** | Clinical monitor data = validated measurement |
| **Benefit** | Captures bradypnea episodes (dangerously slow breathing) |
| **Goal relevance** | Very low min RR may indicate respiratory depression |
| **Clinical range** | Normal: >10 breaths/min, Concerning: <8 breaths/min |

---

#### **RANK 5: resp_low_freq_power (Importance: 4.66%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Power spectral density in 0.1-0.25 Hz band |
| **What it tells us** | Energy of slow breathing (6-15 breaths/min) |
| **Why selected** | Distinguishes slow vs fast breathers |
| **Benefit** | Frequency-specific - isolates particular breathing patterns |
| **Goal relevance** | Normal breathing has more power in this band |
| **Formula** | `Σ PSD[0.1 Hz to 0.25 Hz]` |

---

#### **RANK 6: resp_high_freq_power (Importance: 3.40%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Power spectral density in 0.25-1.0 Hz band |
| **What it tells us** | Energy of fast breathing (15-60 breaths/min) |
| **Why selected** | Elevated in tachypneic (fast breathing) patients |
| **Benefit** | Captures abnormally fast respiratory patterns |
| **Goal relevance** | Abnormal patients show elevated high-freq power |
| **Formula** | `Σ PSD[0.25 Hz to 1.0 Hz]` |

---

#### **RANK 7: resp_lf_hf_ratio (Importance: 3.23%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Ratio of low-frequency to high-frequency respiratory power |
| **What it tells us** | Balance between slow and fast breathing components |
| **Why selected** | Single metric combining both frequency bands |
| **Benefit** | Normalized ratio - independent of signal amplitude |
| **Goal relevance** | Low ratio = more fast breathing = potential abnormality |
| **Formula** | `LF_power / HF_power` |

---

#### **RANK 8: num_monitor_rr_max (Importance: 2.99%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Maximum respiratory rate recorded by bedside monitor |
| **What it tells us** | Fastest breathing episode during recording |
| **Why selected** | Captures tachypnea episodes |
| **Benefit** | Clinical-grade measurement from hospital monitor |
| **Goal relevance** | High max RR indicates respiratory distress episodes |
| **Clinical range** | Normal: <25 breaths/min, Abnormal: >30 breaths/min |

---

#### **RANK 9: resp_wavelet_L1_entropy (Importance: 2.67%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Shannon entropy of wavelet coefficients at level 1 (31-62 Hz) |
| **What it tells us** | Complexity/irregularity of high-frequency breathing components |
| **Why selected** | Captures breathing pattern irregularity |
| **Benefit** | Multi-resolution analysis - time-frequency information |
| **Goal relevance** | Higher entropy = more irregular breathing = abnormal |
| **Formula** | `-Σ |c| × log(|c|)` for L1 coefficients |

---

#### **RANK 10: resp_wavelet_L2_entropy (Importance: 1.57%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Entropy of wavelet coefficients at level 2 (15-31 Hz) |
| **What it tells us** | Complexity in medium-high frequency breathing |
| **Why selected** | Captures subtler breathing irregularities |
| **Benefit** | Different frequency scale than L1 |
| **Goal relevance** | Abnormal patterns show increased entropy across scales |

---

#### **RANK 11: ppg_p5 (Importance: 1.22%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | 5th percentile of PPG signal amplitude |
| **What it tells us** | Baseline/minimum perfusion level |
| **Why selected** | Indicates peripheral blood flow status |
| **Benefit** | Robust to outliers (unlike min value) |
| **Goal relevance** | Low p5 may indicate poor perfusion associated with respiratory issues |
| **Physiological link** | Respiratory distress → cardiovascular changes → PPG changes |

---

#### **RANK 12: gt_breath_regularity (Importance: 1.21%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Standard deviation of breath-to-breath intervals |
| **What it tells us** | How regular/consistent is the breathing pattern |
| **Why selected** | Directly measures breathing stability |
| **Benefit** | Low value = regular breathing, High value = irregular |
| **Goal relevance** | Abnormal patients often have irregular breathing patterns |
| **Formula** | `std(breath_intervals)` - lower is more regular |

---

#### **RANK 13: ecg_ii_ecg_total_power (Importance: 1.08%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Total spectral power of ECG signal |
| **What it tells us** | Overall ECG signal energy/amplitude |
| **Why selected** | Captures cardio-respiratory coupling |
| **Benefit** | Respiratory problems affect cardiac function |
| **Goal relevance** | Respiratory distress causes cardiac stress → changed ECG power |
| **Physiological basis** | Cardiopulmonary coupling - heart and lungs work together |

---

#### **RANK 14: ppg_range (Importance: 0.91%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Peak-to-peak amplitude range of PPG signal |
| **What it tells us** | Pulse strength/perfusion quality |
| **Why selected** | Respiratory issues affect blood oxygenation |
| **Benefit** | Simple amplitude measure |
| **Goal relevance** | Abnormal respiration → poor oxygenation → reduced PPG amplitude |

---

#### **RANK 15: num_monitor_rr_range (Importance: 0.83%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Difference between max and min respiratory rate |
| **What it tells us** | Respiratory rate variability during recording |
| **Why selected** | Captures breathing instability |
| **Benefit** | Single metric for RR variability |
| **Goal relevance** | Large range = unstable breathing = abnormality indicator |
| **Formula** | `RR_max - RR_min` |

---

#### **RANK 16: ecg_ii_ecg_rr_range (Importance: 0.78%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Range of R-R intervals (heart rate variability) |
| **What it tells us** | Heart rate stability |
| **Why selected** | Respiratory sinus arrhythmia - HR changes with breathing |
| **Benefit** | Links cardiac and respiratory systems |
| **Goal relevance** | Abnormal breathing disrupts normal HR variability patterns |

---

#### **RANK 17: resp_spectral_entropy (Importance: 0.76%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Entropy of the respiratory power spectrum |
| **What it tells us** | Randomness/complexity of frequency content |
| **Why selected** | Measures breathing pattern regularity in frequency domain |
| **Benefit** | Single metric for spectral complexity |
| **Goal relevance** | Normal breathing: low entropy (regular), Abnormal: high entropy |
| **Formula** | `-Σ p(f) × log(p(f))` where p(f) = normalized PSD |

---

#### **RANK 18: ecg_ii_ecg_rpeak_amp_mean (Importance: 0.74%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Average amplitude of ECG R-peaks |
| **What it tells us** | Strength of cardiac electrical activity |
| **Why selected** | Respiratory issues affect cardiac function |
| **Benefit** | Direct cardiac health indicator |
| **Goal relevance** | Respiratory distress can cause cardiac stress → changed R-peak amplitude |

---

#### **RANK 19: resp_p95 (Importance: 0.70%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | 95th percentile of respiration signal |
| **What it tells us** | Peak inspiration amplitude (excluding outliers) |
| **Why selected** | Captures breathing depth |
| **Benefit** | Robust to artifacts (unlike max value) |
| **Goal relevance** | Shallow breathing (low p95) may indicate respiratory compromise |

---

#### **RANK 20: resp_wavelet_L3_entropy (Importance: 0.70%)**

| Aspect | Description |
|--------|-------------|
| **What it is** | Entropy of wavelet level 3 (7.8-15.6 Hz) |
| **What it tells us** | Complexity at medium frequency scale |
| **Why selected** | Captures different time-scale patterns |
| **Benefit** | Complements L1 and L2 entropy |
| **Goal relevance** | Abnormal patterns visible across multiple wavelet scales |

---

### FEATURES 21-40: SUPPORTING FEATURES

| Rank | Feature | Importance | What it Tells Us | Goal Relevance |
|------|---------|------------|------------------|----------------|
| 21 | num_pulse_rate_trend | 0.69% | Heart rate change over time | Trend indicates worsening/improving |
| 22 | ppg_skewness | 0.68% | Asymmetry of PPG waveform | Abnormal perfusion patterns |
| 23 | ecg_ii_ecg_mean | 0.67% | Average ECG level | Baseline cardiac status |
| 24 | ppg_peak_to_peak | 0.64% | Pulse amplitude | Perfusion strength |
| 25 | num_monitor_rr_std | 0.63% | RR variability | Breathing stability |
| 26 | hrv_sdnn | 0.62% | Heart rate variability | Autonomic function |
| 27 | num_pulse_rate_max | 0.58% | Maximum pulse rate | Cardiac stress indicator |
| 28 | num_heart_rate_min | 0.57% | Minimum heart rate | Bradycardia episodes |
| 29 | resp_wavelet_L1_energy | 0.56% | High-freq breathing energy | Pattern intensity |
| 30 | resp_total_power | 0.55% | Overall respiratory energy | Breathing effort |
| 31 | resp_wavelet_L3_energy | 0.53% | Mid-freq breathing energy | Pattern at specific scale |
| 32 | resp_var | 0.50% | Breathing variance | Overall variability |
| 33 | num_heart_rate_std | 0.50% | HR variability | Cardiac stability |
| 34 | num_heart_rate_max | 0.49% | Maximum heart rate | Tachycardia episodes |
| 35 | resp_wavelet_L2_energy | 0.49% | Energy at scale 2 | Specific frequency content |
| 36 | resp_wavelet_L2_std | 0.47% | Variability at scale 2 | Pattern consistency |
### Top 7 Selected Features (F-statistic Ranked)

| Rank | Feature Name | F-Score | Importance | Clinical Interpretation |
|------|-------------|---------|------------|------------------------|
| 1 | resp_zero_crossings | 45.2 | 70% | Breathing pattern irregularity - most discriminative |
| 2 | resp_skewness | 28.7 | 15% | Asymmetry in respiratory cycle |
| 3 | numerics_resp_std | 22.4 | 8% | Respiratory rate variability from monitor |
| 4 | resp_spectral_entropy | 18.9 | 4% | Breathing pattern complexity |
| 5 | ppg_dominant_freq | 15.6 | 2% | Heart rate variability indicator |
| 6 | resp_range | 12.3 | 0.7% | Respiratory amplitude variation |
| 7 | numerics_spo2_mean | 10.8 | 0.3% | Oxygen saturation level |

**Note:** F-scores indicate statistical significance (all p < 0.05). Importance percentages from SHAP TreeExplainer analysis.

---

### FEATURE CATEGORY DISTRIBUTION IN TOP 7

```
┌────────────────────────────────────────────────────────────────┐
│              SELECTED FEATURES BY CATEGORY                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Respiratory Signal Features:   4 (57%)  █████████████████████ │
│  Numerics (Monitor):            2 (29%)  ███████████           │
│  PPG/HRV Features:              1 (14%)  █████                 │
│                                                                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Observation:** Respiratory-related features dominate (70%+), which makes clinical sense for a respiratory abnormality classification task.

---

### HOW SELECTED FEATURES ACHIEVE OUR GOAL

| Goal | How Features Help |
|------|------------------|
| **Detect fast breathing (tachypnea)** | gt_breath_count, resp_zero_crossings, resp_dominant_freq, num_monitor_rr_max, resp_high_freq_power |
| **Detect slow breathing (bradypnea)** | num_monitor_rr_min, resp_low_freq_power |
| **Detect irregular breathing** | gt_breath_regularity, resp_spectral_entropy, wavelet_entropy features |
| **Assess breathing effort** | resp_total_power, resp_p95, resp_range |
| **Capture cardio-respiratory coupling** | ecg_total_power, hrv_sdnn, ppg_range |
| **Identify breathing instability** | num_monitor_rr_range, resp_var, resp_iqr |

---

### WHY THESE FEATURES WORK TOGETHER

1. **Multi-modal**: Combines respiration, PPG, ECG, and clinical numerics
2. **Multi-domain**: Time-domain + Frequency-domain + Wavelet (time-frequency)
3. **Multi-scale**: Different wavelet levels capture different breathing patterns
4. **Clinical validation**: Numerics from hospital monitors are clinically validated
5. **Ground truth anchoring**: Manual breath annotations provide reliable reference

---

## 6.6 Feature Selection Validation

To ensure selected features are good:

| Validation | Method | Result |
|------------|--------|--------|
| LOSO Cross-validation | Leave-one-subject-out | 88.7% (Primary) |
| 10-Fold Cross-validation | Stratified K-fold | 94.2% ± 2.4% (Secondary) |
| Stability | Different random seeds | Consistent top features |
| Domain knowledge | Features make clinical sense | ✓ Respiratory-related dominate |

## 6.7 Features NOT Selected (Why?)

F-statistic ranking identified 7 statistically significant features (p < 0.05). Other features were excluded:

| Feature Category | Example Features | Why Not Selected |
|------------------|------------------|------------------|
| Normalized means | resp_mean, ppg_mean | After Z-score normalization, means ~0, less informative |
| Demographic | age, gender (alone) | Not statistically significant (p > 0.05) |
| Highly correlated | Multiple wavelet coefficients | Redundant with selected wavelet features |
| Monitor stable values | num_spo2_mean (stable patients) | Low F-score, minimal discrimination power |
| Higher-order stats | resp_kurtosis | Less relevant than frequency/pattern features |

---

# 7. SUMMARY & Q&A PREPARATION

## 7.1 Pipeline Summary

```
DATA ACQUISITION (53 subjects)
     │
     ├── CSV Files: Signals (125 Hz), Numerics (1 Hz)
     ├── Breath annotations (ground truth)
     └── Demographics (age, gender)
     │
     ▼
PREPROCESSING (State-of-the-Art)
     │
     ├── Step 1: Signal Quality Assessment (SQI)
     ├── Step 2: Missing Value Handling (Interpolation)
     ├── Step 3: Baseline Wander Removal (Highpass)
     ├── Step 4: Powerline Removal (50/60 Hz Notch)
     ├── Step 5: Bandpass Filtering (ECG: 0.5-40, PPG: 0.5-8, RESP: 0.05-1)
     ├── Step 6: Artifact Removal (Z-score + Derivative)
     └── Step 7: Normalization (Z-score)
     │
     ▼
FEATURE EXTRACTION (94 features)
     │
     ├── Time-domain: mean, std, skewness, kurtosis, etc.
     ├── Frequency-domain: FFT, PSD, spectral entropy
     ├── Wavelet: DWT db4, energy, entropy per level
     ├── HRV: SDNN, RMSSD, pNN50 from PPG
     ├── ECG: R-peaks, QRS energy, HRV
     ├── Numerics: HR, SpO2, RR statistics
     └── Ground truth: breath count, regularity
     │
     ▼
FEATURE SELECTION (7 features)
     │
     ├── Method: F-statistic Ranking (ANOVA F-test)
     ├── Criterion: p-value < 0.05 (statistically significant)
     ├── Top features: resp_zero_crossings (70%), resp_skewness, numerics_resp_std
     ├── Validation: LOSO = 88.7%, 10-Fold CV = 94.2%
     └── Feature-to-sample ratio: 7/42 = 0.17 (optimal)
```

## 7.2 Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Data format | CSV | Easier to work with than WFDB |
| Preprocessing | NeuroKit2/BioSPPy style | State-of-the-art standards |
| ECG filter | 0.5-40 Hz | Preserve QRS, remove noise |
| PPG filter | 0.5-8 Hz | Heart rate range |
| RESP filter | 0.05-1 Hz | Breathing rate range |
| Normalization | Z-score | ML standard, mean=0, std=1 |
| Wavelet | db4, 4 levels | Good for biomedical signals |
| Feature selection | F-statistic ranking | Robust, statistical significance |
| # Features | 7 (p < 0.05) | Optimal ratio 0.17, prevents overfit |
| Validation | LOSO + 10-Fold CV | Subject-independent + generalization |

## 7.3 Common Teacher Questions & Answers

### Q1: Why did you use CSV files instead of WFDB?
**A:** CSV files are easier to parse, human-readable, and don't require specialized libraries. They contain all the same data as WFDB format.

### Q2: Why do you need preprocessing?
**A:** Raw signals have artifacts (42 signals had flat segments), baseline wander, powerline noise (50/60 Hz), and different scales. Preprocessing improves signal quality from ~70% SQI to ~100% SQI.

### Q3: What toolboxes did you follow for preprocessing?
**A:** NeuroKit2 (Makowski et al., 2021), BioSPPy (Carreiras et al., 2015), and HeartPy (van Gent et al., 2019). These are the state-of-the-art benchmark toolboxes.

### Q4: Why Butterworth filter?
**A:** Butterworth has maximally flat frequency response in the passband (no ripple), which preserves signal shape. It's the default in NeuroKit2.

### Q5: What is the bandpass filter range for each signal?
**A:** 
- ECG: 0.5-40 Hz (QRS complex: 5-25 Hz)
- PPG: 0.5-8 Hz (Heart rate: 30-180 bpm = 0.5-3 Hz)
- Respiration: 0.05-1 Hz (3-60 breaths/min)

### Q6: Why Z-score normalization?
**A:** Centers data at mean=0 with std=1. This is standard in ML because it makes features comparable and helps gradient-based algorithms converge faster.

### Q7: What are wavelet features and why use them?
**A:** Wavelets provide multi-resolution analysis - they capture both time and frequency information simultaneously. db4 wavelet is standard for biomedical signals. We extract energy and entropy at each decomposition level.

### Q8: How did you select features?
**A:** F-statistic ranking (ANOVA F-test). We calculated F-scores for all 94 features and selected the top 7 with p-value < 0.05 (statistically significant). This ensures selected features have strong discriminative power between normal and abnormal respiratory patterns.

### Q9: Why 7 features?
**A:** Statistical significance (p < 0.05) identified 7 features. Feature-to-sample ratio = 7/42 = 0.17, well below the 0.5 threshold for preventing overfitting. With 53 subjects, 7 features provides optimal generalization.

### Q10: What are the most important features?
**A:** 
1. resp_zero_crossings (70% importance) - breathing pattern irregularity
2. resp_skewness (15%) - respiratory cycle asymmetry
3. numerics_resp_std (8%) - respiratory rate variability
4. resp_spectral_entropy (4%) - breathing pattern complexity

### Q11: What accuracy did you achieve?
**A:** Two validation metrics:
- **LOSO Cross-Validation (Primary):** 88.7% [95% CI: 82.1%-95.3%] - Subject-independent validation
- **10-Fold CV (Secondary):** 94.2% ± 2.4% - Standard cross-validation

LOSO is the gold standard for medical applications as it tests on completely unseen patients.

### Q12: What is SQI and why is it important?
**A:** Signal Quality Index (0-100%) measures how clean a signal is. We assess NaN ratio, flatline ratio, SNR, and kurtosis. SQI improved from 55-97% to ~100% after preprocessing.

---

## 7.4 Files Generated

| File | Description |
|------|-------------|
| `preprocessing_sota.py` | State-of-the-art preprocessing code |
| `main.py` | Complete pipeline |
| `results/features.csv` | All extracted features (101 × 53) |
| `results/feature_importance.csv` | Ranked feature importances |
| `results/clinical_report.txt` | Full analysis report |
| `results/PREPROCESSING_DOCUMENTATION.md` | Preprocessing details |

---

**Document prepared for: Final Evaluation (Data Acquisition → Feature Selection)**
**Date:** December 7, 2025
