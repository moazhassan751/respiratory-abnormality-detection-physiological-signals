# State-of-the-Art Preprocessing Pipeline Documentation

> **📋 NOTE:** This document covers preprocessing methodology (current).
> For final project results and metrics, see: [RESULTS_WITH_CONFIDENCE_INTERVALS.md](../RESULTS_WITH_CONFIDENCE_INTERVALS.md)

## Overview

This document describes the **State-of-the-Art Signal Preprocessing Pipeline** implemented for the BIDMC PPG and Respiration Analysis project. The preprocessing follows established benchmark toolbox standards from:

1. **NeuroKit2** (Makowski et al., 2021)
2. **BioSPPy** (Carreiras et al., 2015)
3. **HeartPy** (van Gent et al., 2019)

---

## References

### NeuroKit2
```bibtex
@article{Makowski2021neurokit,
  author = {Dominique Makowski and Tam Pham and Zen J. Lau and Jan C. Brammer and 
            François Lespinasse and Hung Pham and Christopher Schölzel and S. H. Annabel Chen},
  title = {NeuroKit2: A Python toolbox for neurophysiological signal processing},
  journal = {Behavior Research Methods},
  volume = {53},
  number = {4},
  pages = {1689--1696},
  year = {2021},
  doi = {10.3758/s13428-020-01516-y}
}
```

### BioSPPy
```bibtex
@misc{biosppy,
  author = {Carreiras, Carlos and Alves, Ana Priscila and Lourenço, André and 
            Canento, Filipe and Silva, Hugo and Fred, Ana and others},
  title = {BioSPPy - Biosignal Processing in Python},
  year = {2015},
  url = {https://github.com/PIA-Group/BioSPPy/}
}
```

### HeartPy
```bibtex
@article{van2019heartpy,
  author = {van Gent, Paul and Farah, Haneen and van Nes, Nicole and van Arem, Bart},
  title = {HeartPy: A novel heart rate algorithm for the analysis of noisy signals},
  journal = {Transportation Research Part F: Traffic Psychology and Behaviour},
  volume = {66},
  pages = {368--378},
  year = {2019},
  doi = {10.1016/j.trf.2019.09.015}
}
```

---

## Preprocessing Pipeline Steps

### Step 1: Signal Quality Assessment (SQI)

Before preprocessing, we assess signal quality using multiple metrics:

| Metric | Description | Reference |
|--------|-------------|-----------|
| NaN/Inf Ratio | Percentage of missing/invalid values | Orphanidou et al., 2015 |
| Flatline Ratio | Percentage of signal dropout | Elgendi et al., 2016 |
| Clipping Ratio | Percentage of saturated values | - |
| SNR (dB) | Signal-to-Noise Ratio | NeuroKit2 |
| Kurtosis | Shape of distribution (artifact detection) | - |
| Template Score | Pattern regularity | NeuroKit2 |

**SQI Formula:**
```
SQI = 100 - (nan_penalty + flatline_penalty + clipping_penalty + snr_penalty + kurtosis_penalty)
```

### Step 2: Missing Value Handling

**Method:** Linear Interpolation (NeuroKit2 standard)

```python
# NeuroKit2 approach
from scipy.interpolate import interp1d

valid_idx = np.where(~np.isnan(signal))[0]
interp_func = interp1d(valid_idx, signal[valid_idx], kind='linear', 
                       bounds_error=False, fill_value='extrapolate')
clean_signal = interp_func(np.arange(len(signal)))
```

**Reference:** `nk.signal_fixpeaks()` approach

### Step 3: Baseline Wander Removal

**Method:** Butterworth High-pass Filter (4th order)

| Signal | Cutoff Frequency | Rationale |
|--------|-----------------|-----------|
| ECG | 0.5 Hz | Remove electrode drift |
| PPG | 0.5 Hz | Remove motion artifacts |
| Respiration | 0.05 Hz | Remove very slow drift |

**Reference:** 
- NeuroKit2: `nk.signal_detrend()`
- BioSPPy: `biosppy.tools.filter_signal()` with highpass

```python
# Butterworth highpass (NeuroKit2/BioSPPy standard)
from scipy.signal import butter, filtfilt

b, a = butter(4, cutoff / nyquist, btype='high')
clean_signal = filtfilt(b, a, signal)
```

### Step 4: Powerline Interference Removal

**Method:** IIR Notch Filter at 50 Hz and 60 Hz

```python
# Remove 50 Hz (Europe) and 60 Hz (US) powerline
from scipy.signal import iirnotch, filtfilt

b, a = iirnotch(50, Q=30, fs=125)  # 50 Hz
signal = filtfilt(b, a, signal)

b, a = iirnotch(60, Q=30, fs=125)  # 60 Hz  
signal = filtfilt(b, a, signal)
```

**Reference:** NeuroKit2 `nk.signal_filter()` with `powerline=50`

### Step 5: Bandpass Filtering (Signal-Specific)

**Method:** Butterworth Bandpass Filter with Second-Order Sections (SOS)

| Signal Type | Low Cutoff | High Cutoff | Order | Rationale |
|-------------|-----------|-------------|-------|-----------|
| **ECG** | 0.5 Hz | 40 Hz | 4 | QRS complex: 5-25 Hz |
| **PPG** | 0.5 Hz | 8 Hz | 3 | Heart rate: 0.5-3 Hz |
| **Respiration** | 0.05 Hz | 1.0 Hz | 3 | 3-60 breaths/min |

**Reference:**
- NeuroKit2: `nk.ecg_clean()`, `nk.ppg_clean()`, `nk.rsp_clean()`
- BioSPPy: Signal-specific processing functions

```python
# SOS filter for numerical stability (NeuroKit2 standard)
from scipy.signal import butter, sosfiltfilt

sos = butter(order, [lowcut/nyquist, highcut/nyquist], 
             btype='band', output='sos')
clean_signal = sosfiltfilt(sos, signal)
```

### Step 6: Artifact Detection & Removal

**Methods:**
1. **Z-score outlier detection** (threshold = 4)
2. **Derivative-based jump detection**
3. **Flatline detection** (>0.5s segments)

**Reference:** NeuroKit2 `nk.signal_fixpeaks()` approach

```python
# Z-score based detection
z_scores = np.abs((signal - mean) / std)
artifact_mask = z_scores > 4

# Derivative-based (sudden jumps)
diff = np.abs(np.diff(signal))
diff_threshold = np.median(diff) + 4 * np.std(diff)
jump_mask = diff > diff_threshold

# Interpolate artifacts
clean_signal = interpolate(signal, artifact_mask)
```

### Step 7: Normalization

**Method:** Z-score Standardization (NeuroKit2 default)

```python
# Z-score normalization (NeuroKit2 standard)
normalized = (signal - mean) / std
# Result: mean=0, std=1
```

**Reference:** NeuroKit2 `nk.standardize()`

---

## Comparison with Benchmark Toolboxes

### NeuroKit2 Equivalent Functions

| Our Function | NeuroKit2 Equivalent |
|--------------|---------------------|
| `preprocess_ecg()` | `nk.ecg_clean()` |
| `preprocess_ppg()` | `nk.ppg_clean()` |
| `preprocess_respiration()` | `nk.rsp_clean()` |
| `bandpass_filter()` | `nk.signal_filter()` |
| `remove_baseline_wander()` | `nk.signal_detrend()` |
| `normalize()` | `nk.standardize()` |

### BioSPPy Equivalent Functions

| Our Function | BioSPPy Equivalent |
|--------------|-------------------|
| `preprocess_ecg()` | `biosppy.signals.ecg.ecg()` |
| `bandpass_filter()` | `biosppy.tools.filter_signal()` |
| `detect_artifacts()` | Signal-specific modules |

---

## Implementation Details

### File Structure
```
preprocessing_sota.py
├── SignalQualityAssessment     # SQI computation
├── StateOfTheArtPreprocessor   # Core preprocessing
│   ├── handle_missing_values()
│   ├── remove_baseline_wander()
│   ├── remove_powerline_interference()
│   ├── bandpass_filter()
│   ├── detect_artifacts()
│   ├── remove_artifacts()
│   ├── normalize()
│   ├── preprocess_ecg()
│   ├── preprocess_ppg()
│   └── preprocess_respiration()
└── PreprocessingPipeline       # High-level interface
    └── preprocess_all()
```

### Usage
```python
from preprocessing_sota import PreprocessingPipeline

# Initialize pipeline
pipeline = PreprocessingPipeline(sampling_rate=125, verbose=True)

# Preprocess all subjects
preprocessed_data = pipeline.preprocess_all(raw_data)
```

---

## Results

### Signal Quality Improvement

| Subject | Signal | SQI Before | SQI After | Improvement |
|---------|--------|-----------|-----------|-------------|
| bidmc_01 | RESP | 70.8% | 100% | +29.2% |
| bidmc_01 | PPG | 84.6% | 100% | +15.4% |
| bidmc_01 | ECG | 77.9% | 98.7% | +20.8% |
| bidmc_02 | RESP | 68.4% | 100% | +31.6% |
| bidmc_02 | PPG | 97.8% | 100% | +2.2% |
| bidmc_02 | ECG | 82.1% | 100% | +17.9% |
| bidmc_03 | RESP | 87.8% | 100% | +12.2% |
| bidmc_03 | PPG | 96.0% | 100% | +4.0% |
| bidmc_03 | ECG | 55.1% | 100% | +44.9% |

### Classification Performance

| Metric | Without Preprocessing | With Preprocessing |
|--------|----------------------|-------------------|
| LOSO Accuracy | 82.3% | **88.7% [82.1%-95.3%]** |
| 10-Fold CV | 90.1% | **94.2% ± 2.4%** |
| ROC AUC | 0.92 | **0.962** |
| Sensitivity | 91.1% | **92.6%** |
| Specificity | 83.0% | **96.2%** |

---

## Summary

This preprocessing pipeline implements state-of-the-art biomedical signal processing techniques following established benchmark toolboxes:

1. ✅ **Signal Quality Assessment** (SQI) - Following Orphanidou et al.
2. ✅ **Missing Value Handling** - Linear interpolation (NeuroKit2)
3. ✅ **Baseline Wander Removal** - Butterworth highpass (NeuroKit2/BioSPPy)
4. ✅ **Powerline Removal** - 50/60 Hz notch filter (NeuroKit2)
5. ✅ **Bandpass Filtering** - Signal-specific (ECG: 0.5-40Hz, PPG: 0.5-8Hz, RESP: 0.05-1Hz)
6. ✅ **Artifact Detection/Removal** - Z-score + derivative (NeuroKit2)
7. ✅ **Normalization** - Z-score standardization (NeuroKit2)

The implementation achieves **88.7% LOSO accuracy** (with 95% confidence interval [82.1%-95.3%]) and **94.2% 10-Fold CV accuracy** with proper preprocessing, demonstrating the importance of signal preprocessing in biomedical signal analysis. Using 7 statistically significant features (F-statistic ranked, p<0.05) provides optimal generalization while preventing overfitting.
