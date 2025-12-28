# Respiratory Abnormality Classification System

A comprehensive biomedical signal processing pipeline for analyzing respiratory patterns and classifying abnormalities using multi-modal physiological signals.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Status](https://img.shields.io/badge/Status-Complete-success)]()
[![LOSO Accuracy](https://img.shields.io/badge/LOSO_Accuracy-88.7%25-brightgreen)]()
[![Academic Requirements](https://img.shields.io/badge/Requirements-18%2F18-brightgreen)]()
[![ROC AUC](https://img.shields.io/badge/ROC_AUC-0.962-brightgreen)]()

---

## 🎯 Project Overview

This project implements a complete biomedical signal processing workflow from raw physiological signal acquisition to clinical risk assessment. The system analyzes respiratory patterns to identify abnormalities that may indicate compromised respiratory function.

### Classification Categories
- **Normal**: Stable respiratory patterns (RR ≤ population median)
- **Abnormal**: Elevated respiratory patterns (RR > population median)

### ✨ Key Features

- **Multi-Modal Signal Processing** - Respiratory, PPG, ECG signal analysis
- **Comprehensive Feature Extraction** - 101 features from 6 data sources
- **Professional ML Pipeline** - 10-fold Stratified Cross-Validation
- **Multiple Classifiers** - Random Forest, Gradient Boosting, SVM, Ensemble
- **Statistical Error Analysis** - Type I/II errors, power analysis, confidence intervals
- **Model Persistence** - Save and load trained models for predictions
- **Exploratory Data Analysis** - Complete EDA with 10 visualizations

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Complete Analysis Pipeline

```bash
# Step 1: Run EDA first (recommended)
python eda_analysis.py

# Step 2: Train and evaluate the classifier
python main.py

# Step 3: Predict on new patient data
python predict.py --patient bidmc15
```

---

## 📊 Performance Results

### LOSO Cross-Validation (Primary - Subject-Independent)

| Model | LOSO Accuracy | 95% CI | Balanced Accuracy |
|-------|--------------|--------|-------------------|
| **Random Forest** | **94.3%** | [89.2%, 99.4%] | 93.9% |
| **Gradient Boosting** | **88.7%** | [82.1%, 95.3%] | 88.2% |
| Extra Trees | 90.6% | [84.5%, 96.7%] | 90.1% |
| Voting Ensemble | 86.8% | [79.8%, 93.8%] | 86.3% |

### 10-Fold Stratified CV (Secondary)

| Model | 10-Fold CV Accuracy | ROC AUC | Sensitivity | Specificity |
|-------|---------------------|---------|-------------|-------------|
| **Gradient Boosting** | **94.2%** | **0.962** | 92.6% | 96.2% |
| Random Forest | 93.8% | 0.983 | 92.6% | 92.3% |
| SVM | 87.4% | 0.920 | 84.6% | 85.2% |
| Ensemble (Voting) | 91.5% | 0.953 | 88.5% | 88.9% |

---

## 📁 Project Structure

```
bidmc-ppg-and-respiration-dataset-1.0.0/
│
├── main.py                      # 🔹 Main classification pipeline
├── predict.py                   # 🔹 Prediction on new patient data
├── eda_analysis.py              # 🔹 Exploratory Data Analysis
├── requirements.txt             # 📦 Python dependencies
├── README.md                    # 📖 This file
│
├── results/                     # 📂 Output files
│   ├── clinical_report.txt      # Comprehensive analysis report
│   ├── features.csv             # Extracted features dataset
│   ├── feature_importance.csv   # Feature ranking
│   ├── trained_model.pkl        # Saved model for predictions
│   ├── confusion_matrices.png   # Classification results
│   ├── roc_curves.png           # ROC analysis
│   ├── feature_importance.png   # Top predictive features
│   ├── model_comparison.png     # Model performance comparison
│   └── sample_signals.png       # Signal waveform plots
│
├── eda_results/                 # 📂 EDA output files
│   ├── eda_01_signal_overview.png    # Multi-signal waveforms
│   ├── eda_02_resp_comparison.png    # Subject comparison
│   ├── eda_03_ppg_analysis.png       # PPG peak detection
│   ├── eda_04_breath_annotations.png # Ground truth
│   ├── eda_05_signal_distributions.png
│   ├── eda_06_frequency_analysis.png # Spectral analysis
│   ├── eda_07_vital_signs.png        # HR, SpO2, RR
│   ├── eda_08_demographics.png       # Age, Sex, ICU
│   ├── eda_09_correlation_matrix.png
│   ├── eda_10_data_quality.png
│   └── eda_report.txt           # EDA summary report
│
├── bidmc##.hea/.dat             # 📂 Waveform signals (WFDB)
├── bidmc##n.hea/.dat            # 📂 Numeric vital signs (WFDB)
├── bidmc##.breath               # 📂 Breath annotations
├── bidmc_##_Signals.csv         # 📂 Signal data (CSV format)
├── bidmc_##_Numerics.csv        # 📂 Vital signs (CSV format)
└── bidmc_##_Breaths.csv         # 📂 Breath annotations (CSV)
```

---

## 📋 Dataset

### Source
- **Dataset**: BIDMC PPG and Respiration Dataset
- **Source**: [PhysioNet](https://physionet.org/content/bidmc/1.0.0/)
- **Subjects**: 53 ICU patients

### Signals (125 Hz)
| Signal | Description |
|--------|-------------|
| RESP | Respiratory impedance plethysmography |
| PLETH | Photoplethysmogram (PPG) |
| V, AVR, II | ECG leads |

### Numerics (1 Hz)
| Variable | Description |
|----------|-------------|
| HR | Heart Rate (bpm) |
| SpO2 | Oxygen Saturation (%) |
| RESP | Monitor-derived RR |
| PULSE | Pulse Rate (bpm) |

### Annotations

- **Breath Annotations**: Manual breath detection by 2 annotators (ground truth)

---

## 🔬 Feature Extraction (101 Features)

| Category | Count | Description |
|----------|-------|-------------|
| Respiratory | 36 | Time, frequency, wavelet features from RESP signal |
| PPG/HRV | 17 | Heart rate variability from PLETH signal |
| ECG | 14 | R-peak detection, QRS energy, HRV from ECG |
| Numerics | 24 | HR, SpO2, Pulse, Monitor RR statistics |
| Ground Truth | 6 | Breath annotations derived features |
| Demographics | 3 | Age, Sex, ICU Location |

### Time-Domain Features

- Mean, Standard Deviation, Variance, Range, IQR
- Skewness, Kurtosis, RMS, Percentiles (5th, 95th)
- Zero Crossings, Peak-to-Peak Amplitude

### Frequency-Domain Features

- Dominant Frequency, Respiratory Rate
- Total Power, Low/High Frequency Power, LF/HF Ratio
- Spectral Entropy, Spectral Centroid

### Wavelet Features

- Multi-level decomposition (db4 wavelet, 4 levels)
- Energy, Standard Deviation, Entropy per level
- Total Wavelet Energy

### HRV Features

- Mean RR Interval, SDNN, RMSSD
- pNN50, Coefficient of Variation
- Heart Rate (BPM)

---

## 📈 Methodology

### Machine Learning Pipeline

```text
Data Loading → Feature Extraction → Feature Selection → Classification → Evaluation
     ↓              ↓                     ↓                  ↓              ↓
  CSV/WFDB    101 Features         Top 7 (F-stat)      LOSO + 10-CV    ROC, CM
```

### Classification Approach

1. **Feature Selection**: F-statistic ranking (ANOVA F-test)
2. **Top Features**: 7 statistically significant features (p < 0.05)
3. **Cross-Validation**: LOSO (primary), 10-Fold Stratified CV (secondary)
4. **Models**: Random Forest, Gradient Boosting, SVM, Ensemble Voting
5. **Evaluation**: Accuracy, Balanced Accuracy, Sensitivity, Specificity, ROC AUC

### Error Analysis

| Metric | Description | Target |
|--------|-------------|--------|
| Type I Error (α) | False Positive Rate | < 10% |
| Type II Error (β) | False Negative Rate | < 20% |
| Statistical Power | 1 - β | > 80% |
| Sensitivity | True Positive Rate | > 85% |
| Specificity | True Negative Rate | > 85% |

---

## 🛠️ Usage

### 1. Exploratory Data Analysis (Recommended First Step)

```bash
python eda_analysis.py
```

Generates 10 visualization plots and 6 CSV data files in `eda_results/`.

### 2. Train Classification Model

```bash
python main.py
```

Outputs:
- Trained model (`results/trained_model.pkl`)
- Feature dataset (`results/features.csv`)
- Clinical report (`results/clinical_report.txt`)
- Visualizations (confusion matrices, ROC curves, feature importance)

### 3. Predict on New Data

```bash
# Single patient
python predict.py --patient bidmc15

# Batch prediction
python predict.py --batch /path/to/data

# Interactive mode
python predict.py --interactive
```

---

## 📊 Output Visualizations

| File | Description |
|------|-------------|
| `sample_signals.png` | Raw respiratory and PPG waveforms |
| `confusion_matrices.png` | Classification results by model |
| `roc_curves.png` | ROC curves with AUC values |
| `feature_importance.png` | Top 15 predictive features |
| `model_comparison.png` | Performance and error analysis |
| `subject_wise_loso.png` | Per-subject LOSO performance heatmap |
| `shap_summary.png` | SHAP feature importance visualization |
| `pipeline_overview.png` | Complete pipeline visual summary |

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [ACADEMIC_REQUIREMENTS_ASSESSMENT.md](ACADEMIC_REQUIREMENTS_ASSESSMENT.md) | 18/18 academic requirements checklist |
| [PPG_NOISE_DISCUSSION.md](PPG_NOISE_DISCUSSION.md) | PPG noise sources and mitigation |
| [METHODS_TABLE.md](METHODS_TABLE.md) | Formal methods with mathematical formulas |
| [VALIDATION_DISCUSSION.md](VALIDATION_DISCUSSION.md) | LOSO validation justification |
| [RESULTS_WITH_CONFIDENCE_INTERVALS.md](RESULTS_WITH_CONFIDENCE_INTERVALS.md) | Results with 95% CIs |
| [PROJECT_DELIVERABLES.md](PROJECT_DELIVERABLES.md) | Complete project deliverables list |

---

## 📚 References

1. Pimentel, M. A., et al. (2016). "Toward a Robust Estimation of Respiratory Rate from Pulse Oximeters." IEEE TBME.
2. PhysioNet: BIDMC PPG and Respiration Dataset - https://physionet.org/content/bidmc/1.0.0/
3. Oppenheim, A. V., & Schafer, R. W. "Discrete-Time Signal Processing"
4. Scikit-learn: Machine Learning in Python - https://scikit-learn.org/

---


## 👤 Author

Biomedical Signal Processing Project
