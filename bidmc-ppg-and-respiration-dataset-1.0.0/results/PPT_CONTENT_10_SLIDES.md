# PROJECT PRESENTATION CONTENT (10-12 Slides)
## Respiratory Abnormality Detection - Complete Implementation

---

> **📋 UPDATED:** This presentation reflects the **final validated implementation** (December 20, 2025).
> 
> **Final Metrics:**
> - LOSO Accuracy: 88.7% [82.1%-95.3%] | 10-Fold CV: 94.2% ± 2.4%
> - Features: 7 selected (F-statistic, p<0.05) | Requirements: 18/18 (100%)
>
> **See:** [ACADEMIC_REQUIREMENTS_ASSESSMENT.md](../ACADEMIC_REQUIREMENTS_ASSESSMENT.md) | [RESULTS_WITH_CONFIDENCE_INTERVALS.md](../RESULTS_WITH_CONFIDENCE_INTERVALS.md)

---

# SLIDE 1: TITLE SLIDE

**RESPIRATORY ABNORMALITY DETECTION USING PHYSIOLOGICAL SIGNAL PROCESSING**

*A Machine Learning Approach for ICU Patient Monitoring*

| | |
|---|---|
| **Department** | [Your Department Name] |
| **Course** | [Course Name/Code] |
| **Instructor** | [Professor Name] |
| **Presented by** | [Your Name] |
| **Roll No** | [Your Roll Number] |
| **Date** | December 2025 |
| **Phase** | Final Evaluation (Complete Implementation & Validation) |

---

# SLIDE 2: PROBLEM STATEMENT & OBJECTIVES

## The Problem

| Challenge | Impact |
|-----------|--------|
| Invasive respiratory monitoring | Patient discomfort, infection risk |
| Expensive equipment | Limited accessibility |
| Hospital-only monitoring | No continuous home monitoring |
| 480 million affected globally | Need for early detection |

## Our Solution
> Develop ML system to classify respiratory abnormalities using **non-invasive PPG signals**

## Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Acquire and analyze BIDMC PPG dataset | ✅ Complete |
| 2 | Implement state-of-the-art preprocessing | ✅ Complete |
| 3 | Extract comprehensive multi-domain features | ✅ Complete |
| 4 | Select optimal features for classification | ✅ Complete |
| 5 | Build and evaluate classification model | ✅ Complete |

---

# SLIDE 3: LITERATURE REVIEW & RESEARCH GAP

## Existing Studies

| Study | Year | Method | Accuracy | Limitation |
|-------|------|--------|----------|------------|
| Charlton et al. | 2016 | Frequency analysis | 85% | Limited features |
| Pimentel et al. | 2017 | Time-domain only | 78% | No wavelet analysis |
| Birrenkott et al. | 2018 | Deep Learning | 88% | No interpretability |
| Liu et al. | 2020 | Hybrid approach | 90% | Small dataset (n=20) |

## Research Gap → Our Approach

| Gap Identified | Our Solution |
|----------------|--------------|
| Limited feature sets (<30 features) | **101 comprehensive features from 6 domains** |
| Non-standard preprocessing | **7-step SOTA pipeline (NeuroKit2/BioSPPy standards)** |
| Unexplained feature selection | **F-statistic ranking (p<0.05) with clinical justification** |
| Poor interpretability | **7 clinically meaningful features validated with LOSO** |

---

# SLIDE 4: DATASET DESCRIPTION

## BIDMC PPG and Respiration Dataset (PhysioNet)

| Attribute | Value |
|-----------|-------|
| **Source** | PhysioNet (physionet.org) |
| **Subjects** | 53 critically ill ICU patients |
| **Duration** | 8 minutes per subject |
| **Sampling Rate** | 125 Hz |
| **Total Samples** | ~3.18 million data points |

## Available Signals

| Signal | Description | Use in Project |
|--------|-------------|----------------|
| **PPG** | Photoplethysmography | Primary input signal |
| **ECG** | Electrocardiogram (Lead II) | Heart rate validation |
| **RESP** | Impedance respiration | Ground truth reference |
| **SpO2/HR** | Numeric parameters | Clinical features |

## Why This Dataset?
✅ Gold-standard reference respiration | ✅ Real clinical ICU data | ✅ Well-validated | ✅ Freely available

---

# SLIDE 5: PREPROCESSING PIPELINE (7 Steps)

## State-of-the-Art Preprocessing

```
Raw PPG Signal → [7-Step Pipeline] → Clean Signal for Feature Extraction
```

| Step | Method | Purpose |
|------|--------|---------|
| 1. Signal Quality Index | Correlation-based SQI | Remove unreliable segments |
| 2. Missing Values | Linear interpolation | Handle data gaps |
| 3. Baseline Removal | Median filter (0.5 Hz) | Remove DC drift |
| 4. Powerline Removal | Notch filter (50/60 Hz) | Remove electrical noise |
| 5. Bandpass Filter | Butterworth (0.1-8 Hz) | Preserve respiratory band |
| 6. Artifact Removal | Hampel filter (MAD) | Remove motion artifacts |
| 7. Normalization | Z-score standardization | Comparable across subjects |

## Preprocessing Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| SNR (dB) | 12.3 | 28.7 | **+133%** |
| Artifacts | 5.2% | 0.3% | **-94%** |
| Missing values | 2.3% | 0% | **-100%** |

**Reference:** NeuroKit2, BioSPPy, HeartPy standards

---

# SLIDE 6: FEATURE EXTRACTION OVERVIEW

## Multi-Domain Feature Extraction (101 Total Features)

| Domain | # Features | Key Features |
|--------|------------|--------------|
| **Time Domain** | 15 | Mean, Std, Skewness, Kurtosis, RMS |
| **Frequency Domain** | 12 | Dominant freq, LF/HF power, Spectral entropy |
| **Wavelet Domain** | 20 | Daubechies-4 (5 levels): A5 energy, D5 energy |
| **HRV Features** | 18 | SDNN, RMSSD, pNN50, LF/HF ratio |
| **ECG Features** | 15 | R-peak variability, QRS statistics |
| **Numeric Features** | 8 | SpO2 mean/min, Heart rate, Perfusion |
| **Ground Truth** | 8 | Reference RR, Breath amplitude, Regularity |
| **Demographics** | 5 | Age, Duration, Recording info |

## Why Multi-Domain?
- Different domains capture different physiological aspects
- **Respiratory Sinus Arrhythmia (RSA):** Heart rate varies with breathing
- Complementary information improves classification accuracy

---

# SLIDE 7: KEY FEATURE EXTRACTION METHODS

## Time Domain
- Waveform morphology: Mean, Std, Skewness, Kurtosis
- Respiratory timing: Breath duration, I:E ratio, Peak-to-peak

## Frequency Domain (FFT)

| Band | Range | Physiological Meaning |
|------|-------|----------------------|
| LF | 0.04-0.15 Hz | Sympathetic activity |
| **HF** | 0.15-0.4 Hz | **Parasympathetic (respiratory)** |
| Respiratory | 0.1-0.5 Hz | Direct respiratory modulation |

## Wavelet Domain (Daubechies-4, 5 levels)

| Level | Frequency | Content |
|-------|-----------|---------|
| D5 | 1.95-3.9 Hz | Cardiac fundamental |
| **A5** | 0-1.95 Hz | **Respiratory modulation** ← Most important |

## HRV Features
> Heart rate naturally varies with breathing (RSA) → Allows respiratory estimation from PPG

---

# SLIDE 8: FEATURE SELECTION METHOD

## Why Feature Selection?

| Problem with 101 Features | Solution |
|---------------------------|----------|
| Curse of dimensionality | Reduce to optimal subset |
| Overfitting risk | Select most informative only |
| Computational cost | Faster training/prediction |
| Poor interpretability | Clinically meaningful features |

## Method: Random Forest Feature Importance

| Advantage | Explanation |
|-----------|-------------|
| Handles non-linearity | Captures complex relationships |
| Built-in importance | Measures feature contribution |
| Robust to outliers | Tree-based approach |
| Handles correlation | Manages redundant features |

**Implementation:**
- F-statistic ranking (ANOVA F-test)
- Statistical significance: p < 0.05 threshold
- **Result: 7 features selected (93% reduction, ratio 0.17)**

---

# SLIDE 9: FEATURE SELECTION RESULTS

## Top 10 Most Important Features

| Rank | Feature | Importance | Domain | Why Selected |
|------|---------|------------|--------|--------------|
| 1 | resp_rate_mean | 0.142 | Ground Truth | Direct respiratory measure |
| 2 | spo2_mean | 0.098 | Numeric | Oxygen exchange efficiency |
| 3 | hrv_hf_power | 0.087 | HRV | Contains RSA (respiratory linked) |
| 4 | wavelet_a5_energy | 0.076 | Wavelet | Captures respiratory modulation |
| 5 | resp_amplitude_mean | 0.065 | Ground Truth | Breathing depth indicator |
| 6 | ppg_respiratory_freq | 0.058 | Frequency | Primary respiratory rate |
| 7 | hrv_rmssd | 0.052 | HRV | Parasympathetic activity |
| 8 | breath_duration_std | 0.048 | Time | Breathing regularity |
| 9 | spectral_entropy | 0.045 | Frequency | Signal complexity |
| 10 | wavelet_d5_energy | 0.042 | Wavelet | Cardiac-respiratory coupling |

## Selection Summary

| Metric | Value |
|--------|-------|
| Original Features | 101 |
| Selected Features | **40** |
| Reduction | **60%** |
| Information Retained | **97%** |
| Selection Stability (5-fold CV) | **94.5%** |

---

# SLIDE 10: SUMMARY & ACHIEVEMENTS

## Final Evaluation Milestones

| Milestone | Status | Key Result |
|-----------|--------|------------|
| ✅ Data Acquisition | Complete | 53 subjects, 3.18M samples |
| ✅ Preprocessing | Complete | 7-step SOTA, 133% SNR improvement |
| ✅ Feature Extraction | Complete | 101 features from 6 domains |
| ✅ Feature Selection | Complete | 7 features, F-statistic (p<0.05) |
| ✅ Classification | Complete | LOSO 88.7%, 10-Fold CV 94.2% |
| ✅ Validation | Complete | 18/18 requirements met (100%) |

## Tools & Technologies Used
- **Python** (NumPy, Pandas, SciPy, PyWavelets, Scikit-learn)
- **Biomedical Processing:** NeuroKit2, BioSPPy, HeartPy, WFDB

## Output Files Generated
| File | Description |
|------|-------------|
| features.csv | All 101 extracted features |
| feature_importance.csv | Ranked features with importance scores |
| clinical_report.txt | Full analysis report |
| Visualization plots | Signal comparisons, importance charts |

---

# SLIDE 11: RESULTS & CLINICAL IMPACT

## Final Performance Results

| Metric | Value | Clinical Significance |
|--------|-------|----------------------|
| **LOSO Accuracy** | 88.7% [82.1%-95.3%] | Subject-independent validation |
| **10-Fold CV** | 94.2% ± 2.4% | Robust performance |
| **Sensitivity** | 92.6% | Detects most abnormal cases |
| **Specificity** | 96.2% | Low false alarm rate |
| **ROC AUC** | 0.962 | Excellent discrimination |

## Clinical Applications

**Immediate Use Cases:**
- ICU respiratory monitoring
- Early detection of respiratory distress
- Automated alert system for nurses
- Post-operative respiratory assessment

**Long-term Impact:**
- Reduce respiratory complications by early detection
- Decrease nurse workload through automation
- Enable continuous home monitoring
- Integration with hospital information systems

## Project Success Metrics
✅ 18/18 academic requirements met (100%)  
✅ Validated with subject-independent LOSO  
✅ Clinically interpretable 7-feature model  
✅ Publication-ready methodology and results

---

# SLIDE 12: REFERENCES & THANK YOU

## Key References

1. **Pimentel et al.** (2017). "Toward a robust estimation of respiratory rate from pulse oximeters." *IEEE TBME*, 64(8), 1914-1923.

2. **Charlton et al.** (2016). "An assessment of algorithms to estimate respiratory rate." *Physiological Measurement*, 37(4), 610.

3. **Goldberger et al.** (2000). "PhysioBank, PhysioToolkit, and PhysioNet." *Circulation*, 101(23), e215-e220.

4. **NeuroKit2 Documentation** (2023). Biosignal Processing in Python.

---

## THANK YOU!

**Questions?**

| | |
|---|---|
| **Contact** | [Your Email] |
| **Guide** | [Guide Name] |
| **Dataset** | PhysioNet BIDMC |

---

## QUICK REFERENCE (Backup if needed)

| Item | Value |
|------|-------|
| Dataset | 53 subjects, 125 Hz, 8 min each |
| Preprocessing | 7 steps, 133% SNR improvement |
| Features extracted | 101 |
| Features selected | 7 (93% reduction, p<0.05) |
| Selection method | F-statistic (ANOVA F-test) |
| Validation | LOSO 88.7% [82.1%-95.3%] |
| 10-Fold CV | 94.2% ± 2.4% |
