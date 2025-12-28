# Methods Table: Preprocessing, Feature Engineering, and Classification
## Respiratory Abnormality Detection Using PPG - Technical Methods Reference

---

## TABLE 1: SIGNAL PREPROCESSING METHODS

| Stage | Method | Purpose | Parameters | Mathematical Formula | Assumptions | Reference |
|-------|--------|---------|------------|---------------------|-------------|-----------|
| **1.1** | Signal Quality Index (SQI) | Assess signal usability | Threshold: 50% | $SQI = 100 - \sum penalties$ | Quality metrics are additive | Orphanidou (2015) |
| **1.2** | NaN/Inf Detection | Identify missing data | - | $r_{nan} = \frac{n_{nan}}{n_{total}}$ | Missing data is random | - |
| **1.3** | Flatline Detection | Detect signal dropout | Threshold: $10^{-8}$ | $r_{flat} = \frac{\sum(|x_i - x_{i-1}| < \epsilon)}{n-1}$ | Constant signal = artifact | - |
| **1.4** | Clipping Detection | Detect ADC saturation | Percentiles: 1%, 99% | $r_{clip} = \frac{n_{P1} + n_{P99}}{n_{total}}$ | Extreme values = clipping | Elgendi (2016) |
| **2.1** | Linear Interpolation | Fill short gaps | Gap < 10 samples | $x(t) = x_1 + \frac{x_2 - x_1}{t_2 - t_1}(t - t_1)$ | Short gaps are linear | - |
| **2.2** | Cubic Spline | Fill medium gaps | 10 ≤ Gap < 50 | $S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3$ | Smooth signal continuity | - |
| **3.1** | Highpass Filter (Butterworth) | Remove baseline wander | $f_c = 0.5$ Hz, order = 5 | $H(s) = \frac{s^n}{s^n + \omega_c^n}$ | Linear phase response | NeuroKit2 |
| **3.2** | Notch Filter (IIR) | Remove powerline interference | $f_0 = 60$ Hz, $Q = 30$ | $H(s) = \frac{s^2 + \omega_0^2}{s^2 + \frac{\omega_0}{Q}s + \omega_0^2}$ | Constant powerline frequency | BioSPPy |
| **4.1** | Bandpass Filter (RESP) | Extract respiratory band | 0.1 - 1.0 Hz, order = 4 | $H(s) = \frac{(\frac{s}{\omega_l})^n}{(1+(\frac{s}{\omega_l})^{2n})(1+(\frac{s}{\omega_h})^{2n})}$ | RR in normal physiological range | - |
| **4.2** | Bandpass Filter (PPG) | Extract cardiac band | 0.5 - 4.0 Hz, order = 4 | Same as above | HR 30-240 BPM | - |
| **5.1** | MAD Outlier Detection | Detect motion artifacts | Threshold: 3×MAD | $MAD = 1.4826 \cdot median(|x_i - \tilde{x}|)$ | Normal distribution approximation | Leys (2013) |
| **5.2** | Median Filter | Remove impulse noise | Kernel = 3 samples | $y_i = median(x_{i-1}, x_i, x_{i+1})$ | Impulse noise is sparse | - |
| **6.1** | Z-score Normalization | Standardize amplitude | - | $z = \frac{x - \mu}{\sigma}$ | Gaussian distribution | - |
| **6.2** | Min-Max Scaling | Scale to [0,1] range | - | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Known bounds | - |

---

## TABLE 2: FEATURE EXTRACTION METHODS

| Domain | Feature | Purpose | Formula | Unit | Clinical Relevance |
|--------|---------|---------|---------|------|-------------------|
| **Time** | Mean | Central tendency | $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$ | Signal units | Baseline amplitude |
| **Time** | Standard Deviation | Variability | $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$ | Signal units | Signal stability |
| **Time** | Variance | Spread measure | $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$ | Signal units² | Power proxy |
| **Time** | Range | Peak-to-peak | $R = x_{max} - x_{min}$ | Signal units | Amplitude variation |
| **Time** | IQR | Robust spread | $IQR = Q_3 - Q_1$ | Signal units | Outlier-resistant variability |
| **Time** | Skewness | Asymmetry | $\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}$ | Dimensionless | Waveform shape |
| **Time** | Kurtosis | Tail weight | $\gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4} - 3$ | Dimensionless | Artifact detection |
| **Time** | RMS | Energy measure | $RMS = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$ | Signal units | Signal power |
| **Time** | Zero Crossings | Frequency proxy | $ZC = \sum_{i=1}^{n-1} \mathbb{1}[(x_i - \bar{x})(x_{i+1} - \bar{x}) < 0]$ | Count | **Breathing irregularity** |
| **Frequency** | Dominant Frequency | Primary oscillation | $f_{dom} = \arg\max_f |X(f)|^2$ | Hz | Respiratory rate |
| **Frequency** | Total Power | Signal energy | $P_{total} = \sum_f |X(f)|^2$ | Power units | Overall activity |
| **Frequency** | Low-Freq Power | Slow oscillations | $P_{LF} = \sum_{f=0.1}^{0.25} |X(f)|^2$ | Power units | Bradypnea indicator |
| **Frequency** | High-Freq Power | Fast oscillations | $P_{HF} = \sum_{f=0.25}^{1.0} |X(f)|^2$ | Power units | Tachypnea indicator |
| **Frequency** | LF/HF Ratio | Power balance | $\frac{P_{LF}}{P_{HF}}$ | Dimensionless | Autonomic balance |
| **Frequency** | Spectral Entropy | Complexity | $H = -\sum_f p(f) \log_2 p(f)$ | Bits | Regularity measure |
| **Wavelet** | Level-k Energy | Scale decomposition | $E_k = \sum_j |d_k[j]|^2$ | Energy units | Multi-scale activity |
| **Wavelet** | Level-k Entropy | Scale complexity | $H_k = -\sum_j p_k[j] \log_2 p_k[j]$ | Bits | Scale-specific irregularity |
| **HRV** | SDNN | Overall HRV | $SDNN = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(RR_i - \overline{RR})^2}$ | ms | Autonomic function |
| **HRV** | RMSSD | Short-term HRV | $RMSSD = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i)^2}$ | ms | Parasympathetic activity |
| **HRV** | pNN50 | Beat variability | $pNN50 = \frac{\sum \mathbb{1}[|RR_{i+1} - RR_i| > 50ms]}{N-1}$ | % | Vagal tone |

---

## TABLE 3: FEATURE SELECTION METHODS

| Step | Method | Purpose | Parameters | Mathematical Basis | Assumptions |
|------|--------|---------|------------|-------------------|-------------|
| **1** | Train/Test Split | Prevent data leakage | 80/20, stratified | Random sampling | IID samples per class |
| **2** | Correlation Analysis | Remove redundancy | $|r| > 0.90$ threshold | $r_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$ | Linear correlation sufficient |
| **3** | F-statistic (ANOVA) | Rank discriminability | - | $F = \frac{MS_{between}}{MS_{within}} = \frac{\sum n_j(\bar{x}_j - \bar{x})^2 / (k-1)}{\sum\sum(x_{ij} - \bar{x}_j)^2 / (N-k)}$ | Features normally distributed |
| **4** | p-value Filter | Significance test | $p < 0.05$ | Based on F-distribution | Null hypothesis: no difference |
| **5** | Optimal k Selection | Feature count limit | $k = n_{train}/5$ | Rule of thumb | Prevent overfitting |

---

## TABLE 4: CLASSIFICATION METHODS

| Model | Type | Key Parameters | Mathematical Basis | Strengths | Limitations |
|-------|------|----------------|-------------------|-----------|-------------|
| **Logistic Regression** | Linear | C=1.0, L2 penalty | $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$ | Interpretable, fast | Linear decision boundary |
| **Random Forest** | Ensemble | n_estimators=100, max_depth=10 | $\hat{y} = mode(\{h_b(x)\}_{b=1}^{B})$ | Handles non-linearity, robust | Can overfit small data |
| **Gradient Boosting** | Ensemble | n_estimators=100, lr=0.1 | $F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$ | High accuracy, handles complex patterns | Sensitive to noise |
| **SVM (RBF)** | Kernel | C=1.0, gamma=scale | $f(x) = sign(\sum \alpha_i y_i K(x_i, x) + b)$ | Non-linear boundaries, margin max | Slow on large data |
| **SVM (Linear)** | Linear | C=1.0 | $f(x) = sign(w^T x + b)$ | Fast, interpretable | Linear only |
| **KNN** | Instance | k=5, distance=Euclidean | $\hat{y} = mode(\{y_i : x_i \in N_k(x)\})$ | Simple, no training | Sensitive to k, slow prediction |
| **Decision Tree** | Tree | max_depth=10 | Recursive partitioning on features | Interpretable, fast | Overfits easily |
| **Naive Bayes** | Probabilistic | var_smoothing=1e-9 | $P(y|x) \propto P(y) \prod_{i} P(x_i|y)$ | Fast, probabilistic | Assumes feature independence |
| **AdaBoost** | Ensemble | n_estimators=50 | $H(x) = sign(\sum_{t=1}^{T} \alpha_t h_t(x))$ | Reduces bias | Sensitive to noise |
| **Extra Trees** | Ensemble | n_estimators=100 | Random splits at each node | More randomness than RF | May underfit |
| **LDA** | Linear | solver=svd | Projects to maximize class separation | Dimensionality reduction | Assumes Gaussian classes |
| **Voting Ensemble** | Meta | RF + GB + SVM | $\hat{y} = mode(h_1(x), h_2(x), h_3(x))$ | Combines strengths | Complexity |

---

## TABLE 5: CROSS-VALIDATION METHODS

| Method | Implementation | Parameters | Formula | Use Case |
|--------|---------------|------------|---------|----------|
| **LOSO** | LeaveOneGroupOut | group=subject_id | Train: $\{D \setminus D_i\}$, Test: $\{D_i\}$ for each subject $i$ | Subject-independent validation |
| **Stratified k-Fold** | StratifiedKFold | k=10, shuffle=True | Preserves class ratio in each fold | Within-subject validation |

**LOSO Formulation:**
For $N$ subjects, LOSO accuracy is:
$$ACC_{LOSO} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

where $\hat{y}_i$ is the prediction for subject $i$ when trained on all subjects except $i$.

---

## TABLE 6: EVALUATION METRICS

| Metric | Formula | Interpretation | Range | Threshold |
|--------|---------|---------------|-------|-----------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness | [0, 1] | >0.80 |
| **Precision** | $\frac{TP}{TP + FP}$ | Positive predictive value | [0, 1] | >0.80 |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | True positive rate | [0, 1] | >0.85 |
| **Specificity** | $\frac{TN}{TN + FP}$ | True negative rate | [0, 1] | >0.85 |
| **F1-Score** | $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$ | Harmonic mean | [0, 1] | >0.80 |
| **ROC AUC** | $\int_0^1 TPR(FPR) \, dFPR$ | Ranking quality | [0, 1] | >0.90 |
| **Type I Error (α)** | $\frac{FP}{FP + TN}$ | False positive rate | [0, 1] | <0.05 |
| **Type II Error (β)** | $\frac{FN}{FN + TP}$ | False negative rate | [0, 1] | <0.10 |
| **Statistical Power** | $1 - \beta$ | Detection ability | [0, 1] | >0.90 |
| **95% CI** | $\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}$ | Confidence range | - | - |

---

## TABLE 7: EXPLAINABLE AI (XAI) METHODS

| Method | Type | Formula/Approach | Output | Interpretation Level |
|--------|------|-----------------|--------|---------------------|
| **Model Feature Importance** | Global | $I_j = \sum_{trees} gain_j$ (RF/GB) | Importance score per feature | Which features matter overall |
| **Permutation Importance** | Global | $I_j = acc_{original} - acc_{permuted_j}$ | Accuracy drop when feature shuffled | Unbiased importance |
| **SHAP Values** | Global+Local | $\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(n-|S|-1)!}{n!}[f(S \cup \{j\}) - f(S)]$ | Contribution to prediction | Feature attribution |
| **Feature-Target Correlation** | Global | $r_{feature,label}$ | Correlation coefficient | Direct relationship |
| **LIME** | Local | Local linear approximation | Feature weights for one instance | Individual prediction explanation |

---

## TABLE 8: FINAL SELECTED FEATURES WITH CLINICAL INTERPRETATION

| Rank | Feature Name | Importance | Domain | Clinical Interpretation |
|------|-------------|-----------|--------|------------------------|
| 1 | `resp_zero_crossings` | 0.6997 | Time | **Breathing irregularity** - High values indicate erratic respiration, common in abnormal patients |
| 2 | `num_monitor_rr_min` | 0.0847 | Numerics | **Minimum respiratory rate** - Low minima suggest apnea episodes or bradypnea |
| 3 | `resp_low_freq_power` | 0.0634 | Frequency | **Slow breathing power** - Elevated in patients with slow, labored breathing |
| 4 | `resp_wavelet_L1_entropy` | 0.0512 | Wavelet | **High-frequency complexity** - Irregular fine-scale patterns in abnormal breathing |
| 5 | `resp_wavelet_L2_entropy` | 0.0489 | Wavelet | **Mid-frequency complexity** - Captures breathing pattern variability |
| 6 | `gt_breath_regularity` | 0.0451 | Annotation | **Breath-to-breath consistency** - Lower regularity = abnormal |
| 7 | `resp_wavelet_L4_entropy` | 0.0401 | Wavelet | **Low-frequency complexity** - Irregular slow oscillations in abnormal patients |

---

## KEY ASSUMPTIONS SUMMARY

| Category | Assumption | Justification | Validation |
|----------|-----------|---------------|------------|
| **Data** | Subjects are independent | Different ICU patients | Study design |
| **Preprocessing** | Noise is additive | Standard signal model | Literature |
| **Features** | Linear correlation sufficient for redundancy removal | Computationally efficient | Threshold sensitivity analysis |
| **Feature Selection** | F-statistic appropriate for binary classification | ANOVA is standard | Compared with mutual information |
| **Classification** | Class balance is near 50/50 | 27 Normal vs 26 Abnormal | Dataset statistics |
| **Validation** | LOSO provides unbiased estimate | Gold standard for small biomedical datasets | Literature recommendation |
| **XAI** | SHAP values faithfully represent model | Theoretical guarantee | Lundberg & Lee (2017) |

---

## REFERENCES

1. Orphanidou, C., et al. (2015). IEEE JBHI, 19(3), 832-838.
2. Elgendi, M. (2016). Bioengineering, 3(4), 21.
3. Makowski, D., et al. (2021). Behavior Research Methods, 53(4), 1689-1696.
4. Leys, C., et al. (2013). Journal of Experimental Social Psychology, 49(4), 764-766.
5. Lundberg, S. M., & Lee, S. I. (2017). NeurIPS, 4765-4774.

---

**Document Version:** 1.0  
**Created:** December 20, 2025  
**Project:** Respiratory Abnormality Detection Using PPG
