"""
Configuration Settings for Respiratory Pattern Analysis Project
================================================================
Biomedical Signal Processing Pipeline Configuration
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"

# Original dataset location (WFDB format)
BIDMC_CSV_DIR = PROJECT_ROOT / "bidmc_csv"

# ============================================================================
# SIGNAL ACQUISITION PARAMETERS
# ============================================================================
# ADC (Analog-to-Digital Converter) Specifications
SAMPLING_RATE = 125  # Hz - Sampling frequency
ADC_RESOLUTION = 16  # bits
NYQUIST_FREQ = SAMPLING_RATE / 2  # 62.5 Hz

# Signal channels in dataset
SIGNAL_CHANNELS = {
    'RESP': {'name': 'Respiratory Impedance', 'unit': 'arbitrary'},
    'PLETH': {'name': 'Photoplethysmogram (PPG)', 'unit': 'arbitrary'},
    'II': {'name': 'ECG Lead II', 'unit': 'mV'},
    'V': {'name': 'ECG Lead V', 'unit': 'mV'},
    'AVR': {'name': 'ECG Lead AVR', 'unit': 'mV'},
}

# Signal types list for compatibility
SIGNAL_TYPES = list(SIGNAL_CHANNELS.keys())

# Clinical respiratory rate thresholds (breaths per minute)
RR_LOW = 12   # Lower limit of normal respiratory rate
RR_HIGH = 20  # Upper limit of normal respiratory rate

# ============================================================================
# DIGITAL SIGNAL PROCESSING PARAMETERS
# ============================================================================
# Filtering parameters
FILTER_CONFIG = {
    'respiratory': {
        'lowcut': 0.1,    # Hz - Remove DC and slow drift
        'highcut': 1.0,   # Hz - Remove high-frequency noise
        'order': 4,
        'type': 'bandpass'
    },
    'ppg': {
        'lowcut': 0.5,    # Hz
        'highcut': 8.0,   # Hz
        'order': 4,
        'type': 'bandpass'
    },
    'ecg': {
        'lowcut': 0.5,    # Hz
        'highcut': 40.0,  # Hz
        'order': 4,
        'type': 'bandpass'
    },
    'powerline': {
        'freq': 60.0,     # Hz (US) or 50.0 Hz (EU)
        'Q': 30,          # Quality factor for notch filter
        'type': 'notch'
    }
}

# ============================================================================
# FREQUENCY DOMAIN ANALYSIS PARAMETERS
# ============================================================================
# FFT parameters
FFT_CONFIG = {
    'window': 'hann',           # Window function
    'nperseg': 256,             # Samples per segment
    'noverlap': 128,            # Overlap between segments
    'nfft': 512,                # FFT length (zero-padding)
}

# Respiratory frequency bands (Hz)
RESPIRATORY_BANDS = {
    'very_low': (0.01, 0.04),   # Very low frequency
    'low': (0.04, 0.15),        # Low frequency  
    'high': (0.15, 0.4),        # High frequency (normal breathing)
    'very_high': (0.4, 1.0),    # Very high (tachypnea)
}

# Spectrogram parameters for non-stationary analysis
SPECTROGRAM_CONFIG = {
    'window': 'hann',
    'nperseg': 256,
    'noverlap': 200,
    'mode': 'magnitude'
}

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================
# Statistical features (non-parametric)
STATISTICAL_FEATURES = [
    'mean', 'std', 'var', 'skewness', 'kurtosis', 
    'rms', 'min', 'max', 'range', 'iqr',
    'median', 'p5', 'p25', 'p75', 'p95'
]

# Auto-Regressive model parameters (parametric)
AR_MODEL_ORDER = 10  # Number of AR coefficients

# Wavelet transform parameters
WAVELET_CONFIG = {
    'wavelet': 'db4',           # Daubechies wavelet
    'levels': 5,                # Decomposition levels
    'mode': 'symmetric'         # Signal extension mode
}

# ============================================================================
# CLINICAL THRESHOLDS (from ATS, WHO, GINA guidelines)
# ============================================================================
CLINICAL_THRESHOLDS = {
    'respiratory_rate': {
        'bradypnea': (0, 12),       # Abnormally slow
        'normal': (12, 20),          # Normal adult range
        'mild_tachypnea': (20, 25),  # Mildly elevated
        'moderate_tachypnea': (25, 30),
        'severe_tachypnea': (30, 100)
    },
    'heart_rate': {
        'bradycardia': (0, 60),
        'normal': (60, 100),
        'tachycardia': (100, 200)
    },
    'spo2': {
        'severe_hypoxemia': (0, 85),
        'moderate_hypoxemia': (85, 90),
        'mild_hypoxemia': (90, 95),
        'normal': (95, 100)
    }
}

# ============================================================================
# MULTIVARIATE ANALYSIS PARAMETERS
# ============================================================================
# Measurement scales for variables
MEASUREMENT_SCALES = {
    # Metric (Quantitative) - Ratio scale
    'respiratory_rate': {'scale': 'ratio', 'unit': 'breaths/min'},
    'heart_rate': {'scale': 'ratio', 'unit': 'bpm'},
    'spo2': {'scale': 'ratio', 'unit': '%'},
    'signal_amplitude': {'scale': 'ratio', 'unit': 'arbitrary'},
    
    # Metric (Quantitative) - Interval scale
    'spectral_entropy': {'scale': 'interval', 'unit': 'bits'},
    
    # Non-metric (Qualitative) - Ordinal scale
    'severity_class': {'scale': 'ordinal', 'categories': ['Normal', 'Abnormal']},
    
    # Non-metric (Qualitative) - Nominal scale  
    'subject_id': {'scale': 'nominal', 'type': 'identifier'}
}

# PCA parameters
PCA_CONFIG = {
    'n_components': 0.95,  # Retain 95% variance
    'whiten': True
}

# ============================================================================
# CLASSIFICATION PARAMETERS
# ============================================================================
# Train/test split
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5

# Statistical significance
ALPHA = 0.05  # Type I error rate (false positive)

# Model hyperparameters
MODEL_CONFIG = {
    'gradient_boosting': {
        'n_estimators': 150,
        'max_depth': 4,
        'learning_rate': 0.1,
        'min_samples_split': 3
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 8,
        'min_samples_split': 3,
        'class_weight': 'balanced'
    },
    'svm': {
        'kernel': 'rbf',
        'C': 10.0,
        'gamma': 'scale',
        'class_weight': 'balanced'
    }
}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'
RANDOM_SEED = 42
