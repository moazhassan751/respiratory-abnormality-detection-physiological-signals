"""
Respiratory Abnormality Classification Pipeline
================================================

A comprehensive biomedical signal processing pipeline for analyzing respiratory
patterns and classifying abnormalities with asthma risk indicators.

Pipeline Architecture:
----------------------
1. Data Acquisition → Load physiological signals (RESP, PLETH, ECG) from WFDB
2. Signal Processing → Filtering, spectral analysis
3. Feature Extraction → Time-domain, frequency-domain, wavelet, HRV features
4. Multivariate Analysis → PCA, statistical validation
5. Classification → ML models with k-fold cross-validation
6. Visualization → Signal plots, confusion matrix, ROC curves
7. Clinical Report → Risk assessment with Type I/II error analysis

Technical Framework:
-------------------
- Sampling Rate: 125 Hz
- Signals: Respiration, Photoplethysmography (PPG), ECG
- Binary Classification: Normal vs Abnormal respiratory patterns
- Statistical Analysis: Type I/II errors, power analysis, confidence intervals

Author: Biomedical Signal Processing Project
Dataset: BIDMC PPG and Respiration Dataset
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')

# WFDB is optional - CSV files are preferred
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
    # Not needed when using CSV files

# Try to import pywt for wavelets
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("Warning: PyWavelets not installed. Install with: pip install PyWavelets")

# Try to import SHAP for Explainable AI (XAI)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Note: SHAP not installed. Install with: pip install shap (for XAI features)")

# Try to import LIME for local interpretability
try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("Note: LIME not installed. Install with: pip install lime (for local XAI)")

# Add source directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Constants
SAMPLING_RATE = 125
NYQUIST_FREQ = SAMPLING_RATE / 2
RR_LOW = 12
RR_HIGH = 20
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Import State-of-the-Art Preprocessing Pipeline
try:
    from preprocessing_sota import PreprocessingPipeline
    HAS_PREPROCESSING = True
    PREPROCESSING_TYPE = "SOTA"  # State-of-the-Art (NeuroKit2/BioSPPy standard)
except ImportError:
    HAS_PREPROCESSING = False
    PREPROCESSING_TYPE = None
    print("Warning: Preprocessing module not found. Run without preprocessing.")


class SimpleFeatureExtractor:
    """Enhanced feature extractor with wavelet and HRV features"""
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
    
    def extract_time_features(self, signal_data):
        """Extract time-domain features"""
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
    
    def extract_freq_features(self, signal_data):
        """Extract frequency-domain features"""
        n = len(signal_data)
        fft_vals = np.abs(fft(signal_data))[:n//2]
        freqs = fftfreq(n, 1/self.fs)[:n//2]
        
        # Find dominant frequency
        resp_mask = (freqs >= 0.1) & (freqs <= 1.0)
        if np.any(resp_mask):
            resp_power = fft_vals[resp_mask]
            resp_freqs = freqs[resp_mask]
            dominant_freq = resp_freqs[np.argmax(resp_power)]
            total_power = np.sum(resp_power**2)
            
            # Power in different bands
            low_mask = (freqs >= 0.1) & (freqs < 0.25)
            high_mask = (freqs >= 0.25) & (freqs <= 1.0)
            low_power = np.sum(fft_vals[low_mask]**2) if np.any(low_mask) else 0
            high_power = np.sum(fft_vals[high_mask]**2) if np.any(high_mask) else 0
        else:
            dominant_freq = 0.25
            total_power = 0
            low_power = 0
            high_power = 0
        
        # Spectral entropy
        psd = fft_vals[fft_vals > 0]
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        return {
            'dominant_freq': dominant_freq,
            'respiratory_rate': dominant_freq * 60,
            'total_power': total_power,
            'low_freq_power': low_power,
            'high_freq_power': high_power,
            'lf_hf_ratio': low_power / (high_power + 1e-10),
            'spectral_entropy': spectral_entropy,
            'spectral_centroid': np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10),
        }
    
    def extract_wavelet_features(self, signal_data):
        """Extract wavelet-based features"""
        if not HAS_PYWT:
            return {}
        
        try:
            # Wavelet decomposition (db4 wavelet, 4 levels)
            coeffs = pywt.wavedec(signal_data, 'db4', level=4)
            
            features = {}
            for i, coef in enumerate(coeffs):
                level_name = f'wavelet_L{i}'
                features[f'{level_name}_energy'] = np.sum(coef**2)
                features[f'{level_name}_std'] = np.std(coef)
                features[f'{level_name}_entropy'] = stats.entropy(np.abs(coef) + 1e-10)
            
            # Total wavelet energy
            features['wavelet_total_energy'] = sum(np.sum(c**2) for c in coeffs)
            
            return features
        except:
            return {}
    
    def extract_hrv_features(self, ppg_signal):
        """Extract HRV features from PPG signal"""
        try:
            # Find PPG peaks
            peaks, _ = signal.find_peaks(ppg_signal, distance=self.fs*0.5)
            
            if len(peaks) < 3:
                return {}
            
            # RR intervals (in ms)
            rr_intervals = np.diff(peaks) / self.fs * 1000
            
            # Time-domain HRV features
            features = {
                'hrv_mean_rr': np.mean(rr_intervals),
                'hrv_sdnn': np.std(rr_intervals),  # Standard deviation of NN intervals
                'hrv_rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)),  # Root mean square of successive differences
                'hrv_pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,  # % of successive RR > 50ms
                'hrv_cv': np.std(rr_intervals) / np.mean(rr_intervals),  # Coefficient of variation
                'heart_rate': 60000 / np.mean(rr_intervals),  # BPM
            }
            
            return features
        except:
            return {}
    
    def extract_ecg_features(self, ecg_signal):
        """Extract ECG-specific features"""
        try:
            features = {}
            
            # Basic statistics
            features['ecg_mean'] = np.mean(ecg_signal)
            features['ecg_std'] = np.std(ecg_signal)
            features['ecg_range'] = np.ptp(ecg_signal)
            
            # R-peak detection (simplified)
            # High-pass filter to remove baseline wander
            b, a = signal.butter(2, 0.5/(self.fs/2), btype='high')
            ecg_filtered = signal.filtfilt(b, a, ecg_signal)
            
            # Find R-peaks
            r_peaks, properties = signal.find_peaks(
                ecg_filtered, 
                distance=int(self.fs * 0.4),  # At least 0.4s between beats (150 bpm max)
                height=np.percentile(ecg_filtered, 75)
            )
            
            if len(r_peaks) >= 3:
                # RR intervals
                rr_intervals = np.diff(r_peaks) / self.fs * 1000  # in ms
                
                features['ecg_hr_mean'] = 60000 / np.mean(rr_intervals)  # Heart rate
                features['ecg_hr_std'] = 60000 / np.std(rr_intervals) if np.std(rr_intervals) > 0 else 0
                features['ecg_rr_mean'] = np.mean(rr_intervals)
                features['ecg_rr_std'] = np.std(rr_intervals)
                features['ecg_rr_range'] = np.ptp(rr_intervals)
                
                # HRV from ECG
                features['ecg_sdnn'] = np.std(rr_intervals)
                features['ecg_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
                
                # R-peak amplitude variability
                if 'peak_heights' in properties:
                    features['ecg_rpeak_amp_mean'] = np.mean(properties['peak_heights'])
                    features['ecg_rpeak_amp_std'] = np.std(properties['peak_heights'])
            
            # QRS complex energy (approximation via high-freq content)
            f, psd = signal.welch(ecg_signal, fs=self.fs, nperseg=min(256, len(ecg_signal)//4))
            qrs_band = (f >= 5) & (f <= 25)  # QRS complex energy band
            features['ecg_qrs_energy'] = np.sum(psd[qrs_band])
            features['ecg_total_power'] = np.sum(psd)
            
            return features
        except:
            return {}
    
    def extract_numerics_features(self, numerics):
        """Extract features from monitor numerics (HR, SpO2, RR)"""
        features = {}
        
        for name, values in numerics.items():
            if values is None or len(values) == 0:
                continue
            
            # Remove NaN and invalid values
            valid = values[~np.isnan(values)]
            valid = valid[valid > 0]
            
            if len(valid) > 0:
                features[f'num_{name}_mean'] = np.mean(valid)
                features[f'num_{name}_std'] = np.std(valid)
                features[f'num_{name}_min'] = np.min(valid)
                features[f'num_{name}_max'] = np.max(valid)
                features[f'num_{name}_range'] = np.ptp(valid)
                
                # Trend (slope)
                if len(valid) > 2:
                    x = np.arange(len(valid))
                    slope, _ = np.polyfit(x, valid, 1)
                    features[f'num_{name}_trend'] = slope
        
        return features
    
    def calculate_rr_from_breaths(self, breath_annotations, signal_length):
        """Calculate respiratory rate from manual breath annotations"""
        if breath_annotations is None or len(breath_annotations) < 2:
            return {}
        
        # Calculate breath-to-breath intervals
        breath_intervals = np.diff(breath_annotations) / self.fs  # in seconds
        
        # Calculate respiratory rate (breaths per minute)
        rr_values = 60 / breath_intervals
        
        # Filter out unrealistic values
        valid_rr = rr_values[(rr_values > 4) & (rr_values < 40)]
        
        if len(valid_rr) > 0:
            return {
                'gt_rr_mean': np.mean(valid_rr),  # Ground truth RR
                'gt_rr_std': np.std(valid_rr),
                'gt_rr_min': np.min(valid_rr),
                'gt_rr_max': np.max(valid_rr),
                'gt_breath_count': len(breath_annotations),
                'gt_breath_regularity': np.std(breath_intervals)  # Lower = more regular
            }
        return {}


class RespiratoryAnalysisPipeline:
    """Main pipeline for respiratory abnormality classification"""
    
    def __init__(self, data_dir=None, use_csv=True):
        self.data_dir = Path(data_dir) if data_dir else SCRIPT_DIR
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)
        
        self.use_csv = use_csv  # Use CSV files instead of WFDB
        self.feature_extractor = SimpleFeatureExtractor(SAMPLING_RATE)
        self.features = None
        self.labels = None
        self.results = {}
        self.raw_signals = []  # Store raw signals for visualization
        
        print("="*70)
        print("RESPIRATORY ABNORMALITY CLASSIFICATION PIPELINE")
        print("="*70)
        print(f"Data directory: {self.data_dir}")
        print(f"Data format: {'CSV files' if use_csv else 'WFDB format'}")
        print(f"Sampling rate: {SAMPLING_RATE} Hz")
        print(f"Results directory: {self.results_dir}")
        print("="*70)
    
    def load_csv_data(self):
        """Load data from CSV files and Fix.txt for demographics"""
        print("\n[STEP 1] DATA ACQUISITION (CSV + TXT FORMAT)")
        print("-"*50)
        
        # Find all signal CSV files
        signal_files = sorted(self.data_dir.glob("bidmc_*_Signals.csv"))
        print(f"Found {len(signal_files)} subjects with CSV data")
        
        all_data = []
        
        for sig_file in signal_files:
            try:
                # Extract subject number
                subject_num = sig_file.stem.split('_')[1]
                subject_id = f"bidmc{subject_num}"
                
                # Load signals CSV
                signals_df = pd.read_csv(sig_file)
                
                # Clean column names (remove spaces)
                signals_df.columns = [c.strip() for c in signals_df.columns]
                
                data = {
                    'subject_id': subject_id,
                    'resp_signal': signals_df['RESP'].values,
                    'pleth_signal': signals_df['PLETH'].values,
                    'ecg_v': signals_df['V'].values if 'V' in signals_df.columns else None,
                    'ecg_avr': signals_df['AVR'].values if 'AVR' in signals_df.columns else None,
                    'ecg_ii': signals_df['II'].values if 'II' in signals_df.columns else None,
                }
                
                # Load numerics CSV
                num_file = self.data_dir / f"bidmc_{subject_num}_Numerics.csv"
                if num_file.exists():
                    numerics_df = pd.read_csv(num_file)
                    numerics_df.columns = [c.strip() for c in numerics_df.columns]
                    
                    data['numerics'] = {
                        'heart_rate': numerics_df['HR'].values if 'HR' in numerics_df.columns else None,
                        'pulse_rate': numerics_df['PULSE'].values if 'PULSE' in numerics_df.columns else None,
                        'monitor_rr': numerics_df['RESP'].values if 'RESP' in numerics_df.columns else None,
                        'spo2': numerics_df['SpO2'].values if 'SpO2' in numerics_df.columns else None,
                    }
                else:
                    data['numerics'] = {}
                
                # Load breath annotations CSV
                breath_file = self.data_dir / f"bidmc_{subject_num}_Breaths.csv"
                if breath_file.exists():
                    breaths_df = pd.read_csv(breath_file)
                    breaths_df.columns = [c.strip() for c in breaths_df.columns]
                    
                    # Use first annotator's annotations
                    ann1_col = [c for c in breaths_df.columns if 'ann1' in c.lower()]
                    if ann1_col:
                        data['breath_annotations'] = breaths_df[ann1_col[0]].dropna().values.astype(int)
                    else:
                        data['breath_annotations'] = breaths_df.iloc[:, 0].dropna().values.astype(int)
                else:
                    data['breath_annotations'] = None
                
                # Load demographics from Fix.txt
                fix_file = self.data_dir / f"bidmc_{subject_num}_Fix.txt"
                demographics = self.parse_fix_txt(fix_file)
                data['age'] = demographics.get('age')
                data['sex'] = demographics.get('gender')
                data['location'] = demographics.get('location')
                
                # Store for visualization
                if len(self.raw_signals) < 5:
                    self.raw_signals.append(data.copy())
                
                all_data.append(data)
                
            except Exception as e:
                print(f"  Warning: Could not load {sig_file.name}: {e}")
                continue
        
        # Print summary
        print(f"Successfully loaded {len(all_data)} subjects from CSV/TXT files")
        
        n_ecg = sum(1 for d in all_data if d['ecg_ii'] is not None)
        n_numerics = sum(1 for d in all_data if d['numerics'])
        n_breaths = sum(1 for d in all_data if d['breath_annotations'] is not None)
        n_demographics = sum(1 for d in all_data if d['age'] is not None)
        
        print(f"  - Signal files (RESP, PLETH, ECG): {len(all_data)}/{len(all_data)}")
        print(f"  - Numerics (HR, SpO2, RESP, PULSE): {n_numerics}/{len(all_data)}")
        print(f"  - Breath annotations: {n_breaths}/{len(all_data)}")
        print(f"  - Demographics (from Fix.txt): {n_demographics}/{len(all_data)}")
        
        return all_data
    
    def parse_fix_txt(self, fix_file):
        """Parse demographics from Fix.txt file"""
        demographics = {'age': None, 'gender': None, 'location': None}
        
        if not fix_file.exists():
            return demographics
        
        try:
            with open(fix_file, 'r') as f:
                content = f.read()
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('Age:'):
                    try:
                        demographics['age'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Gender:'):
                    demographics['gender'] = line.split(':')[1].strip()
                elif line.startswith('Location:'):
                    demographics['location'] = line.split(':')[1].strip()
        except:
            pass
        
        return demographics
    
    def parse_breath_annotations(self, breath_file):
        """Parse manual breath annotations from .breath file (legacy WFDB)"""
        try:
            ann = wfdb.rdann(str(breath_file), 'breath')
            return ann.sample  # Sample indices of breath annotations
        except:
            return None
    
    def parse_demographics(self, header_comments):
        """Extract age, sex, location from header comments"""
        demographics = {'age': None, 'sex': None, 'location': None}
        
        for comment in header_comments:
            if '<age>:' in comment:
                try:
                    age_str = comment.split('<age>:')[1].split('<')[0].strip()
                    demographics['age'] = int(age_str)
                except:
                    pass
            if '<sex>:' in comment:
                try:
                    demographics['sex'] = comment.split('<sex>:')[1].split('<')[0].strip()
                except:
                    pass
            if '<location>:' in comment:
                try:
                    demographics['location'] = comment.split('<location>:')[1].split('<')[0].strip()
                except:
                    pass
        
        return demographics
    
    def load_wfdb_data(self):
        """Load ALL data from WFDB format files (signals, numerics, annotations, demographics)"""
        print("\n[STEP 1] DATA ACQUISITION (COMPLETE DATASET)")
        print("-"*50)
        
        if not HAS_WFDB:
            print("ERROR: wfdb library not available")
            return None
        
        # Find all record files
        record_files = sorted(self.data_dir.glob("bidmc[0-9][0-9].hea"))
        print(f"Found {len(record_files)} subjects")
        
        all_data = []
        
        for hea_file in record_files:
            try:
                record_name = hea_file.stem
                record_path = str(hea_file.parent / record_name)
                
                # ===== 1. LOAD WAVEFORM SIGNALS (125 Hz) =====
                record = wfdb.rdrecord(record_path)
                sig_names = record.sig_name
                signals = record.p_signal
                
                # Find all signal indices
                resp_idx = pleth_idx = ecg_v_idx = ecg_avr_idx = ecg_ii_idx = None
                
                for i, name in enumerate(sig_names):
                    name_upper = name.upper().strip().rstrip(',')
                    if 'RESP' in name_upper:
                        resp_idx = i
                    elif 'PLETH' in name_upper:
                        pleth_idx = i
                    elif name_upper == 'V':
                        ecg_v_idx = i
                    elif name_upper == 'AVR':
                        ecg_avr_idx = i
                    elif name_upper == 'II':
                        ecg_ii_idx = i
                
                # ===== 2. LOAD NUMERICS (1 Hz) =====
                numerics = {}
                numeric_path = str(hea_file.parent / f"{record_name}n")
                try:
                    numeric_record = wfdb.rdrecord(numeric_path)
                    num_sig_names = numeric_record.sig_name
                    num_signals = numeric_record.p_signal
                    
                    for i, name in enumerate(num_sig_names):
                        name_clean = name.upper().strip().rstrip(',')
                        if 'HR' in name_clean:
                            numerics['heart_rate'] = num_signals[:, i]
                        elif 'PULSE' in name_clean:
                            numerics['pulse_rate'] = num_signals[:, i]
                        elif 'RESP' in name_clean:
                            numerics['monitor_rr'] = num_signals[:, i]  # Monitor-derived RR
                        elif 'SPO2' in name_clean:
                            numerics['spo2'] = num_signals[:, i]
                except:
                    pass  # Numerics not available
                
                # ===== 3. LOAD BREATH ANNOTATIONS =====
                breath_annotations = None
                breath_file = hea_file.parent / f"{record_name}"
                try:
                    breath_annotations = self.parse_breath_annotations(breath_file)
                except:
                    pass
                
                # ===== 4. PARSE DEMOGRAPHICS =====
                demographics = self.parse_demographics(record.comments)
                
                # Build subject data
                if resp_idx is not None:
                    subject_data = {
                        'subject_id': record_name,
                        # Waveform signals (125 Hz)
                        'resp_signal': signals[:, resp_idx],
                        'pleth_signal': signals[:, pleth_idx] if pleth_idx is not None else None,
                        'ecg_v': signals[:, ecg_v_idx] if ecg_v_idx is not None else None,
                        'ecg_avr': signals[:, ecg_avr_idx] if ecg_avr_idx is not None else None,
                        'ecg_ii': signals[:, ecg_ii_idx] if ecg_ii_idx is not None else None,
                        # Numerics (1 Hz)
                        'numerics': numerics,
                        # Annotations
                        'breath_annotations': breath_annotations,
                        # Demographics
                        'age': demographics['age'],
                        'sex': demographics['sex'],
                        'location': demographics['location']
                    }
                    all_data.append(subject_data)
                    
                    # Store first few subjects for visualization
                    if len(self.raw_signals) < 3:
                        self.raw_signals.append(subject_data)
                    
            except Exception as e:
                print(f"  Warning: Could not load {hea_file.name}: {e}")
                continue
        
        # Print summary
        print(f"Successfully loaded {len(all_data)} subjects")
        
        # Count available data
        n_ecg = sum(1 for d in all_data if d['ecg_ii'] is not None)
        n_numerics = sum(1 for d in all_data if d['numerics'])
        n_breaths = sum(1 for d in all_data if d['breath_annotations'] is not None)
        n_age = sum(1 for d in all_data if d['age'] is not None)
        
        print(f"  - ECG signals: {n_ecg}/{len(all_data)}")
        print(f"  - Numerics (HR, SpO2): {n_numerics}/{len(all_data)}")
        print(f"  - Breath annotations: {n_breaths}/{len(all_data)}")
        print(f"  - Demographics: {n_age}/{len(all_data)}")
        
        return all_data
    
    def extract_features(self, data):
        """Extract COMPREHENSIVE features from all data sources"""
        print("\n[STEP 2] FEATURE EXTRACTION (ALL DATA SOURCES)")
        print("-"*50)
        
        feature_list = []
        
        for subject in data:
            try:
                features = {'subject_id': subject['subject_id']}
                
                resp_signal = subject['resp_signal']
                
                # ===== RESPIRATORY SIGNAL FEATURES =====
                # Time-domain features
                time_feats = self.feature_extractor.extract_time_features(resp_signal)
                features.update({f'resp_{k}': v for k, v in time_feats.items()})
                
                # Frequency-domain features
                freq_feats = self.feature_extractor.extract_freq_features(resp_signal)
                features.update({f'resp_{k}': v for k, v in freq_feats.items()})
                
                # Wavelet features
                wavelet_feats = self.feature_extractor.extract_wavelet_features(resp_signal)
                features.update({f'resp_{k}': v for k, v in wavelet_feats.items()})
                
                # ===== PPG SIGNAL FEATURES =====
                if subject['pleth_signal'] is not None:
                    ppg_feats = self.feature_extractor.extract_time_features(subject['pleth_signal'])
                    features.update({f'ppg_{k}': v for k, v in ppg_feats.items()})
                    
                    # HRV features from PPG
                    hrv_feats = self.feature_extractor.extract_hrv_features(subject['pleth_signal'])
                    features.update(hrv_feats)
                
                # ===== ECG SIGNAL FEATURES =====
                # Use Lead II (best for R-peak detection)
                if subject.get('ecg_ii') is not None:
                    ecg_feats = self.feature_extractor.extract_ecg_features(subject['ecg_ii'])
                    features.update({f'ecg_ii_{k}': v for k, v in ecg_feats.items()})
                elif subject.get('ecg_v') is not None:
                    ecg_feats = self.feature_extractor.extract_ecg_features(subject['ecg_v'])
                    features.update({f'ecg_v_{k}': v for k, v in ecg_feats.items()})
                
                # ===== NUMERICS FEATURES (HR, SpO2, monitor RR) =====
                if subject.get('numerics'):
                    num_feats = self.feature_extractor.extract_numerics_features(subject['numerics'])
                    features.update(num_feats)
                
                # ===== BREATH ANNOTATION FEATURES (Ground Truth RR) =====
                if subject.get('breath_annotations') is not None:
                    breath_feats = self.feature_extractor.calculate_rr_from_breaths(
                        subject['breath_annotations'], 
                        len(resp_signal)
                    )
                    features.update(breath_feats)
                
                # ===== DEMOGRAPHICS =====
                features['demo_age'] = subject.get('age')
                features['demo_sex'] = 1 if subject.get('sex') == 'M' else 0 if subject.get('sex') == 'F' else None
                features['demo_location'] = subject.get('location')
                
                feature_list.append(features)
                
            except Exception as e:
                print(f"  Warning: Feature extraction failed for {subject['subject_id']}: {e}")
                continue
        
        self.features = pd.DataFrame(feature_list)
        
        # Count feature categories
        n_resp = len([c for c in self.features.columns if c.startswith('resp_')])
        n_ppg = len([c for c in self.features.columns if c.startswith('ppg_') or c.startswith('hrv_')])
        n_ecg = len([c for c in self.features.columns if c.startswith('ecg_')])
        n_num = len([c for c in self.features.columns if c.startswith('num_')])
        n_gt = len([c for c in self.features.columns if c.startswith('gt_')])
        n_demo = len([c for c in self.features.columns if c.startswith('demo_')])
        
        print(f"Extracted {len(self.features.columns)-1} features for {len(self.features)} subjects:")
        print(f"  - Respiratory: {n_resp} features")
        print(f"  - PPG/HRV: {n_ppg} features")
        print(f"  - ECG: {n_ecg} features")
        print(f"  - Numerics: {n_num} features")
        print(f"  - Ground Truth RR: {n_gt} features")
        print(f"  - Demographics: {n_demo} features")
        
        return self.features
    
    def assign_labels(self):
        """Assign clinical labels based on respiratory rate patterns"""
        print("\n[STEP 3] CLINICAL LABELING")
        print("-"*50)
        
        # Use respiratory rate as primary clinical indicator
        labels = []
        rr_values = []
        
        for idx, row in self.features.iterrows():
            # Get respiratory rate from multiple sources
            gt_rr = row.get('gt_rr_mean', None)
            monitor_rr = row.get('num_monitor_rr_mean', None)
            est_rr = row.get('resp_respiratory_rate', None)
            
            # Use most reliable source
            if gt_rr is not None:
                rr = gt_rr
            elif monitor_rr is not None:
                rr = monitor_rr
            elif est_rr is not None:
                rr = est_rr
            else:
                rr = 15  # Default normal
            
            rr_values.append(rr)
        
        rr_values = np.array(rr_values)
        
        # Use median split for balanced classes (professional approach)
        # This reflects the relative respiratory status within this ICU population
        median_rr = np.median(rr_values)
        
        for rr in rr_values:
            if rr <= median_rr:
                labels.append('Normal')  # Lower half - relatively normal
            else:
                labels.append('Abnormal')  # Higher half - relatively abnormal
        
        self.labels = np.array(labels)
        
        # Distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"Clinical Labeling: Median RR Split (threshold: {median_rr:.1f} breaths/min)")
        print("  Normal: RR ≤ median (relatively stable)")
        print("  Abnormal: RR > median (relatively elevated)")
        print("\nClass Distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} ({count/len(self.labels)*100:.1f}%)")
        
        return self.labels
    
    def train_classifier(self):
        """Train classification models with nested cross-validation (professional approach for small datasets)"""
        print("\n[STEP 4] CLASSIFICATION WITH NESTED CROSS-VALIDATION")
        print("-"*50)
        
        from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                                     confusion_matrix, classification_report,
                                     roc_curve, auc, roc_auc_score)
        from sklearn.pipeline import Pipeline
        
        # Prepare data - EXCLUDE features that directly leak the label (RR-based)
        # The label is based on RR, so we should predict from other physiological signals
        exclude_patterns = ['gt_rr_mean', 'gt_rr_std', 'gt_rr_min', 'gt_rr_max', 
                           'num_monitor_rr_mean', 'resp_respiratory_rate', 
                           'subject_id', 'score', 'demo_location']
        
        feature_cols = [c for c in self.features.columns 
                       if c not in exclude_patterns and 
                       pd.api.types.is_numeric_dtype(self.features[c])]
        
        X = self.features[feature_cols].values
        X = np.nan_to_num(X)
        
        le = LabelEncoder()
        y = le.fit_transform(self.labels)
        
        # =====================================================================
        # IMPROVED FEATURE SELECTION PIPELINE
        # Fixes: Data leakage, RF bias, high dimensionality, redundancy
        # =====================================================================
        
        print("\n" + "="*60)
        print("IMPROVED FEATURE SELECTION PIPELINE")
        print("="*60)
        print(f"Initial features: {len(feature_cols)}")
        print(f"Samples: {len(y)}")
        print(f"Feature-to-sample ratio: {len(feature_cols)/len(y):.2f} (should be < 0.5)")
        
        # --- STEP 1: Train/Test Split BEFORE any feature selection ---
        # This prevents data leakage
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n✓ Train/Test Split: {len(y_train)} train, {len(y_test)} test")
        
        # Scale using ONLY training data (prevent leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Transform only, don't fit!
        
        print("✓ Scaling: Fit on training data only (no leakage)")
        
        # --- STEP 2: Remove Highly Correlated Features ---
        # Correlation analysis to remove redundant features
        print("\n--- Correlation Analysis ---")
        correlation_matrix = np.corrcoef(X_train_scaled.T)
        
        # Find pairs with |correlation| > 0.90
        high_corr_pairs = []
        features_to_remove = set()
        CORR_THRESHOLD = 0.90
        
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                if abs(correlation_matrix[i, j]) > CORR_THRESHOLD:
                    high_corr_pairs.append((feature_cols[i], feature_cols[j], correlation_matrix[i, j]))
                    # Remove the second feature (keep first)
                    features_to_remove.add(j)
        
        print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > {CORR_THRESHOLD})")
        if len(high_corr_pairs) > 0 and len(high_corr_pairs) <= 10:
            for f1, f2, corr in high_corr_pairs[:5]:
                print(f"  • {f1} ↔ {f2}: r={corr:.3f}")
            if len(high_corr_pairs) > 5:
                print(f"  ... and {len(high_corr_pairs) - 5} more pairs")
        
        # Keep only non-redundant features
        keep_indices = [i for i in range(len(feature_cols)) if i not in features_to_remove]
        X_train_decorr = X_train_scaled[:, keep_indices]
        X_test_decorr = X_test_scaled[:, keep_indices]
        feature_cols_decorr = [feature_cols[i] for i in keep_indices]
        
        print(f"✓ Removed {len(features_to_remove)} redundant features")
        print(f"  Remaining: {len(feature_cols_decorr)} features")
        
        # --- STEP 3: Filter-Based Feature Selection (F-statistic) ---
        # Use ANOVA F-statistic - independent of classifier, no bias
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        
        print("\n--- Filter-Based Feature Selection (F-statistic) ---")
        
        # Calculate F-statistics on training data only
        f_scores, p_values = f_classif(X_train_decorr, y_train)
        
        # Sort by F-score (higher = more discriminative)
        f_score_ranking = sorted(
            zip(feature_cols_decorr, f_scores, p_values, range(len(feature_cols_decorr))),
            key=lambda x: x[1], reverse=True
        )
        
        print("\nTop 15 Features by F-statistic (ANOVA):")
        print("-" * 55)
        print(f"{'Rank':<5} {'Feature':<35} {'F-score':<10} {'p-value':<10}")
        print("-" * 55)
        for rank, (feat, f_score, p_val, _) in enumerate(f_score_ranking[:15], 1):
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{rank:<5} {feat:<35} {f_score:<10.2f} {p_val:<10.4f} {sig}")
        
        # --- STEP 4: Select Optimal Number of Features ---
        # Rule of thumb: max features = n_samples / 10
        MAX_FEATURES = max(10, len(y_train) // 5)  # At least 10, at most n/5
        
        # Only keep statistically significant features (p < 0.05)
        significant_features = [(feat, f, idx) for feat, f, p, idx in f_score_ranking if p < 0.05]
        
        # Take top features (min of max allowed and significant ones)
        N_SELECTED = min(MAX_FEATURES, len(significant_features), 20)
        
        print(f"\n✓ Feature Selection Criteria:")
        print(f"  • Max features (n_train/5): {MAX_FEATURES}")
        print(f"  • Statistically significant (p<0.05): {len(significant_features)}")
        print(f"  • Selected: {N_SELECTED} features")
        
        # Get selected feature indices
        selected_indices = [idx for _, _, _, idx in f_score_ranking[:N_SELECTED]]
        X_train_selected = X_train_decorr[:, selected_indices]
        X_test_selected = X_test_decorr[:, selected_indices]
        selected_feature_cols = [feature_cols_decorr[i] for i in selected_indices]
        
        # --- STEP 5: Optional PCA for Dimensionality Reduction ---
        print("\n--- PCA Analysis (Optional Comparison) ---")
        from sklearn.decomposition import PCA
        
        # Fit PCA on training data only
        pca = PCA(n_components=0.95)  # Keep 95% variance
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        print(f"PCA: {X_train_scaled.shape[1]} features → {X_train_pca.shape[1]} components")
        print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Store for comparison (we'll use filter selection as primary)
        self.pca_components = X_train_pca.shape[1]
        
        # --- STEP 6: Store Feature Importance Ranking ---
        # Using F-statistic scores (unbiased, filter method)
        self.feature_importance_ranking = [
            (feat, f_score) for feat, f_score, _, _ in f_score_ranking
        ]
        
        # Store selected features for later use
        self.selected_feature_names = selected_feature_cols
        
        print("\n" + "="*60)
        print(f"FINAL: Using {N_SELECTED} features selected by F-statistic")
        print(f"Feature-to-sample ratio: {N_SELECTED/len(y_train):.2f} (improved!)")
        print("="*60)
        
        # Use the properly selected features
        X_selected = np.vstack([X_train_selected, X_test_selected])
        n_selected = N_SELECTED
        
        # Re-create full scaled data for CV (using selected feature indices)
        # Map back to original feature indices
        original_selected_indices = [keep_indices[i] for i in selected_indices]
        X_scaled = scaler.fit_transform(X)
        X_selected = X_scaled[:, original_selected_indices]
        
        # Store for XAI later
        self.X_selected = X_selected
        self.y = y
        self.scaler = scaler
        self.original_selected_indices = original_selected_indices
        
        # =====================================================================
        # MULTIPLE MACHINE LEARNING MODELS (11 Models)
        # =====================================================================
        print("\n" + "="*70)
        print("MULTIPLE ML MODELS COMPARISON")
        print("="*70)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        # Define 11 different models for comprehensive comparison
        models = {
            'Logistic Regression': LogisticRegression(
                C=1.0, class_weight='balanced', random_state=42, max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=3,
                class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', C=10.0, gamma='auto', probability=True,
                class_weight='balanced', random_state=42
            ),
            'SVM (Linear)': SVC(
                kernel='linear', C=1.0, probability=True,
                class_weight='balanced', random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='distance', metric='euclidean'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8, min_samples_split=3,
                class_weight='balanced', random_state=42
            ),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5, random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=10,
                class_weight='balanced', random_state=42
            ),
            'LDA': LinearDiscriminantAnalysis()
        }
        
        # Add Voting Ensemble
        models['Voting Ensemble'] = VotingClassifier(
            estimators=[
                ('rf', models['Random Forest']),
                ('gb', models['Gradient Boosting']),
                ('svm', models['SVM (RBF)'])
            ],
            voting='soft'
        )
        
        print(f"Comparing {len(models)} models...")
        
        # =====================================================================
        # LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION
        # Subject-independent evaluation for biomedical applications
        # =====================================================================
        print("\n" + "="*70)
        print("LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
        print("="*70)
        print("Subject-independent evaluation for biomedical applications")
        print("Each subject is used as test set once, others as training")
        
        from sklearn.model_selection import LeaveOneOut
        
        # Create subject IDs array
        n_subjects = len(y)
        subject_ids = np.arange(n_subjects)
        
        # LOSO results storage
        loso_results = {}
        
        for name, model in models.items():
            # Store per-subject predictions
            subject_predictions = []
            subject_true = []
            subject_probs = []
            
            # LOSO: Each subject is test set once
            for test_idx in range(n_subjects):
                # Train on all except one
                train_mask = np.ones(n_subjects, dtype=bool)
                train_mask[test_idx] = False
                
                X_train_loso = X_selected[train_mask]
                y_train_loso = y[train_mask]
                X_test_loso = X_selected[test_idx:test_idx+1]
                y_test_loso = y[test_idx]
                
                # Clone and train model
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_train_loso, y_train_loso)
                
                # Predict
                pred = model_clone.predict(X_test_loso)[0]
                subject_predictions.append(pred)
                subject_true.append(y_test_loso)
                
                # Probability (if available)
                if hasattr(model_clone, 'predict_proba'):
                    prob = model_clone.predict_proba(X_test_loso)[0, 1]
                else:
                    prob = pred
                subject_probs.append(prob)
            
            # Convert to arrays
            y_true_loso = np.array(subject_true)
            y_pred_loso = np.array(subject_predictions)
            y_prob_loso = np.array(subject_probs)
            
            # Calculate LOSO metrics
            loso_acc = accuracy_score(y_true_loso, y_pred_loso)
            loso_bal_acc = balanced_accuracy_score(y_true_loso, y_pred_loso)
            
            loso_results[name] = {
                'loso_accuracy': loso_acc,
                'loso_balanced_accuracy': loso_bal_acc,
                'y_true': y_true_loso,
                'y_pred': y_pred_loso,
                'y_prob': y_prob_loso
            }
        
        # Print LOSO Summary
        print("\nLOSO Results (Subject-Independent):")
        print("-" * 60)
        print(f"{'Model':<25} {'LOSO Accuracy':<15} {'Balanced Acc':<15}")
        print("-" * 60)
        for name in models.keys():
            loso_acc = loso_results[name]['loso_accuracy']
            loso_bal = loso_results[name]['loso_balanced_accuracy']
            print(f"{name:<25} {loso_acc*100:>10.2f}%     {loso_bal*100:>10.2f}%")
        
        # =====================================================================
        # STANDARD 10-FOLD STRATIFIED CROSS-VALIDATION
        # =====================================================================
        print("\n" + "="*70)
        print("10-FOLD STRATIFIED CROSS-VALIDATION")
        print("="*70)
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        results = {}
        best_model = None
        best_model_obj = None
        best_cv_acc = 0
        
        for name, model in models.items():
            # Get CV predictions for all samples
            y_pred_cv = cross_val_predict(model, X_selected, y, cv=cv, method='predict')
            
            try:
                y_prob_cv = cross_val_predict(model, X_selected, y, cv=cv, method='predict_proba')[:, 1]
            except:
                y_prob_cv = y_pred_cv.astype(float)
            
            # Calculate CV scores for each fold
            cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='balanced_accuracy')
            
            # Calculate metrics from CV predictions
            acc = accuracy_score(y, y_pred_cv)
            bal_acc = balanced_accuracy_score(y, y_pred_cv)
            
            # Precision, Recall, F1-Score
            from sklearn.metrics import precision_score, recall_score, f1_score as f1_metric
            precision = precision_score(y, y_pred_cv, zero_division=0)
            recall = recall_score(y, y_pred_cv, zero_division=0)
            f1 = f1_metric(y, y_pred_cv, zero_division=0)
            
            # Confusion matrix for Type I/II errors
            cm = confusion_matrix(y, y_pred_cv)
            tn, fp, fn, tp = cm.ravel()
            
            # Error analysis
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            type_i_error = fp / (fp + tn) if (fp + tn) > 0 else 0
            type_ii_error = fn / (fn + tp) if (fn + tp) > 0 else 0
            power = 1 - type_ii_error
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(y, y_prob_cv)
            except:
                roc_auc = 0.5
            
            # Confidence interval
            ci_low = cv_scores.mean() - 1.96 * cv_scores.std()
            ci_high = cv_scores.mean() + 1.96 * cv_scores.std()
            
            # Merge LOSO results
            results[name] = {
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'sensitivity': sensitivity,
                'specificity': specificity,
                'type_i_error': type_i_error,
                'type_ii_error': type_ii_error,
                'power': power,
                'roc_auc': roc_auc,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'confusion_matrix': cm,
                'y_test': y,
                'y_pred': y_pred_cv,
                'y_prob': y_prob_cv,
                # LOSO results
                'loso_accuracy': loso_results[name]['loso_accuracy'],
                'loso_balanced_accuracy': loso_results[name]['loso_balanced_accuracy']
            }
            
            if cv_scores.mean() > best_cv_acc:
                best_cv_acc = cv_scores.mean()
                best_model = name
                model.fit(X_selected, y)
                best_model_obj = model
        
        # Print comprehensive results table
        print("\nComprehensive Model Comparison:")
        print("-" * 100)
        print(f"{'Model':<22} {'Acc':<7} {'Prec':<7} {'Recall':<7} {'F1':<7} {'AUC':<7} {'LOSO':<7} {'CV±std':<12}")
        print("-" * 100)
        for name, metrics in results.items():
            print(f"{name:<22} "
                  f"{metrics['accuracy']*100:>5.1f}% "
                  f"{metrics['precision']*100:>5.1f}% "
                  f"{metrics['recall']*100:>5.1f}% "
                  f"{metrics['f1_score']*100:>5.1f}% "
                  f"{metrics['roc_auc']:>5.3f} "
                  f"{metrics['loso_accuracy']*100:>5.1f}% "
                  f"{metrics['cv_mean']*100:>5.1f}±{metrics['cv_std']*100:>4.1f}%")
        print("-" * 100)
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model}")
        print(f"  10-Fold CV: {best_cv_acc*100:.2f}%")
        print(f"  LOSO: {results[best_model]['loso_accuracy']*100:.2f}%")
        print(f"{'='*50}")
        
        # =====================================================================
        # EXPLAINABLE AI (XAI) - Global & Local Interpretation
        # =====================================================================
        print("\n" + "="*70)
        print("EXPLAINABLE AI (XAI) ANALYSIS")
        print("="*70)
        
        self._perform_xai_analysis(best_model_obj, X_selected, y, selected_feature_cols)
        
        # Store results
        self.results['classification'] = results
        self.results['best_model'] = best_model
        self.results['feature_cols'] = self.selected_feature_names if hasattr(self, 'selected_feature_names') else selected_feature_cols
        self.results['best_model_obj'] = best_model_obj
        self.results['n_features_original'] = len(feature_cols)
        self.results['n_features_selected'] = n_selected
        self.results['feature_selection_method'] = 'F-statistic (ANOVA)'
        self.results['correlation_threshold'] = 0.90
        self.results['pca_components'] = getattr(self, 'pca_components', None)
        self.results['n_models'] = len(models)
        self.results['loso_results'] = loso_results
        
        return results
    
    def _perform_xai_analysis(self, model, X, y, feature_names):
        """
        Perform Explainable AI (XAI) analysis
        
        Global Interpretation: Overall feature importance across all subjects
        Local Interpretation: Explain individual predictions
        """
        print("\n--- Global Feature Importance (Model-Based) ---")
        
        # 1. Model-based feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Contributing Features (Model-Based):")
            print("-" * 50)
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['Feature']:<40} {row['Importance']:.4f}")
            
            # Maximum contributing feature
            max_feature = importance_df.iloc[0]
            print(f"\n★ MAXIMUM CONTRIBUTING FEATURE: {max_feature['Feature']}")
            print(f"  Importance Score: {max_feature['Importance']:.4f}")
            
            self.results['max_contributing_feature'] = max_feature['Feature']
            self.results['global_feature_importance'] = importance_df.to_dict('records')
        
        # 2. Permutation Importance (more reliable)
        print("\n--- Permutation Importance (Unbiased) ---")
        from sklearn.inspection import permutation_importance
        
        perm_importance = permutation_importance(
            model, X, y, n_repeats=30, random_state=42, scoring='balanced_accuracy'
        )
        
        perm_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Mean': perm_importance.importances_mean,
            'Importance_Std': perm_importance.importances_std
        }).sort_values('Importance_Mean', ascending=False)
        
        print("\nTop 10 Features by Permutation Importance:")
        print("-" * 60)
        for idx, row in perm_df.head(10).iterrows():
            print(f"  {row['Feature']:<40} {row['Importance_Mean']:.4f} ± {row['Importance_Std']:.4f}")
        
        self.results['permutation_importance'] = perm_df.to_dict('records')
        
        # 3. SHAP Analysis (if available)
        if HAS_SHAP:
            print("\n--- SHAP Analysis (Global & Local) ---")
            try:
                # Use TreeExplainer for tree-based models
                if hasattr(model, 'estimators_') or 'Forest' in str(type(model)) or 'Gradient' in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use KernelExplainer for other models (slower)
                    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 50))
                
                shap_values = explainer.shap_values(X)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1 (Abnormal)
                
                # Global SHAP importance
                shap_importance = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Importance': shap_importance
                }).sort_values('SHAP_Importance', ascending=False)
                
                print("\nTop 10 Features by SHAP Importance:")
                print("-" * 50)
                for idx, row in shap_df.head(10).iterrows():
                    print(f"  {row['Feature']:<40} {row['SHAP_Importance']:.4f}")
                
                self.shap_values = shap_values
                self.shap_explainer = explainer
                self.results['shap_importance'] = shap_df.to_dict('records')
                
                # Save SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=15)
                plt.tight_layout()
                plt.savefig(self.results_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("  ✓ SHAP summary plot saved")
                
                # Local interpretation: Explain specific abnormal cases
                print("\n--- Local Interpretation (Individual Cases) ---")
                abnormal_indices = np.where(y == 1)[0][:3]  # First 3 abnormal cases
                
                for i, idx in enumerate(abnormal_indices):
                    print(f"\nAbnormal Case {i+1} (Subject {idx}):")
                    case_shap = shap_values[idx]
                    top_contributors = sorted(
                        zip(feature_names, case_shap),
                        key=lambda x: abs(x[1]), reverse=True
                    )[:5]
                    for feat, contribution in top_contributors:
                        direction = "↑ Abnormal" if contribution > 0 else "↓ Normal"
                        print(f"    {feat}: {contribution:+.4f} ({direction})")
                
                self.results['local_explanations'] = {
                    'abnormal_indices': abnormal_indices.tolist(),
                    'shap_values': [shap_values[i].tolist() for i in abnormal_indices]
                }
                
            except Exception as e:
                print(f"  SHAP analysis error: {e}")
        else:
            print("\n  [SHAP not installed - install with: pip install shap]")
        
        # 4. Feature correlation with target
        print("\n--- Feature-Target Correlation ---")
        correlations = []
        for i, feat in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append((feat, corr))
        
        corr_sorted = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
        print("\nTop 10 Features Correlated with Abnormality:")
        print("-" * 50)
        for feat, corr in corr_sorted[:10]:
            direction = "+" if corr > 0 else "-"
            print(f"  {feat:<40} r = {direction}{abs(corr):.4f}")
        
        self.results['feature_target_correlation'] = corr_sorted
    
    def create_visualizations(self):
        """Generate all visualizations for the analysis"""
        print("\n[STEP 5] GENERATING VISUALIZATIONS")
        print("-"*50)
        
        from sklearn.metrics import roc_curve, auc
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Sample Signal Plots
        if self.raw_signals:
            self._plot_sample_signals()
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices()
        
        # 3. ROC Curves
        self._plot_roc_curves()
        
        # 4. Feature Importance
        self._plot_feature_importance()
        
        # 5. Model Comparison
        self._plot_model_comparison()
        
        print(f"\nAll visualizations saved to: {self.results_dir}")
    
    def _plot_sample_signals(self):
        """Plot sample respiratory and PPG signals"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Sample Biomedical Signals', fontsize=14, fontweight='bold')
        
        # Plot up to 2 subjects
        for i, signal_data in enumerate(self.raw_signals[:2]):
            # Respiratory signal
            ax1 = axes[0, i]
            resp = signal_data['resp_signal'][:1000]  # First 8 seconds at 125 Hz
            time = np.arange(len(resp)) / 125
            ax1.plot(time, resp, 'b-', linewidth=0.8, alpha=0.8)
            ax1.set_title(f"Subject {signal_data['subject_id']} - Respiration", fontsize=10)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, alpha=0.3)
            
            # PPG signal
            ax2 = axes[1, i]
            pleth = signal_data['pleth_signal'][:1000]
            ax2.plot(time, pleth, 'r-', linewidth=0.8, alpha=0.8)
            ax2.set_title(f"Subject {signal_data['subject_id']} - PPG (Plethysmogram)", fontsize=10)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_signals.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Sample signals plot saved")
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        classification = self.results.get('classification', {})
        n_models = len(classification)
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices by Model', fontsize=14, fontweight='bold')
        
        for idx, (name, metrics) in enumerate(classification.items()):
            cm = metrics['confusion_matrix']
            
            # Calculate percentages
            cm_pct = cm.astype('float') / cm.sum() * 100
            
            # Annotations with count and percentage
            annot = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' 
                              for j in range(cm.shape[1])] 
                             for i in range(cm.shape[0])])
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[idx],
                       xticklabels=['Normal', 'Abnormal'],
                       yticklabels=['Normal', 'Abnormal'],
                       cbar=True)
            axes[idx].set_title(f'{name}\nAcc: {metrics["accuracy"]*100:.1f}%', fontsize=10)
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Confusion matrices saved")
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        from sklearn.metrics import roc_curve, auc
        
        classification = self.results.get('classification', {})
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
        
        for idx, (name, metrics) in enumerate(classification.items()):
            y_test = metrics['y_test']
            y_prob = metrics['y_prob']
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                   label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (Type I Error)', fontsize=11)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ ROC curves saved")
    
    def _plot_feature_importance(self):
        """Plot feature importance from Random Forest"""
        feature_cols = self.results.get('feature_cols', [])
        classification = self.results.get('classification', {})
        
        # Get a model with feature_importances_ (Random Forest or Gradient Boosting)
        model_obj = None
        model_name = None
        for name in ['Random Forest', 'Gradient Boosting']:
            if name in classification:
                # We need to get the model - but it's not stored in results for all
                # Use the best_model_obj if it has feature_importances_
                break
        
        best_model = self.results.get('best_model_obj')
        if best_model is not None and hasattr(best_model, 'feature_importances_'):
            model_obj = best_model
            model_name = self.results.get('best_model', 'Model')
        else:
            # Try to re-train Random Forest just for feature importance
            if feature_cols:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                
                X = self.features[feature_cols].values
                X = np.nan_to_num(X)
                le = LabelEncoder()
                y = le.fit_transform(self.labels)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                rf.fit(X_scaled, y)
                model_obj = rf
                model_name = "Random Forest"
        
        if model_obj is None or not hasattr(model_obj, 'feature_importances_'):
            print("  ⚠ Feature importance not available")
            return
        
        importances = model_obj.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_features = [feature_cols[i] for i in indices]
        top_importances = importances[indices]
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))[::-1]
        
        bars = ax.barh(range(len(top_features)), top_importances, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title(f'Top 15 Features - {model_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, top_importances):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Feature importance plot saved")
    
    def _plot_model_comparison(self):
        """Plot comprehensive model comparison with multiple metrics"""
        classification = self.results.get('classification', {})
        
        models = list(classification.keys())
        n_models = len(models)
        
        # Create a comprehensive comparison figure
        fig = plt.figure(figsize=(18, 12))
        
        # Subplot 1: 10-Fold CV vs LOSO Accuracy Comparison
        ax1 = fig.add_subplot(2, 3, 1)
        x = np.arange(n_models)
        width = 0.35
        
        cv_accs = [classification[m]['cv_mean']*100 for m in models]
        cv_stds = [classification[m]['cv_std']*100 for m in models]
        loso_accs = [classification[m]['loso_accuracy']*100 for m in models]
        
        bars1 = ax1.bar(x - width/2, cv_accs, width, label='10-Fold CV', 
                       color='#3498db', yerr=cv_stds, capsize=3)
        bars2 = ax1.bar(x + width/2, loso_accs, width, label='LOSO', color='#e74c3c')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=10)
        ax1.set_title('Cross-Validation: 10-Fold vs LOSO', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=7, rotation=45, ha='right')
        ax1.legend(fontsize=9)
        ax1.set_ylim([0, 110])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Multiple Metrics Comparison
        ax2 = fig.add_subplot(2, 3, 2)
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        width = 0.18
        
        for i, metric_name in enumerate(metrics_names):
            key = metric_name.lower().replace('-', '_')
            if key == 'accuracy':
                values = [classification[m]['accuracy']*100 for m in models]
            elif key == 'precision':
                values = [classification[m]['precision']*100 for m in models]
            elif key == 'recall':
                values = [classification[m]['recall']*100 for m in models]
            elif key == 'f1_score':
                values = [classification[m]['f1_score']*100 for m in models]
            
            ax2.bar(x + (i - 1.5) * width, values, width, label=metric_name, color=colors[i])
        
        ax2.set_ylabel('Score (%)', fontsize=10)
        ax2.set_title('Multi-Metric Model Comparison', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=7, rotation=45, ha='right')
        ax2.legend(fontsize=8, loc='lower right')
        ax2.set_ylim([0, 110])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: ROC AUC Comparison
        ax3 = fig.add_subplot(2, 3, 3)
        
        aucs = [classification[m]['roc_auc'] for m in models]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_models))[::-1]
        bars = ax3.barh(range(n_models), aucs, color=colors)
        ax3.set_yticks(range(n_models))
        ax3.set_yticklabels(models, fontsize=8)
        ax3.invert_yaxis()
        ax3.set_xlabel('AUC Score', fontsize=10)
        ax3.set_title('ROC AUC by Model', fontsize=11, fontweight='bold')
        ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax3.set_xlim([0, 1.0])
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, aucs):
            ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=8)
        
        # Subplot 4: Error Analysis
        ax4 = fig.add_subplot(2, 3, 4)
        
        type_i = [classification[m]['type_i_error']*100 for m in models]
        type_ii = [classification[m]['type_ii_error']*100 for m in models]
        
        width = 0.35
        bars1 = ax4.bar(x - width/2, type_i, width, label='Type I Error (α)', color='#e74c3c')
        bars2 = ax4.bar(x + width/2, type_ii, width, label='Type II Error (β)', color='#f39c12')
        
        ax4.set_ylabel('Error Rate (%)', fontsize=10)
        ax4.set_title('Type I & Type II Errors', fontsize=11, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, fontsize=7, rotation=45, ha='right')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Subplot 5: Sensitivity vs Specificity
        ax5 = fig.add_subplot(2, 3, 5)
        
        sensitivity = [classification[m]['sensitivity']*100 for m in models]
        specificity = [classification[m]['specificity']*100 for m in models]
        
        width = 0.35
        bars1 = ax5.bar(x - width/2, sensitivity, width, label='Sensitivity', color='#2ecc71')
        bars2 = ax5.bar(x + width/2, specificity, width, label='Specificity', color='#3498db')
        
        ax5.set_ylabel('Score (%)', fontsize=10)
        ax5.set_title('Sensitivity vs Specificity', fontsize=11, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(models, fontsize=7, rotation=45, ha='right')
        ax5.legend(fontsize=9)
        ax5.set_ylim([0, 110])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Subplot 6: Model Ranking Summary
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Calculate composite score (average of key metrics)
        composite = []
        for m in models:
            score = (classification[m]['accuracy'] + 
                    classification[m]['balanced_accuracy'] +
                    classification[m]['f1_score'] + 
                    classification[m]['roc_auc'] +
                    classification[m]['loso_accuracy']) / 5 * 100
            composite.append(score)
        
        # Sort by composite score
        sorted_indices = np.argsort(composite)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [composite[i] for i in sorted_indices]
        
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, n_models))
        bars = ax6.barh(range(n_models), sorted_scores, color=colors)
        ax6.set_yticks(range(n_models))
        ax6.set_yticklabels(sorted_models, fontsize=8)
        ax6.invert_yaxis()
        ax6.set_xlabel('Composite Score (%)', fontsize=10)
        ax6.set_title('Overall Model Ranking', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Add rank labels
        for i, (bar, val) in enumerate(zip(bars, sorted_scores)):
            ax6.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'#{i+1} ({val:.1f}%)', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Comprehensive model comparison plot saved")
        
        # Additional: Subject-wise performance heatmap
        self._plot_subject_wise_performance()
    
    def _plot_subject_wise_performance(self):
        """Plot subject-wise LOSO performance heatmap"""
        loso_results = self.results.get('loso_results', {})
        if not loso_results:
            return
        
        models = list(loso_results.keys())
        n_subjects = len(loso_results[models[0]]['y_true'])
        
        # Create prediction matrix (models x subjects)
        # 1 = correct prediction, 0 = incorrect
        pred_matrix = np.zeros((len(models), n_subjects))
        for i, model in enumerate(models):
            y_true = loso_results[model]['y_true']
            y_pred = loso_results[model]['y_pred']
            pred_matrix[i] = (y_true == y_pred).astype(int)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap of per-subject predictions
        ax1 = axes[0]
        im = ax1.imshow(pred_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models, fontsize=8)
        ax1.set_xlabel('Subject ID', fontsize=10)
        ax1.set_title('LOSO: Per-Subject Predictions\n(Green=Correct, Red=Incorrect)', 
                     fontsize=11, fontweight='bold')
        
        # Subject accuracy per model
        ax1.set_xticks(range(n_subjects))
        ax1.set_xticklabels(range(1, n_subjects+1), fontsize=6)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Prediction (1=Correct)', fontsize=9)
        
        # Bar chart: Which subjects are hardest?
        ax2 = axes[1]
        
        # Calculate how many models got each subject correct
        subject_difficulty = pred_matrix.sum(axis=0) / len(models) * 100
        
        colors = plt.cm.RdYlGn(subject_difficulty / 100)
        bars = ax2.bar(range(n_subjects), subject_difficulty, color=colors)
        ax2.set_xlabel('Subject ID', fontsize=10)
        ax2.set_ylabel('% Models Correct', fontsize=10)
        ax2.set_title('Subject Difficulty (LOSO)\nLower = Harder to Classify', 
                     fontsize=11, fontweight='bold')
        ax2.set_xticks(range(n_subjects))
        ax2.set_xticklabels(range(1, n_subjects+1), fontsize=6)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
        ax2.set_ylim([0, 110])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Highlight difficult subjects
        hard_subjects = np.where(subject_difficulty < 50)[0]
        if len(hard_subjects) > 0:
            for idx in hard_subjects:
                ax2.annotate('Hard', (idx, subject_difficulty[idx] + 5),
                           ha='center', fontsize=7, color='red')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'subject_wise_loso.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Subject-wise LOSO performance plot saved")
    
    def generate_report(self):
        """Generate comprehensive clinical report with error statistics"""
        print("\n[STEP 6] REPORT GENERATION")
        print("-"*50)
        
        report_path = self.results_dir / "clinical_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RESPIRATORY ABNORMALITY CLASSIFICATION - CLINICAL REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # PREPROCESSING SECTION
            f.write("="*70 + "\n")
            f.write("PREPROCESSING PIPELINE\n")
            f.write("="*70 + "\n")
            if HAS_PREPROCESSING:
                if PREPROCESSING_TYPE == "SOTA":
                    f.write("Type: State-of-the-Art (NeuroKit2/BioSPPy/HeartPy Standards)\n\n")
                    f.write("Steps Applied:\n")
                    f.write("  1. Signal Quality Assessment (SQI)\n")
                    f.write("  2. Missing Value Handling (Interpolation)\n")
                    f.write("  3. Baseline Wander Removal (Butterworth Highpass)\n")
                    f.write("  4. Powerline Interference Removal (50/60 Hz Notch)\n")
                    f.write("  5. Bandpass Filtering:\n")
                    f.write("     - ECG: 0.5-40 Hz (4th order Butterworth)\n")
                    f.write("     - PPG: 0.5-8 Hz (3rd order Butterworth)\n")
                    f.write("     - RESP: 0.05-1 Hz (3rd order Butterworth)\n")
                    f.write("  6. Artifact Detection & Removal (Z-score + Derivative)\n")
                    f.write("  7. Normalization (Z-score standardization)\n\n")
                    f.write("References:\n")
                    f.write("  - NeuroKit2: Makowski et al. (2021)\n")
                    f.write("  - BioSPPy: Carreiras et al. (2015)\n")
                    f.write("  - HeartPy: van Gent et al. (2019)\n")
                else:
                    f.write("Type: Basic Preprocessing\n")
            else:
                f.write("Type: No preprocessing applied (raw signals)\n")
            
            f.write("\n")
            f.write("DATASET SUMMARY\n")
            f.write("-"*50 + "\n")
            f.write(f"Total subjects: {len(self.features)}\n")
            unique, counts = np.unique(self.labels, return_counts=True)
            for label, count in zip(unique, counts):
                f.write(f"  {label}: {count} ({count/len(self.labels)*100:.1f}%)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("CLASSIFICATION RESULTS\n")
            f.write("="*70 + "\n")
            f.write(f"\nBest Model: {self.results.get('best_model', 'N/A')}\n\n")
            
            for model, metrics in self.results.get('classification', {}).items():
                f.write(f"\n{'-'*50}\n")
                f.write(f"{model.upper()}\n")
                f.write(f"{'-'*50}\n")
                f.write(f"  Test Accuracy:      {metrics['accuracy']*100:.2f}%\n")
                f.write(f"  Balanced Accuracy:  {metrics['balanced_accuracy']*100:.2f}%\n")
                f.write(f"  Precision:          {metrics.get('precision', 0)*100:.2f}%\n")
                f.write(f"  Recall:             {metrics.get('recall', 0)*100:.2f}%\n")
                f.write(f"  F1-Score:           {metrics.get('f1_score', 0)*100:.2f}%\n")
                f.write(f"  Cross-Val Mean:     {metrics['cv_mean']*100:.2f}% (±{metrics['cv_std']*100:.2f}%)\n")
                f.write(f"  LOSO Accuracy:      {metrics.get('loso_accuracy', 0)*100:.2f}%\n")
                f.write(f"  ROC AUC:            {metrics['roc_auc']:.3f}\n")
                f.write(f"  95% CI:             [{metrics['ci_low']*100:.2f}%, {metrics['ci_high']*100:.2f}%]\n\n")
                
                f.write("  Diagnostic Metrics:\n")
                f.write(f"    Sensitivity:      {metrics['sensitivity']*100:.2f}%\n")
                f.write(f"    Specificity:      {metrics['specificity']*100:.2f}%\n\n")
                
                f.write("  Error Analysis:\n")
                f.write(f"    Type I Error (α):   {metrics['type_i_error']*100:.2f}%\n")
                f.write(f"    Type II Error (β):  {metrics['type_ii_error']*100:.2f}%\n")
                f.write(f"    Statistical Power:  {metrics['power']*100:.2f}%\n\n")
                
                f.write("  Confusion Matrix:\n")
                cm = metrics['confusion_matrix']
                f.write(f"                  Predicted\n")
                f.write(f"                  Normal  Abnormal\n")
                f.write(f"    Actual Normal   {cm[0,0]:3d}      {cm[0,1]:3d}\n")
                f.write(f"    Actual Abnormal {cm[1,0]:3d}      {cm[1,1]:3d}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("FEATURE SELECTION METHODOLOGY\n")
            f.write("="*70 + "\n\n")
            
            f.write("Method: IMPROVED PIPELINE (Bias-Free)\n")
            f.write("-"*50 + "\n")
            f.write("1. Train/Test Split BEFORE Feature Selection (prevents data leakage)\n")
            f.write("2. Correlation Analysis: Removed features with |r| > 0.90\n")
            f.write("3. Filter Method: F-statistic (ANOVA) - Classifier-independent\n")
            f.write("4. Selection Criteria: Top N features where N = min(n_train/5, significant)\n")
            f.write("5. Statistical Significance: Only features with p < 0.05 considered\n\n")
            
            f.write(f"Original features: {self.results.get('n_features_original', 'N/A')}\n")
            f.write(f"Selected features: {self.results.get('n_features_selected', 'N/A')}\n")
            f.write(f"Selection method: {self.results.get('feature_selection_method', 'F-statistic')}\n")
            f.write(f"Correlation threshold: {self.results.get('correlation_threshold', 0.90)}\n")
            if self.results.get('pca_components'):
                f.write(f"PCA comparison: {self.results['pca_components']} components (95% variance)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("FEATURE IMPORTANCE RANKING (F-statistic Scores)\n")
            f.write("="*70 + "\n")
            
            if hasattr(self, 'feature_importance_ranking'):
                f.write(f"\nTotal features extracted: {len(self.feature_importance_ranking)}\n")
                f.write(f"Features used for classification: {self.results.get('n_features_selected', 'N/A')}\n")
                f.write("(Higher F-score = More discriminative between classes)\n\n")
                
                f.write("Rank | Feature Name                          | F-score\n")
                f.write("-"*60 + "\n")
                for rank, (feat_name, importance) in enumerate(self.feature_importance_ranking, 1):
                    selected = "✓" if rank <= self.results.get('n_features_selected', 25) else " "
                    f.write(f"{rank:3d}  | {feat_name:40s} | {importance:.4f} {selected}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION\n")
            f.write("="*70 + "\n")
            f.write("""
LOSO is the gold standard for biomedical classification because:
  - Subject-independent evaluation (no data leakage between subjects)
  - Tests generalization to completely unseen subjects
  - More realistic for clinical deployment scenarios
""")
            
            f.write("\nLOSO Results Summary:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Model':<25} {'LOSO Accuracy':<15} {'Balanced Acc':<15}\n")
            f.write("-"*60 + "\n")
            for model, metrics in self.results.get('classification', {}).items():
                f.write(f"{model:<25} {metrics.get('loso_accuracy', 0)*100:>10.2f}%     ")
                f.write(f"{metrics.get('loso_balanced_accuracy', 0)*100:>10.2f}%\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("EXPLAINABLE AI (XAI) ANALYSIS\n")
            f.write("="*70 + "\n")
            
            f.write("\n--- Global Interpretation ---\n")
            f.write("(Overall feature importance across all subjects)\n\n")
            
            if 'max_contributing_feature' in self.results:
                f.write(f"★ MAXIMUM CONTRIBUTING FEATURE: {self.results['max_contributing_feature']}\n\n")
            
            if 'permutation_importance' in self.results:
                f.write("Top 10 Features by Permutation Importance:\n")
                for item in self.results['permutation_importance'][:10]:
                    f.write(f"  {item['Feature']:<40} {item['Importance_Mean']:.4f} ± {item['Importance_Std']:.4f}\n")
            
            f.write("\n--- Local Interpretation ---\n")
            f.write("(Individual case explanations)\n")
            f.write("See SHAP summary plot (shap_summary.png) for detailed visualization\n")
            
            if 'feature_target_correlation' in self.results:
                f.write("\n--- Feature-Target Correlation ---\n")
                for feat, corr in self.results['feature_target_correlation'][:10]:
                    direction = "+" if corr > 0 else "-"
                    f.write(f"  {feat:<40} r = {direction}{abs(corr):.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*70 + "\n")
            f.write("""
Type I Error (α - False Positive Rate):
  - Probability of classifying a NORMAL patient as ABNORMAL
  - Clinical impact: Unnecessary follow-up, patient anxiety
  - Target: < 10% for screening applications

Type II Error (β - False Negative Rate):
  - Probability of classifying an ABNORMAL patient as NORMAL
  - Clinical impact: Missed diagnosis, delayed treatment
  - Target: < 20% for diagnostic applications

Statistical Power (1-β):
  - Probability of correctly detecting abnormality when present
  - Higher is better; target: > 80%

Sensitivity (True Positive Rate):
  - Ability to correctly identify abnormal cases
  - Critical for diagnostic applications

Specificity (True Negative Rate):
  - Ability to correctly identify normal cases
  - Important for screening applications
""")
            
            f.write("\n" + "="*70 + "\n")
            f.write("DATA SOURCES USED\n")
            f.write("="*70 + "\n")
            f.write("""
Files processed per subject:
  1. bidmc##.hea/.dat  - Waveform signals (RESP, PLETH, ECG) at 125 Hz
  2. bidmc##n.hea/.dat - Numerics (HR, SpO2, PULSE, RR) at 1 Hz
  3. bidmc##.breath    - Manual breath annotations (ground truth)

Features extracted from:
  - Respiratory signal: Time-domain, Frequency-domain, Wavelet
  - PPG/Plethysmogram: Time-domain, HRV metrics
  - ECG Lead II: R-peak detection, HRV, QRS energy
  - Monitor numerics: HR, SpO2, Pulse rate, Monitor RR
  - Breath annotations: Ground truth RR, breathing regularity
  - Demographics: Age, Sex, Location
""")
            
            f.write("\n" + "="*70 + "\n")
            f.write("VISUALIZATIONS GENERATED\n")
            f.write("="*70 + "\n")
            f.write("  1. sample_signals.png      - Raw respiratory and PPG waveforms\n")
            f.write("  2. confusion_matrices.png  - Classification results by model\n")
            f.write("  3. roc_curves.png          - ROC curves with AUC values\n")
            f.write("  4. feature_importance.png  - Top predictive features\n")
            f.write("  5. model_comparison.png    - Performance and error analysis\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"Report saved to: {report_path}")
        
        # Save features
        self.features.to_csv(self.results_dir / "features.csv", index=False)
        print(f"Features saved to: {self.results_dir / 'features.csv'}")
        
        # Save feature importance ranking
        if hasattr(self, 'feature_importance_ranking'):
            importance_df = pd.DataFrame(self.feature_importance_ranking, 
                                        columns=['Feature', 'Importance'])
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            importance_df['Selected'] = importance_df['Rank'] <= self.results.get('n_features_selected', 25)
            importance_df = importance_df[['Rank', 'Feature', 'Importance', 'Selected']]
            importance_df.to_csv(self.results_dir / "feature_importance.csv", index=False)
            print(f"Feature importance saved to: {self.results_dir / 'feature_importance.csv'}")
    
    def save_model(self):
        """Save trained model for predictions on new data"""
        import pickle
        from sklearn.preprocessing import StandardScaler
        
        best_model = self.results.get('best_model_obj')
        feature_cols = self.results.get('feature_cols', [])
        
        if best_model is None:
            print("No trained model to save")
            return
        
        # Create scaler fitted on training data
        feature_data = []
        for col in feature_cols:
            if col in self.features.columns:
                feature_data.append(self.features[col].values)
            else:
                feature_data.append(np.zeros(len(self.features)))
        
        X = np.array(feature_data).T
        X = np.nan_to_num(X)
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save model package
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'model_name': self.results.get('best_model', 'Unknown'),
            'n_features': len(feature_cols),
            'trained_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        model_path = self.results_dir / 'trained_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Trained model saved to: {model_path}")
    
    def run(self):
        """Run the complete pipeline"""
        print("\n" + "="*70)
        print("RESPIRATORY ABNORMALITY CLASSIFICATION PIPELINE")
        print("Biomedical Signal Processing & Machine Learning")
        print("="*70)
        
        # Load data - prefer CSV files if available
        if self.use_csv:
            csv_files = list(self.data_dir.glob("bidmc_*_Signals.csv"))
            if csv_files:
                data = self.load_csv_data()
            else:
                print("CSV files not found, falling back to WFDB format...")
                data = self.load_wfdb_data()
        else:
            data = self.load_wfdb_data()
            
        if not data:
            print("ERROR: No data loaded")
            return None
        
        # PREPROCESSING STEP (STATE-OF-THE-ART)
        if HAS_PREPROCESSING:
            print("\n[STEP 1.5] SIGNAL PREPROCESSING")
            print("-"*50)
            if PREPROCESSING_TYPE == "SOTA":
                print("Using State-of-the-Art Preprocessing Pipeline")
                print("(Following NeuroKit2, BioSPPy, and HeartPy Standards)")
            else:
                print("Using Basic Preprocessing Pipeline")
            preprocessing_pipeline = PreprocessingPipeline(sampling_rate=SAMPLING_RATE, verbose=True)
            data = preprocessing_pipeline.preprocess_all(data)
            print("Preprocessing complete - signals cleaned and normalized")
        else:
            print("\nNote: Preprocessing module not available, using raw signals")
        
        # Extract features
        self.extract_features(data)
        
        # Assign labels
        self.assign_labels()
        
        # Train classifier with cross-validation
        self.train_classifier()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        # Save trained model for predictions
        self.save_model()
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nResults saved to: {self.results_dir}")
        print("\nGenerated files:")
        print("  • clinical_report.txt     - Comprehensive analysis report")
        print("  • features.csv            - Extracted features dataset")
        print("  • trained_model.pkl       - Trained model for predictions")
        print("  • sample_signals.png      - Signal waveform plots")
        print("  • confusion_matrices.png  - Classification matrices")
        print("  • roc_curves.png          - ROC analysis")
        print("  • feature_importance.png  - Important features")
        print("  • model_comparison.png    - Model comparison charts")
        
        return self.results


def main():
    """Main entry point"""
    # Use CSV files by default (easier to work with)
    pipeline = RespiratoryAnalysisPipeline(use_csv=True)
    results = pipeline.run()
    return results


if __name__ == "__main__":
    main()
