"""
Respiratory Abnormality Prediction Script
==========================================
Use this script to predict respiratory status on new patient data.

Data Files Used:
---------------
- bidmc_##_Signals.csv  → RESP, PLETH, V, AVR, II (125 Hz)
- bidmc_##_Numerics.csv → HR, PULSE, RESP, SpO2 (1 Hz)
- bidmc_##_Breaths.csv  → Breath annotations (ground truth)

Usage:
    1. Single patient prediction:
       python predict.py --patient 01
    
    2. Batch prediction from directory:
       python predict.py --batch /path/to/new/data
    
    3. Interactive mode:
       python predict.py --interactive
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
import argparse

warnings.filterwarnings('ignore')

# WFDB is optional - CSV files are preferred
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
    # Not needed when using CSV files

from scipy import signal, stats
from scipy.fft import fft
import pywt

# Constants
SAMPLING_RATE = 125  # Hz
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / 'results' / 'trained_model.pkl'

# Import State-of-the-Art Preprocessing
try:
    from preprocessing_sota import StateOfTheArtPreprocessor
    HAS_PREPROCESSING = True
except ImportError:
    HAS_PREPROCESSING = False


class FeatureExtractor:
    """Extract features from physiological signals"""
    
    def __init__(self, fs=125):
        self.fs = fs
    
    def extract_time_features(self, sig):
        """Extract time-domain features"""
        sig = np.array(sig)
        sig = sig[~np.isnan(sig)]
        
        if len(sig) < 10:
            return {}
        
        features = {
            'mean': np.mean(sig),
            'std': np.std(sig),
            'var': np.var(sig),
            'min': np.min(sig),
            'max': np.max(sig),
            'range': np.ptp(sig),
            'rms': np.sqrt(np.mean(sig**2)),
            'skewness': stats.skew(sig),
            'kurtosis': stats.kurtosis(sig),
            'p5': np.percentile(sig, 5),
            'p25': np.percentile(sig, 25),
            'p75': np.percentile(sig, 75),
            'p95': np.percentile(sig, 95),
            'iqr': np.percentile(sig, 75) - np.percentile(sig, 25),
            'zero_crossings': np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0),
            'peak_to_peak': np.max(sig) - np.min(sig)
        }
        return features
    
    def extract_freq_features(self, sig):
        """Extract frequency-domain features"""
        sig = np.array(sig)
        sig = sig[~np.isnan(sig)]
        
        if len(sig) < 256:
            return {}
        
        # Power spectral density
        freqs, psd = signal.welch(sig, fs=self.fs, nperseg=min(256, len(sig)))
        
        # Respiratory frequency bands
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        
        total_power = np.sum(psd) + 1e-10
        
        features = {
            'total_power': total_power,
            'resp_power': np.sum(psd[resp_mask]),
            'low_freq_power': np.sum(psd[lf_mask]),
            'high_freq_power': np.sum(psd[hf_mask]),
            'lf_hf_ratio': np.sum(psd[lf_mask]) / (np.sum(psd[hf_mask]) + 1e-10),
            'dominant_freq': freqs[np.argmax(psd)],
            'spectral_entropy': stats.entropy(psd / total_power + 1e-10),
            'spectral_centroid': np.sum(freqs * psd) / total_power
        }
        
        # Respiratory rate estimation
        resp_freqs = freqs[resp_mask]
        resp_psd = psd[resp_mask]
        if len(resp_psd) > 0:
            peak_freq = resp_freqs[np.argmax(resp_psd)]
            features['respiratory_rate'] = peak_freq * 60
        
        return features
    
    def extract_wavelet_features(self, sig, wavelet='db4', level=4):
        """Extract wavelet features"""
        sig = np.array(sig)
        sig = sig[~np.isnan(sig)]
        
        if len(sig) < 2**level:
            return {}
        
        try:
            coeffs = pywt.wavedec(sig, wavelet, level=level)
            
            features = {'wavelet_total_energy': 0}
            for i, c in enumerate(coeffs):
                energy = np.sum(c**2)
                features[f'wavelet_L{i}_energy'] = energy
                features[f'wavelet_L{i}_std'] = np.std(c)
                features[f'wavelet_L{i}_entropy'] = stats.entropy(np.abs(c) + 1e-10)
                features['wavelet_total_energy'] += energy
            
            return features
        except:
            return {}
    
    def extract_hrv_features(self, ppg_signal):
        """Extract HRV features from PPG"""
        ppg = np.array(ppg_signal)
        ppg = ppg[~np.isnan(ppg)]
        
        if len(ppg) < 500:
            return {}
        
        try:
            # Find peaks
            peaks, _ = signal.find_peaks(ppg, distance=int(self.fs * 0.5))
            
            if len(peaks) < 3:
                return {}
            
            # RR intervals
            rr_intervals = np.diff(peaks) / self.fs * 1000  # ms
            
            # Filter unrealistic
            valid = (rr_intervals > 300) & (rr_intervals < 2000)
            rr_intervals = rr_intervals[valid]
            
            if len(rr_intervals) < 2:
                return {}
            
            # Time-domain HRV
            features = {
                'hrv_mean_rr': np.mean(rr_intervals),
                'hrv_sdnn': np.std(rr_intervals),
                'hrv_rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)),
                'hrv_cv': np.std(rr_intervals) / np.mean(rr_intervals) * 100,
                'hrv_pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
                'heart_rate': 60000 / np.mean(rr_intervals)
            }
            return features
        except:
            return {}
    
    def extract_ecg_features(self, ecg_signal):
        """Extract ECG features"""
        ecg = np.array(ecg_signal)
        ecg = ecg[~np.isnan(ecg)]
        
        if len(ecg) < 500:
            return {}
        
        try:
            # Basic stats
            features = {
                'ecg_mean': np.mean(ecg),
                'ecg_std': np.std(ecg),
                'ecg_rms': np.sqrt(np.mean(ecg**2))
            }
            
            # R-peak detection
            peaks, properties = signal.find_peaks(ecg, distance=int(self.fs * 0.5), height=np.mean(ecg))
            
            if len(peaks) >= 2:
                rr_intervals = np.diff(peaks) / self.fs * 1000
                valid = (rr_intervals > 300) & (rr_intervals < 2000)
                rr_intervals = rr_intervals[valid]
                
                if len(rr_intervals) > 0:
                    features['ecg_hr'] = 60000 / np.mean(rr_intervals)
                    features['ecg_rr_std'] = np.std(rr_intervals)
                    features['ecg_sdnn'] = np.std(rr_intervals)
                
                if 'peak_heights' in properties:
                    features['ecg_rpeak_amp_mean'] = np.mean(properties['peak_heights'])
                    features['ecg_rpeak_amp_std'] = np.std(properties['peak_heights'])
            
            # Frequency domain
            freqs, psd = signal.welch(ecg, fs=self.fs, nperseg=min(256, len(ecg)))
            features['ecg_total_power'] = np.sum(psd)
            
            # QRS energy
            qrs_mask = (freqs >= 5) & (freqs <= 25)
            features['ecg_qrs_energy'] = np.sum(psd[qrs_mask])
            
            return features
        except:
            return {}
    
    def extract_numerics_features(self, numerics):
        """Extract features from monitor numerics"""
        features = {}
        
        for name, values in numerics.items():
            if values is None:
                continue
            
            valid = np.array(values)
            valid = valid[~np.isnan(valid)]
            
            if len(valid) < 2:
                continue
            
            features[f'num_{name}_mean'] = np.mean(valid)
            features[f'num_{name}_std'] = np.std(valid)
            features[f'num_{name}_min'] = np.min(valid)
            features[f'num_{name}_max'] = np.max(valid)
            features[f'num_{name}_range'] = np.ptp(valid)
            
            if len(valid) > 2:
                x = np.arange(len(valid))
                slope, _ = np.polyfit(x, valid, 1)
                features[f'num_{name}_trend'] = slope
        
        return features
    
    def calculate_rr_from_breaths(self, breath_annotations, signal_length):
        """Calculate RR from breath annotations"""
        if breath_annotations is None or len(breath_annotations) < 2:
            return {}
        
        breath_intervals = np.diff(breath_annotations) / self.fs
        rr_values = 60 / breath_intervals
        valid_rr = rr_values[(rr_values > 4) & (rr_values < 40)]
        
        if len(valid_rr) > 0:
            return {
                'gt_rr_mean': np.mean(valid_rr),
                'gt_rr_std': np.std(valid_rr),
                'gt_rr_min': np.min(valid_rr),
                'gt_rr_max': np.max(valid_rr),
                'gt_breath_count': len(breath_annotations),
                'gt_breath_regularity': np.std(breath_intervals)
            }
        return {}


class RespiratoryPredictor:
    """Load trained model and predict on new data (CSV-based)"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.extractor = FeatureExtractor(SAMPLING_RATE)
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            print(f"Model not found at {self.model_path}")
            print("Please run main.py first to train the model.")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            print(f"✓ Model loaded: {model_data.get('model_name', 'Unknown')}")
            print(f"  Features: {len(self.feature_cols)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_patient_data_csv(self, data_dir, patient_num):
        """Load patient data from CSV files (preferred method)"""
        data_dir = Path(data_dir)
        
        # Format patient number (e.g., "01", "15", etc.)
        if isinstance(patient_num, int):
            patient_num = f"{patient_num:02d}"
        patient_num = patient_num.replace("bidmc", "").replace("_", "").zfill(2)
        
        subject_id = f"bidmc{patient_num}"
        
        # Load signals CSV
        sig_file = data_dir / f"bidmc_{patient_num}_Signals.csv"
        if not sig_file.exists():
            print(f"Signal file not found: {sig_file}")
            return None
        
        try:
            signals_df = pd.read_csv(sig_file)
            signals_df.columns = [c.strip() for c in signals_df.columns]
            
            data = {
                'subject_id': subject_id,
                'resp_signal': signals_df['RESP'].values if 'RESP' in signals_df.columns else None,
                'pleth_signal': signals_df['PLETH'].values if 'PLETH' in signals_df.columns else None,
                'ecg_v': signals_df['V'].values if 'V' in signals_df.columns else None,
                'ecg_avr': signals_df['AVR'].values if 'AVR' in signals_df.columns else None,
                'ecg_ii': signals_df['II'].values if 'II' in signals_df.columns else None,
            }
            
            # Load numerics CSV
            num_file = data_dir / f"bidmc_{patient_num}_Numerics.csv"
            if num_file.exists():
                numerics_df = pd.read_csv(num_file)
                numerics_df.columns = [c.strip() for c in numerics_df.columns]
                data['numerics'] = {
                    'hr': numerics_df['HR'].values if 'HR' in numerics_df.columns else None,
                    'pulse': numerics_df['PULSE'].values if 'PULSE' in numerics_df.columns else None,
                    'resp': numerics_df['RESP'].values if 'RESP' in numerics_df.columns else None,
                    'spo2': numerics_df['SpO2'].values if 'SpO2' in numerics_df.columns else None,
                }
            else:
                data['numerics'] = {}
            
            # Load breath annotations CSV
            breath_file = data_dir / f"bidmc_{patient_num}_Breaths.csv"
            if breath_file.exists():
                breaths_df = pd.read_csv(breath_file)
                data['breath_annotations'] = breaths_df.iloc[:, 0].dropna().astype(int).values
            else:
                data['breath_annotations'] = None
            
            # Load demographics from Fix.txt
            fix_file = data_dir / f"bidmc_{patient_num}_Fix.txt"
            if fix_file.exists():
                try:
                    with open(fix_file, 'r') as f:
                        content = f.read()
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('Age:'):
                            data['age'] = int(line.split(':')[1].strip())
                        elif line.startswith('Gender:'):
                            data['sex'] = line.split(':')[1].strip()
                        elif line.startswith('Location:'):
                            data['location'] = line.split(':')[1].strip()
                except:
                    pass
            
            return data
            
        except Exception as e:
            print(f"Error loading patient {patient_num}: {e}")
            return None
    
    def load_patient_data(self, data_dir, patient_id):
        """Load patient data - tries CSV first, falls back to WFDB"""
        data_dir = Path(data_dir)
        
        # Extract patient number
        patient_num = patient_id.replace("bidmc", "").replace("_", "").zfill(2)
        
        # Try CSV first (preferred)
        csv_file = data_dir / f"bidmc_{patient_num}_Signals.csv"
        if csv_file.exists():
            return self.load_patient_data_csv(data_dir, patient_num)
        
        # Fall back to WFDB if CSV not available
        if not HAS_WFDB:
            print(f"CSV file not found and wfdb library not available")
            return None
        
        return self.load_patient_data_wfdb(data_dir, patient_id)
    
    def load_patient_data_wfdb(self, data_dir, patient_id):
        """Load patient data from WFDB files (fallback method)"""
        if not HAS_WFDB:
            raise ImportError("wfdb library required for WFDB files")
        
        data_dir = Path(data_dir)
        record_path = data_dir / patient_id
        
        # Load waveforms
        try:
            record = wfdb.rdrecord(str(record_path))
            sig_names = record.sig_name
            
            data = {'subject_id': patient_id}
            
            # Extract signals
            for i, name in enumerate(sig_names):
                if 'RESP' in name.upper():
                    data['resp_signal'] = record.p_signal[:, i]
                elif 'PLETH' in name.upper():
                    data['pleth_signal'] = record.p_signal[:, i]
                elif 'II' in name:
                    data['ecg_ii'] = record.p_signal[:, i]
                elif 'V' in name:
                    data['ecg_v'] = record.p_signal[:, i]
                elif 'AVR' in name:
                    data['ecg_avr'] = record.p_signal[:, i]
            
            # Load numerics if available
            numerics_path = data_dir / f"{patient_id}n"
            if (data_dir / f"{patient_id}n.hea").exists():
                try:
                    num_record = wfdb.rdrecord(str(numerics_path))
                    data['numerics'] = {}
                    for i, name in enumerate(num_record.sig_name):
                        clean_name = name.replace(' ', '_').replace('-', '_').lower()
                        data['numerics'][clean_name] = num_record.p_signal[:, i]
                except:
                    data['numerics'] = {}
            else:
                data['numerics'] = {}
            
            # Load breath annotations if available
            breath_file = data_dir / patient_id
            if (data_dir / f"{patient_id}.breath").exists():
                try:
                    ann = wfdb.rdann(str(breath_file), 'breath')
                    data['breath_annotations'] = ann.sample
                except:
                    data['breath_annotations'] = None
            else:
                data['breath_annotations'] = None
            
            return data
            
        except Exception as e:
            print(f"Error loading patient {patient_id}: {e}")
            return None
    
    def extract_features(self, data):
        """Extract features from patient data"""
        features = {'subject_id': data['subject_id']}
        
        resp_signal = data.get('resp_signal')
        if resp_signal is not None:
            # Time features
            time_feats = self.extractor.extract_time_features(resp_signal)
            features.update({f'resp_{k}': v for k, v in time_feats.items()})
            
            # Frequency features
            freq_feats = self.extractor.extract_freq_features(resp_signal)
            features.update({f'resp_{k}': v for k, v in freq_feats.items()})
            
            # Wavelet features
            wavelet_feats = self.extractor.extract_wavelet_features(resp_signal)
            features.update({f'resp_{k}': v for k, v in wavelet_feats.items()})
        
        # PPG features
        if data.get('pleth_signal') is not None:
            ppg_feats = self.extractor.extract_time_features(data['pleth_signal'])
            features.update({f'ppg_{k}': v for k, v in ppg_feats.items()})
            
            hrv_feats = self.extractor.extract_hrv_features(data['pleth_signal'])
            features.update(hrv_feats)
        
        # ECG features
        if data.get('ecg_ii') is not None:
            ecg_feats = self.extractor.extract_ecg_features(data['ecg_ii'])
            features.update({f'ecg_ii_{k}': v for k, v in ecg_feats.items()})
        
        # Numerics features
        if data.get('numerics'):
            num_feats = self.extractor.extract_numerics_features(data['numerics'])
            features.update(num_feats)
        
        # Breath annotations
        if data.get('breath_annotations') is not None:
            breath_feats = self.extractor.calculate_rr_from_breaths(
                data['breath_annotations'], 
                len(resp_signal) if resp_signal is not None else 0
            )
            features.update(breath_feats)
        
        return features
    
    def preprocess_data(self, data):
        """Apply state-of-the-art preprocessing to signals"""
        if not HAS_PREPROCESSING:
            return data
        
        preprocessor = StateOfTheArtPreprocessor(SAMPLING_RATE)
        preprocessed = data.copy()
        
        # Preprocess respiratory signal
        if 'resp_signal' in data and data['resp_signal'] is not None:
            preprocessed['resp_signal'] = preprocessor.preprocess_respiration(data['resp_signal'])
        
        # Preprocess PPG signal  
        if 'pleth_signal' in data and data['pleth_signal'] is not None:
            preprocessed['pleth_signal'] = preprocessor.preprocess_ppg(data['pleth_signal'])
        
        # Preprocess ECG signals
        for ecg_key in ['ecg_ii', 'ecg_v', 'ecg_avr']:
            if ecg_key in data and data[ecg_key] is not None:
                preprocessed[ecg_key] = preprocessor.preprocess_ecg(data[ecg_key])
        
        return preprocessed
    
    def predict(self, data):
        """Make prediction for a patient (with preprocessing)"""
        if self.model is None:
            raise ValueError("Model not loaded. Run main.py first to train.")
        
        # Apply state-of-the-art preprocessing
        data = self.preprocess_data(data)
        
        # Extract features
        features = self.extract_features(data)
        
        # Prepare feature vector
        feature_vector = []
        for col in self.feature_cols:
            feature_vector.append(features.get(col, 0))
        
        X = np.array([feature_vector])
        X = np.nan_to_num(X)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        # Map to label
        label = 'Abnormal' if prediction == 0 else 'Normal'
        confidence = max(probability) * 100
        
        return {
            'patient_id': data['subject_id'],
            'prediction': label,
            'confidence': confidence,
            'probability_abnormal': probability[0] * 100,
            'probability_normal': probability[1] * 100
        }
    
    def predict_from_file(self, data_dir, patient_id):
        """Load and predict from CSV or WFDB files"""
        data = self.load_patient_data(data_dir, patient_id)
        if data is None:
            return None
        return self.predict(data)
    
    def batch_predict(self, data_dir):
        """Predict on all patients in a directory (CSV-based)"""
        data_dir = Path(data_dir)
        
        # Find all patient records - prefer CSV files
        csv_files = sorted(data_dir.glob("bidmc_*_Signals.csv"))
        
        if csv_files:
            # Use CSV files
            patient_nums = [f.stem.split('_')[1] for f in csv_files]
        else:
            # Fall back to WFDB files
            hea_files = sorted(data_dir.glob("bidmc[0-9][0-9].hea"))
            patient_nums = [f.stem.replace("bidmc", "") for f in hea_files]
        
        if not patient_nums:
            print(f"No patient files found in {data_dir}")
            return []
        
        results = []
        for patient_num in patient_nums:
            patient_id = f"bidmc{patient_num}"
            print(f"Processing {patient_id}...", end=" ")
            
            result = self.predict_from_file(data_dir, patient_id)
            if result:
                results.append(result)
                print(f"→ {result['prediction']} ({result['confidence']:.1f}%)")
            else:
                print("→ Failed")
        
        return results


def save_trained_model(pipeline):
    """Save the trained model for later use"""
    from sklearn.preprocessing import StandardScaler
    
    # Get the best model and feature info
    best_model = pipeline.results.get('best_model_obj')
    feature_cols = pipeline.results.get('feature_cols', [])
    
    if best_model is None:
        print("No trained model available")
        return
    
    # Create and fit scaler on the training data
    feature_data = []
    for col in feature_cols:
        if col in pipeline.features.columns:
            feature_data.append(pipeline.features[col].values)
        else:
            feature_data.append(np.zeros(len(pipeline.features)))
    
    X = np.array(feature_data).T
    X = np.nan_to_num(X)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save model package
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': pipeline.results.get('best_model', 'Unknown'),
        'n_features': len(feature_cols)
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description='Predict respiratory abnormalities')
    parser.add_argument('--patient', type=str, help='Patient ID (e.g., bidmc01)')
    parser.add_argument('--batch', type=str, help='Directory for batch prediction')
    parser.add_argument('--data-dir', type=str, default=str(SCRIPT_DIR), 
                       help='Data directory')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RespiratoryPredictor()
    
    if predictor.model is None:
        print("\n" + "="*60)
        print("MODEL NOT FOUND - Training new model...")
        print("="*60)
        
        # Run main.py to train
        from main import RespiratoryAnalysisPipeline
        pipeline = RespiratoryAnalysisPipeline()
        data = pipeline.load_wfdb_data()
        if data:
            pipeline.extract_features(data)
            pipeline.assign_labels()
            pipeline.train_classifier()
            save_trained_model(pipeline)
            
            # Reload
            predictor = RespiratoryPredictor()
    
    if args.patient:
        # Single patient prediction
        result = predictor.predict_from_file(args.data_dir, args.patient)
        if result:
            print("\n" + "="*50)
            print("PREDICTION RESULT")
            print("="*50)
            print(f"Patient: {result['patient_id']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"P(Abnormal): {result['probability_abnormal']:.1f}%")
            print(f"P(Normal): {result['probability_normal']:.1f}%")
            print("="*50)
    
    elif args.batch:
        # Batch prediction
        results = predictor.batch_predict(args.batch)
        
        if results:
            print("\n" + "="*60)
            print("BATCH PREDICTION SUMMARY")
            print("="*60)
            
            n_abnormal = sum(1 for r in results if r['prediction'] == 'Abnormal')
            n_normal = len(results) - n_abnormal
            
            print(f"Total patients: {len(results)}")
            print(f"Abnormal: {n_abnormal} ({n_abnormal/len(results)*100:.1f}%)")
            print(f"Normal: {n_normal} ({n_normal/len(results)*100:.1f}%)")
            
            # Save to CSV
            df = pd.DataFrame(results)
            output_path = Path(args.batch) / 'predictions.csv'
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
    
    elif args.interactive:
        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION MODE")
        print("="*60)
        print("Enter patient ID (e.g., bidmc01) or 'quit' to exit")
        print("-"*60)
        
        while True:
            patient_id = input("\nPatient ID: ").strip()
            
            if patient_id.lower() in ['quit', 'exit', 'q']:
                break
            
            result = predictor.predict_from_file(args.data_dir, patient_id)
            if result:
                print(f"  → {result['prediction']} (Confidence: {result['confidence']:.1f}%)")
            else:
                print("  → Patient not found or error loading data")
    
    else:
        # Demo: predict on first 5 patients
        print("\n" + "="*60)
        print("DEMO: Predicting on first 5 patients")
        print("="*60)
        
        for i in range(1, 6):
            patient_id = f"bidmc{i:02d}"
            result = predictor.predict_from_file(SCRIPT_DIR, patient_id)
            if result:
                print(f"{patient_id}: {result['prediction']} ({result['confidence']:.1f}%)")


if __name__ == "__main__":
    main()
