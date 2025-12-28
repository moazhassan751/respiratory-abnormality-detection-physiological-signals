"""
Generate Realistic Mock Respiratory Data for Testing
=====================================================
Creates synthetic patient data that mimics real BIDMC patterns.
Perfect for testing predict.py on "new" unseen patients.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

# Constants
FS = 125  # Sampling rate (Hz)
DURATION = 60  # Duration in seconds
N_SAMPLES = FS * DURATION

def generate_normal_respiration(duration=60, fs=125, rr_mean=15):
    """Generate realistic normal breathing pattern"""
    n_samples = int(duration * fs)
    time = np.linspace(0, duration, n_samples)
    
    # Normal respiratory rate: 12-20 breaths/min
    rr = rr_mean + np.random.uniform(-2, 2)  # breaths per minute
    freq = rr / 60  # Convert to Hz
    
    # Base sinusoidal breathing with slight variations
    resp = np.sin(2 * np.pi * freq * time)
    
    # Add natural variability (respiratory sinus arrhythmia)
    variability = 0.1 * np.sin(2 * np.pi * freq * time * 0.3)
    resp += variability
    
    # Add small random noise (physiological)
    noise = np.random.normal(0, 0.05, n_samples)
    resp += noise
    
    # Add slight baseline wander
    baseline = 0.2 * np.sin(2 * np.pi * 0.05 * time)
    resp += baseline
    
    # Normalize
    resp = (resp - np.mean(resp)) / np.std(resp)
    
    return resp, freq * 60

def generate_abnormal_respiration(duration=60, fs=125, abnormality_type='irregular'):
    """Generate abnormal breathing patterns"""
    n_samples = int(duration * fs)
    time = np.linspace(0, duration, n_samples)
    
    if abnormality_type == 'irregular':
        # Irregular breathing (varying rate and depth)
        resp = np.zeros(n_samples)
        t = 0
        while t < duration:
            # Random breath duration (irregular)
            breath_duration = np.random.uniform(2, 6)  # 2-6 seconds per breath
            breath_samples = int(breath_duration * fs)
            
            if t + breath_duration > duration:
                break
            
            # Variable amplitude (depth variation)
            amplitude = np.random.uniform(0.5, 2.0)
            
            # Single breath
            breath_time = np.linspace(0, breath_duration, breath_samples)
            breath = amplitude * np.sin(2 * np.pi * (1/breath_duration) * breath_time)
            
            start_idx = int(t * fs)
            end_idx = min(start_idx + breath_samples, n_samples)
            resp[start_idx:end_idx] = breath[:end_idx-start_idx]
            
            t += breath_duration
    
    elif abnormality_type == 'rapid':
        # Tachypnea (rapid breathing > 20 breaths/min)
        rr = np.random.uniform(25, 35)
        freq = rr / 60
        resp = np.sin(2 * np.pi * freq * time)
        # Shallow breathing
        resp *= 0.6
    
    elif abnormality_type == 'slow':
        # Bradypnea (slow breathing < 12 breaths/min)
        rr = np.random.uniform(6, 10)
        freq = rr / 60
        resp = np.sin(2 * np.pi * freq * time)
    
    elif abnormality_type == 'apnea':
        # Periodic apnea (pauses in breathing)
        rr = 12
        freq = rr / 60
        resp = np.sin(2 * np.pi * freq * time)
        
        # Add random apnea events (breathing stops)
        n_apneas = np.random.randint(3, 6)
        for _ in range(n_apneas):
            apnea_start = np.random.randint(0, n_samples - fs * 5)
            apnea_duration = np.random.randint(fs * 2, fs * 5)  # 2-5 second pauses
            resp[apnea_start:apnea_start + apnea_duration] = 0
    
    # Add noise
    noise = np.random.normal(0, 0.08, n_samples)
    resp += noise
    
    # Normalize
    resp = (resp - np.mean(resp)) / np.std(resp)
    
    return resp, 0  # Return 0 for abnormal RR

def generate_ppg_signal(duration=60, fs=125, hr_mean=75):
    """Generate realistic PPG (photoplethysmography) signal"""
    n_samples = int(duration * fs)
    time = np.linspace(0, duration, n_samples)
    
    # Heart rate variation
    hr = hr_mean + np.random.uniform(-5, 5)
    freq = hr / 60
    
    # PPG pulse shape (more complex than ECG)
    ppg = np.zeros(n_samples)
    pulse_width = int(0.4 * fs)  # 0.4 second pulse
    
    for i in range(0, n_samples, int(fs / freq)):
        if i + pulse_width < n_samples:
            # Create PPG pulse shape
            pulse_time = np.linspace(0, 1, pulse_width)
            pulse = np.exp(-5 * pulse_time) * np.sin(2 * np.pi * 2 * pulse_time)
            ppg[i:i+pulse_width] += pulse
    
    # Add respiratory modulation
    resp_mod = 0.1 * np.sin(2 * np.pi * 0.25 * time)
    ppg += resp_mod
    
    # Add noise
    noise = np.random.normal(0, 0.05, n_samples)
    ppg += noise
    
    # Normalize
    ppg = (ppg - np.mean(ppg)) / np.std(ppg)
    
    return ppg

def generate_ecg_signal(duration=60, fs=125, hr_mean=75):
    """Generate simplified ECG signal (Lead II)"""
    n_samples = int(duration * fs)
    time = np.linspace(0, duration, n_samples)
    
    hr = hr_mean + np.random.uniform(-5, 5)
    freq = hr / 60
    
    ecg = np.zeros(n_samples)
    
    # Generate R-peaks
    r_peak_interval = int(fs / freq)
    for i in range(10, n_samples - 10, r_peak_interval):
        # R-peak (sharp spike)
        from scipy.signal.windows import gaussian
        peak = gaussian(20, std=2)
        ecg[i-10:i+10] = peak
        ecg[i] = 1.5  # Amplify R-peak
    
    # Add noise
    noise = np.random.normal(0, 0.02, n_samples)
    ecg += noise
    
    # Normalize
    ecg = (ecg - np.mean(ecg)) / np.std(ecg)
    
    return ecg

def generate_numerics(duration=60, hr_mean=75, rr_mean=15, is_normal=True):
    """Generate monitor numerics (1 Hz sampling)"""
    n_samples = duration  # 1 Hz
    
    hr_values = hr_mean + np.random.normal(0, 3, n_samples)
    hr_values = np.clip(hr_values, 50, 120)
    
    if is_normal:
        rr_values = rr_mean + np.random.normal(0, 2, n_samples)
        rr_values = np.clip(rr_values, 12, 20)
        spo2_values = 97 + np.random.normal(0, 1, n_samples)
        spo2_values = np.clip(spo2_values, 95, 100)
    else:
        # Abnormal - more variable
        rr_values = rr_mean + np.random.normal(0, 5, n_samples)
        rr_values = np.clip(rr_values, 5, 40)
        spo2_values = 94 + np.random.normal(0, 2, n_samples)
        spo2_values = np.clip(spo2_values, 88, 98)
    
    pulse_values = hr_values + np.random.normal(0, 2, n_samples)
    
    return {
        'HR': hr_values,
        'PULSE': pulse_values,
        'RESP': rr_values,
        'SpO2': spo2_values
    }

def generate_breath_annotations(resp_signal, fs=125):
    """Generate breath annotation timestamps"""
    # Find peaks (inhalation peaks)
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(resp_signal, distance=int(fs * 1.5), prominence=0.3)
    
    return peaks

def create_synthetic_patient(patient_num, is_normal=True, output_dir='test_data'):
    """Create a complete synthetic patient dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    patient_id = f"test{patient_num:02d}"
    
    print(f"Generating {'NORMAL' if is_normal else 'ABNORMAL'} patient: {patient_id}...")
    
    # Generate signals
    if is_normal:
        resp, actual_rr = generate_normal_respiration(DURATION, FS, rr_mean=15)
        hr_mean = 75
    else:
        # Random abnormality type
        abnormality = np.random.choice(['irregular', 'rapid', 'slow', 'apnea'])
        resp, actual_rr = generate_abnormal_respiration(DURATION, FS, abnormality)
        hr_mean = np.random.uniform(60, 90)
    
    ppg = generate_ppg_signal(DURATION, FS, hr_mean)
    ecg_ii = generate_ecg_signal(DURATION, FS, hr_mean)
    ecg_v = generate_ecg_signal(DURATION, FS, hr_mean) * 0.8  # Different amplitude
    ecg_avr = -ecg_ii * 0.5  # AVR is inverted
    
    # Create Signals CSV
    signals_df = pd.DataFrame({
        'RESP': resp,
        'PLETH': ppg,
        'V': ecg_v,
        'AVR': ecg_avr,
        'II': ecg_ii
    })
    signals_df.to_csv(output_dir / f"bidmc_{patient_num:02d}_Signals.csv", index=False)
    
    # Create Numerics CSV
    numerics = generate_numerics(DURATION, hr_mean, 15 if is_normal else 25, is_normal)
    numerics_df = pd.DataFrame(numerics)
    numerics_df.to_csv(output_dir / f"bidmc_{patient_num:02d}_Numerics.csv", index=False)
    
    # Create Breaths CSV
    breaths = generate_breath_annotations(resp, FS)
    breaths_df = pd.DataFrame({'Sample': breaths})
    breaths_df.to_csv(output_dir / f"bidmc_{patient_num:02d}_Breaths.csv", index=False)
    
    # Create Fix.txt (demographics)
    age = np.random.randint(25, 80)
    sex = np.random.choice(['M', 'F'])
    with open(output_dir / f"bidmc_{patient_num:02d}_Fix.txt", 'w') as f:
        f.write(f"Age: {age}\n")
        f.write(f"Gender: {sex}\n")
        f.write(f"Location: Test\n")
    
    print(f"  ✓ Created {patient_id} ({'Normal' if is_normal else 'Abnormal'})")
    
    return patient_id

def main():
    """Generate a test dataset with normal and abnormal patients"""
    print("\n" + "="*60)
    print("SYNTHETIC TEST DATA GENERATOR")
    print("="*60)
    print("Creating realistic mock patients for testing predict.py\n")
    
    # Create test directory
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)
    
    # Generate 5 normal patients
    print("\nGenerating NORMAL patients (54-58):")
    print("-" * 60)
    for i in range(54, 59):
        create_synthetic_patient(i, is_normal=True, output_dir='test_data')
    
    # Generate 5 abnormal patients
    print("\nGenerating ABNORMAL patients (59-63):")
    print("-" * 60)
    for i in range(59, 64):
        create_synthetic_patient(i, is_normal=False, output_dir='test_data')
    
    print("\n" + "="*60)
    print("✓ TEST DATA GENERATION COMPLETE")
    print("="*60)
    print(f"\nCreated 10 synthetic patients in: {test_dir.absolute()}")
    print("\nTo test predictions, run:")
    print(f"  python predict.py --patient test54")
    print(f"  python predict.py --batch test_data")
    
    # Create summary
    summary_path = test_dir / 'README.txt'
    with open(summary_path, 'w') as f:
        f.write("Synthetic Test Data Summary\n")
        f.write("="*50 + "\n\n")
        f.write("NORMAL patients (Expected: Normal):\n")
        f.write("  test54, test55, test56, test57, test58\n\n")
        f.write("ABNORMAL patients (Expected: Abnormal):\n")
        f.write("  test59, test60, test61, test62, test63\n\n")
        f.write("These are realistic synthetic respiratory signals\n")
        f.write("generated to test the prediction pipeline.\n")
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
