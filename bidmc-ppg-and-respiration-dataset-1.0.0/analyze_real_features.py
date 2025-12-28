"""
Analyze Real BIDMC Features to Create Better Synthetic Data
=============================================================
Extract feature statistics from real abnormal patients
to generate synthetic data that will actually be classified correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the features extracted from real patients
features_file = Path('results/features.csv')

if features_file.exists():
    df = pd.read_csv(features_file)
    
    print("\n" + "="*70)
    print("REAL BIDMC FEATURE ANALYSIS")
    print("="*70)
    
    # The 7 selected features (from main.py output)
    selected_features = [
        'resp_zero_crossings',
        'num_monitor_rr_min',
        'resp_low_freq_power',
        'gt_breath_regularity',
        'resp_wavelet_L1_entropy',
        'resp_wavelet_L2_entropy',
        'resp_wavelet_L4_entropy'
    ]
    
    # Split by label
    normal = df[df['label'] == 'Normal']
    abnormal = df[df['label'] == 'Abnormal']
    
    print(f"\nDataset: {len(normal)} Normal, {len(abnormal)} Abnormal")
    
    print("\n" + "="*70)
    print("KEY FEATURE STATISTICS")
    print("="*70)
    
    for feat in selected_features:
        if feat in df.columns:
            print(f"\n{feat}:")
            print(f"  Normal    - Mean: {normal[feat].mean():.4f}, Std: {normal[feat].std():.4f}")
            print(f"  Normal    - Range: [{normal[feat].min():.4f}, {normal[feat].max():.4f}]")
            print(f"  Abnormal  - Mean: {abnormal[feat].mean():.4f}, Std: {abnormal[feat].std():.4f}")
            print(f"  Abnormal  - Range: [{abnormal[feat].min():.4f}, {abnormal[feat].max():.4f}]")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR SYNTHETIC DATA")
    print("="*70)
    
    print("\nTo generate abnormal synthetic patients that will be classified correctly:")
    print("Generate signals that produce these feature values:\n")
    
    for feat in selected_features:
        if feat in df.columns:
            abn_mean = abnormal[feat].mean()
            abn_std = abnormal[feat].std()
            print(f"  {feat:35s}: {abn_mean:.4f} ± {abn_std:.4f}")
    
    # Save feature statistics for generator
    stats = {
        'normal': {},
        'abnormal': {}
    }
    
    for feat in selected_features:
        if feat in df.columns:
            stats['normal'][feat] = {
                'mean': float(normal[feat].mean()),
                'std': float(normal[feat].std()),
                'min': float(normal[feat].min()),
                'max': float(normal[feat].max())
            }
            stats['abnormal'][feat] = {
                'mean': float(abnormal[feat].mean()),
                'std': float(abnormal[feat].std()),
                'min': float(abnormal[feat].min()),
                'max': float(abnormal[feat].max())
            }
    
    import json
    with open('feature_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Feature statistics saved to: feature_statistics.json")
    
else:
    print("ERROR: features.csv not found. Run main.py first.")
