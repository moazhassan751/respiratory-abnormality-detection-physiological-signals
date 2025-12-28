"""
Pipeline Visualization Generator
================================
Creates a one-page visual summary of the complete ML pipeline.

Generates: pipeline_overview.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_visualization():
    """Generate comprehensive pipeline overview figure"""
    
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')
    
    # Title
    ax.text(50, 137, 'Respiratory Abnormality Detection Pipeline', 
            fontsize=20, fontweight='bold', ha='center', va='top')
    ax.text(50, 133, 'BIDMC PPG and Respiration Dataset | 53 ICU Patients', 
            fontsize=12, ha='center', va='top', color='gray')
    
    # Color scheme
    colors = {
        'input': '#3498db',      # Blue
        'preprocess': '#9b59b6', # Purple
        'features': '#27ae60',   # Green
        'selection': '#f39c12',  # Orange
        'models': '#e74c3c',     # Red
        'evaluation': '#1abc9c', # Teal
        'results': '#34495e',    # Dark gray
    }
    
    # =========================================================================
    # SECTION 1: DATA INPUT
    # =========================================================================
    y = 125
    
    # Main box
    box = FancyBboxPatch((5, y-8), 90, 12, boxstyle="round,pad=0.02", 
                         facecolor=colors['input'], alpha=0.2, edgecolor=colors['input'], linewidth=2)
    ax.add_patch(box)
    ax.text(50, y+1, '📊 DATA INPUT', fontsize=14, fontweight='bold', ha='center', color=colors['input'])
    
    # Details
    ax.text(10, y-3, '• 53 ICU patients (BIDMC PhysioNet)\n• Signals: RESP, PLETH (PPG), ECG (V, AVR, II)\n• Numerics: HR, SpO2, RR (1 Hz)\n• Sampling: 125 Hz | Duration: ~8-10 min/subject', 
            fontsize=9, va='top', family='monospace')
    
    # Arrow down
    ax.annotate('', xy=(50, y-10), xytext=(50, y-8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # =========================================================================
    # SECTION 2: PREPROCESSING
    # =========================================================================
    y = 105
    
    box = FancyBboxPatch((5, y-18), 90, 22, boxstyle="round,pad=0.02", 
                         facecolor=colors['preprocess'], alpha=0.2, edgecolor=colors['preprocess'], linewidth=2)
    ax.add_patch(box)
    ax.text(50, y+1, '🔧 SIGNAL PREPROCESSING (State-of-the-Art)', fontsize=14, fontweight='bold', ha='center', color=colors['preprocess'])
    
    # Preprocessing steps
    steps = [
        ('1. SQI Assessment', 'Kurtosis, flatline, clipping detection'),
        ('2. Missing Data', 'Linear/cubic interpolation'),
        ('3. Baseline Removal', 'Highpass filter (fc=0.5 Hz)'),
        ('4. Notch Filter', '50/60 Hz powerline removal'),
        ('5. Bandpass Filter', 'RESP: 0.1-1 Hz, PPG: 0.5-4 Hz'),
        ('6. Motion Artifact', 'MAD outlier detection'),
        ('7. Normalization', 'Z-score standardization'),
    ]
    
    for i, (step, desc) in enumerate(steps):
        col = 10 if i < 4 else 55
        row = y - 5 - (i % 4) * 3.5
        ax.text(col, row, f'{step}:', fontsize=8, fontweight='bold')
        ax.text(col + 20, row, desc, fontsize=8, color='gray')
    
    # Arrow down
    ax.annotate('', xy=(50, y-20), xytext=(50, y-18),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # =========================================================================
    # SECTION 3: FEATURE EXTRACTION
    # =========================================================================
    y = 78
    
    box = FancyBboxPatch((5, y-12), 90, 16, boxstyle="round,pad=0.02", 
                         facecolor=colors['features'], alpha=0.2, edgecolor=colors['features'], linewidth=2)
    ax.add_patch(box)
    ax.text(50, y+1, '📈 FEATURE EXTRACTION (94 Features)', fontsize=14, fontweight='bold', ha='center', color=colors['features'])
    
    # Feature domains
    domains = [
        ('Time Domain (24)', 'Mean, Std, Variance, Range, IQR, Skewness, Kurtosis, RMS, Zero Crossings...'),
        ('Frequency Domain (18)', 'Dominant Freq, Total/LF/HF Power, Spectral Entropy, LF/HF Ratio...'),
        ('Wavelet Domain (20)', 'Energy & Entropy at 5 decomposition levels (db4 wavelet)'),
        ('HRV Features (15)', 'SDNN, RMSSD, pNN50, LF/HF Power, Sample Entropy...'),
        ('ECG/Numerics (17)', 'HR mean/std, SpO2, QRS features, Monitor RR...'),
    ]
    
    for i, (domain, features) in enumerate(domains):
        row = y - 4 - i * 2.5
        ax.text(10, row, f'• {domain}:', fontsize=8, fontweight='bold')
        ax.text(35, row, features[:60] + '...', fontsize=7, color='gray')
    
    # Arrow down
    ax.annotate('', xy=(50, y-14), xytext=(50, y-12),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # =========================================================================
    # SECTION 4: FEATURE SELECTION
    # =========================================================================
    y = 57
    
    box = FancyBboxPatch((5, y-10), 90, 14, boxstyle="round,pad=0.02", 
                         facecolor=colors['selection'], alpha=0.2, edgecolor=colors['selection'], linewidth=2)
    ax.add_patch(box)
    ax.text(50, y+1, '🎯 FEATURE SELECTION (94 → 7 Features)', fontsize=14, fontweight='bold', ha='center', color=colors['selection'])
    
    # Selection pipeline
    ax.text(10, y-3.5, '1. Train/Test Split (80/20, stratified) → NO DATA LEAKAGE', fontsize=8)
    ax.text(10, y-6, '2. Correlation Analysis (remove |r| > 0.90) → 72 features', fontsize=8)
    ax.text(10, y-8.5, '3. F-statistic (ANOVA) ranking on TRAINING DATA ONLY → 7 features (p < 0.05)', fontsize=8)
    
    # Selected features box
    ax.text(60, y-3.5, 'Selected Features:', fontsize=8, fontweight='bold', color=colors['selection'])
    selected = ['resp_zero_crossings (70%)', 'num_monitor_rr_min', 'resp_low_freq_power',
                'resp_wavelet_L1_entropy', 'resp_wavelet_L2_entropy', 'gt_breath_regularity', 'resp_wavelet_L4_entropy']
    for i, feat in enumerate(selected):
        ax.text(60, y-5.5-i*1.2, f'• {feat}', fontsize=6, family='monospace')
    
    # Arrow down
    ax.annotate('', xy=(50, y-12), xytext=(50, y-10),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # =========================================================================
    # SECTION 5: CLASSIFICATION
    # =========================================================================
    y = 38
    
    box = FancyBboxPatch((5, y-10), 90, 14, boxstyle="round,pad=0.02", 
                         facecolor=colors['models'], alpha=0.2, edgecolor=colors['models'], linewidth=2)
    ax.add_patch(box)
    ax.text(50, y+1, '🤖 CLASSIFICATION (12 ML Models)', fontsize=14, fontweight='bold', ha='center', color=colors['models'])
    
    # Models grid
    models = [
        'Logistic Regression', 'Random Forest ⭐', 'Gradient Boosting ⭐', 'SVM (RBF)',
        'SVM (Linear)', 'KNN (k=5)', 'Decision Tree', 'Naive Bayes',
        'AdaBoost', 'Extra Trees', 'LDA', 'Voting Ensemble'
    ]
    
    for i, model in enumerate(models):
        col = 10 + (i % 4) * 23
        row = y - 4 - (i // 4) * 2.5
        ax.text(col, row, f'• {model}', fontsize=7)
    
    # Validation
    ax.text(10, y-10, '✓ LOSO Cross-Validation (53 folds) + 10-Fold Stratified CV', 
            fontsize=8, fontweight='bold', color=colors['models'])
    
    # Arrow down
    ax.annotate('', xy=(50, y-12), xytext=(50, y-10),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # =========================================================================
    # SECTION 6: EVALUATION
    # =========================================================================
    y = 19
    
    box = FancyBboxPatch((5, y-10), 90, 14, boxstyle="round,pad=0.02", 
                         facecolor=colors['evaluation'], alpha=0.2, edgecolor=colors['evaluation'], linewidth=2)
    ax.add_patch(box)
    ax.text(50, y+1, '📊 EVALUATION & EXPLAINABILITY', fontsize=14, fontweight='bold', ha='center', color=colors['evaluation'])
    
    # Metrics
    ax.text(10, y-3.5, 'Metrics:', fontsize=8, fontweight='bold')
    ax.text(22, y-3.5, 'Accuracy, Precision, Recall, F1, AUC, Sensitivity, Specificity, Type I/II Error, Power, 95% CI', fontsize=7)
    
    ax.text(10, y-6, 'XAI Methods:', fontsize=8, fontweight='bold')
    ax.text(28, y-6, 'SHAP (TreeExplainer), Permutation Importance, Feature-Target Correlation, LIME', fontsize=7)
    
    ax.text(10, y-8.5, 'Visualizations:', fontsize=8, fontweight='bold')
    ax.text(32, y-8.5, 'Confusion Matrix, ROC Curves, Feature Importance, Subject-wise Heatmap, SHAP Summary', fontsize=7)
    
    # Arrow down
    ax.annotate('', xy=(50, y-12), xytext=(50, y-10),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # =========================================================================
    # SECTION 7: RESULTS
    # =========================================================================
    y = 2
    
    box = FancyBboxPatch((5, y-3), 90, 8, boxstyle="round,pad=0.02", 
                         facecolor=colors['results'], alpha=0.3, edgecolor=colors['results'], linewidth=3)
    ax.add_patch(box)
    ax.text(50, y+2.5, '🏆 FINAL RESULTS', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Results summary
    results_text = (
        'LOSO Accuracy: 88.7% (Gradient Boosting) | 94.3% (Random Forest)\n'
        '10-Fold CV: 94.2% | Precision: 96.2% | Recall: 92.6% | F1: 94.3% | AUC: 0.962\n'
        'Max Contributing Feature: resp_zero_crossings (70% importance)'
    )
    ax.text(50, y-0.5, results_text, fontsize=9, ha='center', va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig('results/figures/pipeline_overview.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('pipeline_overview.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✓ Pipeline visualization saved to:")
    print("  • results/figures/pipeline_overview.png")
    print("  • pipeline_overview.png")

if __name__ == '__main__':
    create_pipeline_visualization()
