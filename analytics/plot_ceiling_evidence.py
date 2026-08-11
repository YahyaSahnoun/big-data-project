import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# On WSL the workspace is the current folder
out_dir = 'error_analysis_results/visuals'
os.makedirs(out_dir, exist_ok=True)

# 1. Load Learning Curve Data
try:
    with open('diagnostics_output/ceiling_diagnostics_report.json', 'r') as f:
        diag_data = json.load(f)
        lc_data = diag_data.get('learning_curve', [])
        fractions = [x['fraction'] * 100 for x in lc_data]
        f1_scores = [x['f1'] for x in lc_data]
except FileNotFoundError:
    fractions = [10, 20, 40, 60, 80, 100]
    f1_scores = [0.202, 0.209, 0.212, 0.213, 0.212, 0.214]

# 2. Hardcode the 20 attempts progression from the report
attempts = [
    ("Baseline (Class Imbalance)", 0.14),
    ("Prétraitement de base", 0.15),
    ("Feature Engineering", 0.16),
    ("Encodage Cible (K-Fold)", 0.221),
    ("Focal Loss", 0.221),
    ("LightGBM (Fast Path)", 0.220),
    ("XGBoost (Tuned)", 0.235),
    ("SMOTE", 0.149),
    ("Modèle par Segment", 0.221),
    ("Ensemble à 4 vues", 0.223)
]
attempt_names = [x[0] for x in attempts]
attempt_scores = [x[1] for x in attempts]

sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(16, 7))

# --- Panel 1: Effort vs F1 (The 20 Attempts) ---
ax1 = plt.subplot(1, 2, 1)
colors = ['#94a3b8' if score < 0.22 else '#3b82f6' for score in attempt_scores]
colors[-1] = '#f59e0b' # highlight the ensemble

y_pos = np.arange(len(attempt_names))
ax1.barh(y_pos, attempt_scores, color=colors)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(attempt_names, fontsize=11)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlim(0, 0.3)
ax1.axvline(x=0.22, color='#ef4444', linestyle='--', alpha=0.7)
ax1.axvline(x=0.24, color='#ef4444', linestyle='--', alpha=0.7)
ax1.axvspan(0.22, 0.24, color='#ef4444', alpha=0.1)

ax1.set_title("Évolution du F1-Score par Modèle (Le 'Plafond')", fontsize=14, fontweight='bold')
ax1.set_xlabel("F1-Score (Classe Positive)", fontsize=12)

# Add text annotation
ax1.text(0.23, len(attempts)-1.5, "Plafond\nIntrinsèque\n(0.22 - 0.24)", 
         color='#b91c1c', fontweight='bold', ha='center', va='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ef4444', boxstyle='round,pad=0.5'))

# --- Panel 2: Volume vs F1 (Learning Curve) ---
ax2 = plt.subplot(1, 2, 2)
ax2.plot(fractions, f1_scores, marker='o', linewidth=3, markersize=8, color='#10b981')
ax2.set_title("Courbe d'Apprentissage (Données vs F1)", fontsize=14, fontweight='bold')
ax2.set_xlabel("% des Données d'Entraînement Utilisées", fontsize=12)
ax2.set_ylabel("F1-Score", fontsize=12)
ax2.set_ylim(0.18, 0.25)
ax2.axhline(y=max(f1_scores), color='#ef4444', linestyle='--', alpha=0.5)

# Add text annotation
ax2.annotate("Stagnation :\nPlus de lignes n'améliore\nplus la performance.",
            xy=(60, f1_scores[3]), xycoords='data',
            xytext=(60, 0.19), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

plt.tight_layout()
out_path = 'error_analysis_results/visuals/ceiling_evidence.png'
plt.savefig(out_path, dpi=200)
print(f"Visual generated at: {out_path}")
