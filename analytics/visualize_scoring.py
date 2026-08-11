"""
Visualize Scoring Results
=========================
Generates publication-quality charts from the batch scoring output
and saves JSON data for the web dashboard.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob

out_dir = '/workspace/error_analysis_results/visuals'
os.makedirs(out_dir, exist_ok=True)

# --- Load scoring results ---
print("Loading scoring results...")
df = pd.read_csv('/workspace/scoring_results.csv')
print(f"Total scored: {len(df)}")

# --- Also load original data for ground truth comparison ---
print("Loading original features for ground truth...")
files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs_orig = [pd.read_parquet(f) for f in files]
df_orig = pd.concat(dfs_orig, ignore_index=True)

# ============================================================
# CHART 1: Probability Distribution (Eligible vs Not Eligible)
# ============================================================
print("\n[1/6] Probability Distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
seuil = 0.5885

ax.hist(df[df['est_eligible'] == 0]['probabilite_eligibilite'], bins=100, alpha=0.7,
        label=f'Non éligibles ({(df["est_eligible"]==0).sum():,})', color='#3B82F6', density=True)
ax.hist(df[df['est_eligible'] == 1]['probabilite_eligibilite'], bins=100, alpha=0.7,
        label=f'Éligibles ({(df["est_eligible"]==1).sum():,})', color='#F97316', density=True)
ax.axvline(x=seuil, color='#EF4444', linestyle='--', linewidth=2, label=f'Seuil optimal = {seuil:.4f}')
ax.set_xlabel('Probabilité d\'éligibilité', fontsize=12)
ax.set_ylabel('Densité', fontsize=12)
ax.set_title('Distribution des Probabilités d\'Éligibilité', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{out_dir}/scoring_probability_distribution.png', dpi=200)
plt.close()
print("  Saved.")

# ============================================================
# CHART 2: Product Recommendation Breakdown (Pie + Bar)
# ============================================================
print("[2/6] Product Recommendation Breakdown...")
eligible = df[df['est_eligible'] == 1]
if 'produit_recommande' in eligible.columns and eligible['produit_recommande'].notnull().any():
    prod_counts = eligible['produit_recommande'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#6366F1', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6']
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(prod_counts.values, labels=prod_counts.index, autopct='%1.1f%%',
                                        colors=colors[:len(prod_counts)], textprops={'fontsize': 10})
    ax1.set_title('Répartition des Produits Recommandés', fontsize=13, fontweight='bold')
    
    # Bar chart
    bars = ax2.barh(prod_counts.index, prod_counts.values, color=colors[:len(prod_counts)])
    ax2.set_xlabel('Nombre de Clients', fontsize=12)
    ax2.set_title('Volume par Produit Recommandé', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, prod_counts.values):
        ax2.text(bar.get_width() + max(prod_counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/scoring_product_breakdown.png', dpi=200)
    plt.close()
    print("  Saved.")
else:
    prod_counts = pd.Series(dtype=int)
    print("  No product recommendations found, skipping.")

# ============================================================
# CHART 3: Confusion Matrix (Predicted vs Actual)
# ============================================================
print("[3/6] Confusion Matrix (Predicted vs Actual)...")
if 'label_eligibilite' in df_orig.columns:
    y_true = df_orig['label_eligibilite'].values
    y_pred = df['est_eligible'].values
    
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax, 
                xticklabels=['Prédit Non-Élig.', 'Prédit Élig.'],
                yticklabels=['Vrai Non-Élig.', 'Vrai Élig.'],
                annot_kws={'size': 16})
    ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valeur Réelle', fontsize=12)
    ax.set_xlabel('Prédiction', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/scoring_confusion_matrix.png', dpi=200)
    plt.close()
    
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"  F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
else:
    f1, prec, rec = 0, 0, 0
    cm = None
    print("  No ground truth available.")

# ============================================================
# CHART 4: Scoring Funnel (Total -> Eligible -> Per Product)
# ============================================================
print("[4/6] Scoring Funnel...")
fig, ax = plt.subplots(figsize=(10, 7))
funnel_labels = ['Population Totale', 'Prédits Éligibles']
funnel_values = [len(df), int((df['est_eligible'] == 1).sum())]

if len(prod_counts) > 0:
    for prod, count in prod_counts.items():
        funnel_labels.append(f'  → {prod}')
        funnel_values.append(int(count))

y_pos = range(len(funnel_labels))
colors_funnel = ['#1E40AF', '#F97316'] + ['#10B981'] * len(prod_counts)
bars = ax.barh(y_pos, funnel_values, color=colors_funnel[:len(funnel_labels)])
ax.set_yticks(y_pos)
ax.set_yticklabels(funnel_labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Nombre de Clients', fontsize=12)
ax.set_title('Entonnoir de Scoring : Population → Éligibles → Produit', fontsize=14, fontweight='bold')

for bar, val in zip(bars, funnel_values):
    ax.text(bar.get_width() + max(funnel_values)*0.01, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{out_dir}/scoring_funnel.png', dpi=200)
plt.close()
print("  Saved.")

# ============================================================
# CHART 5: Calibration Curve (Decile-based)
# ============================================================
print("[5/6] Calibration Curve...")
if 'label_eligibilite' in df_orig.columns:
    df_cal = pd.DataFrame({
        'proba': df['probabilite_eligibilite'],
        'actual': df_orig['label_eligibilite']
    })
    df_cal['decile'] = pd.qcut(df_cal['proba'], 10, duplicates='drop')
    cal_stats = df_cal.groupby('decile').agg(
        mean_proba=('proba', 'mean'),
        actual_rate=('actual', 'mean'),
        count=('actual', 'count')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Calibration parfaite')
    ax.scatter(cal_stats['mean_proba'], cal_stats['actual_rate'], s=cal_stats['count']/500, 
               c='#6366F1', alpha=0.8, label='Déciles du modèle')
    ax.plot(cal_stats['mean_proba'], cal_stats['actual_rate'], color='#6366F1', alpha=0.6)
    ax.set_xlabel('Probabilité Prédite Moyenne', fontsize=12)
    ax.set_ylabel('Taux Réel Observé', fontsize=12)
    ax.set_title('Courbe de Calibration par Décile', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/scoring_calibration.png', dpi=200)
    plt.close()
    print("  Saved.")

# ============================================================
# EXPORT JSON DATA FOR WEB DASHBOARD
# ============================================================
print("[6/6] Exporting JSON for web dashboard...")

dashboard_data = {
    "summary": {
        "total_population": int(len(df)),
        "predicted_eligible": int((df['est_eligible'] == 1).sum()),
        "predicted_not_eligible": int((df['est_eligible'] == 0).sum()),
        "eligibility_rate": float((df['est_eligible'] == 1).mean()),
        "threshold": float(seuil),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec)
    },
    "probability_histogram": {
        "bins": np.histogram(df['probabilite_eligibilite'], bins=50)[1].tolist(),
        "counts_all": np.histogram(df['probabilite_eligibilite'], bins=50)[0].tolist(),
        "counts_eligible": np.histogram(df[df['est_eligible']==1]['probabilite_eligibilite'], bins=50)[0].tolist(),
        "counts_not_eligible": np.histogram(df[df['est_eligible']==0]['probabilite_eligibilite'], bins=50)[0].tolist(),
    },
    "product_breakdown": {k: int(v) for k, v in prod_counts.items()} if len(prod_counts) > 0 else {},
    "funnel": {
        "labels": funnel_labels,
        "values": funnel_values
    }
}

if cm is not None:
    dashboard_data["confusion_matrix"] = {
        "tn": int(cm[0][0]), "fp": int(cm[0][1]),
        "fn": int(cm[1][0]), "tp": int(cm[1][1])
    }

with open(f'{out_dir}/dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print(f"  Saved dashboard JSON to {out_dir}/dashboard_data.json")

print("\n✅ All visualizations generated successfully!")
print(f"   Output directory: {out_dir}")
