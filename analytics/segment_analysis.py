import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

out_dir = '/workspace/error_analysis_results/visuals'
os.makedirs(out_dir, exist_ok=True)

print("Loading data...")
files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Calculate segment statistics
segment_col = 'CUSTOMER_RATING'
df[segment_col] = df[segment_col].fillna('MISSING').astype(str)

stats = df.groupby(segment_col).agg(
    Total_Customers=('label_eligibilite', 'count'),
    Class_1_Eligible=('label_eligibilite', 'sum')
).reset_index()

stats['Class_0_Not_Eligible'] = stats['Total_Customers'] - stats['Class_1_Eligible']
stats['Positive_Rate'] = stats['Class_1_Eligible'] / stats['Total_Customers']
stats = stats.sort_values('Total_Customers', ascending=False)

print("\n--- CUSTOMER RATING SEGMENT BREAKDOWN ---")
print(stats.to_string(index=False))

# Create Visualizations
plt.figure(figsize=(14, 10))

# Subplot 1: Stacked Bar Chart (Log Scale) to show absolute volumes
plt.subplot(2, 1, 1)
sns.set_theme(style="whitegrid")
bars1 = plt.bar(stats[segment_col], stats['Class_0_Not_Eligible'], label='Not Eligible (Class 0)', color='#1f77b4')
bars2 = plt.bar(stats[segment_col], stats['Class_1_Eligible'], bottom=stats['Class_0_Not_Eligible'], label='Eligible (Class 1)', color='#ff7f0e')
plt.yscale('log')
plt.ylabel('Number of Customers (Log Scale)')
plt.title('Customer Volume by Rating (Class 0 entirely dwarfs Class 1)')
plt.legend()
plt.xticks(rotation=45)

# Subplot 2: Positive Rate Line Chart
plt.subplot(2, 1, 2)
ax = sns.barplot(x=segment_col, y='Total_Customers', data=stats, color='lightgray', alpha=0.6, label='Total Volume')
plt.ylabel('Total Customers')
plt.xticks(rotation=45)

ax2 = ax.twinx()
sns.lineplot(x=segment_col, y='Positive_Rate', data=stats, color='red', marker='o', linewidth=2, ax=ax2, label='Positive Rate')
ax2.set_ylabel('Positive Rate (Eligibility %)')
ax2.set_ylim(0, max(stats['Positive_Rate']) * 1.2)
ax2.grid(False)

# Formatting legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Positive Rate vs Total Volume by Segment')
plt.tight_layout()

plt.savefig(f'{out_dir}/segment_breakdown.png', dpi=300)
print(f"\nSaved visualization to {out_dir}/segment_breakdown.png")
