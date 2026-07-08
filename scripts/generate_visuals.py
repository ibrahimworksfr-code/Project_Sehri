import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Create images directory using a strict Absolute Path
# Dynamically locate the root directory regardless of where the repo is cloned
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
output_dir = os.path.join(PROJECT_ROOT, "docs", "images")
os.makedirs(output_dir, exist_ok=True)

# 1. Aesthetic Constraints (White Paper / Big 4 Style)
sns.set_theme(style="whitegrid")
corporate_palette = ["#708090", "#000080", "#CC6666", "#4682B4"]
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif', 'figure.dpi': 300})

# --- Visual 1: The Temporal Inversion (Grouped Bar Chart) ---
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Afternoon', 'Evening (Iftar)']
baseline = [40, 58]  # percentages
ramadan = [40, 80]   # percentages

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#708090')
rects2 = ax.bar(x + width/2, ramadan, width, label='Ramadan', color='#000080')

ax.set_ylabel('Order Volume (%)', fontsize=12, weight='bold')
ax.set_title('The Temporal Inversion: Baseline vs. Ramadan Order Volume', fontsize=14, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend()
ax.set_ylim(0, 100)

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'VISUAL_1_Temporal_Inversion.png'))
plt.close()

# --- Visual 2: The Digital Twin Upscaling (SMOTEN) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

classes = ['Negative', 'Positive']
pre_smote = [123, 7]
post_smote = [5000, 5000]

ax1.bar(classes, pre_smote, color=['#708090', '#CC6666'])
ax1.set_title('Pre-SMOTEN (Original N=130)', fontsize=14, weight='bold')
ax1.set_ylabel('Number of Samples', weight='bold')
ax1.set_ylim(0, 5500)
for i, v in enumerate(pre_smote):
    ax1.text(i, v + 150, str(v), ha='center', fontsize=12, weight='bold')

ax2.bar(classes, post_smote, color=['#708090', '#4682B4'])
ax2.set_title('Post-SMOTEN (Digital Twin N=10,000)', fontsize=14, weight='bold')
ax2.set_ylim(0, 5500)
for i, v in enumerate(post_smote):
    ax2.text(i, v + 150, str(v), ha='center', fontsize=12, weight='bold')

plt.suptitle('Synthetic Data Generation: Resolving Extreme Class Imbalance', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'VISUAL_2_SMOTEN_Upscaling.png'))
plt.close()

# --- Visual 3: Market Segmentation Archetypes (K-Modes) ---
fig, ax = plt.subplots(figsize=(10, 6))
segments = ['Family Cart Maximizer', 'Zepto Household Routine', 'Solo Scroller']
counts = [59, 43, 28]

y_pos = np.arange(len(segments))
ax.barh(y_pos, counts, align='center', color='#000080')
ax.set_yticks(y_pos)
ax.set_yticklabels(segments, fontsize=12, weight='bold')
ax.invert_yaxis()  
ax.set_xlabel('Number of Users (N=130)', fontsize=12, weight='bold')
ax.set_title('K-Modes Market Segmentation', fontsize=14, weight='bold', pad=20)
ax.text(0.5, -0.15, '* All extracted clusters explicitly share a Price Inelasticity Score of 0', 
        ha='center', va='center', transform=ax.transAxes, fontsize=11, fontstyle='italic', color='#CC6666')

for i, v in enumerate(counts):
    ax.text(v + 1, i, str(v), va='center', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'VISUAL_3_KModes_Segmentation.png'))
plt.close()

# --- Visual 4: TRUE Business Drivers (Pure L2 Penalized Logit) ---
features = [
    'is_gatekeeper', 'macro_zone_Old City & South', 'macro_zone_Unknown',
    'macro_zone_Periphery', 'macro_zone_Malakpet/Saidabad', 'is_ramadan',
    'macro_zone_West Zone/IT Corridor', 'macro_zone_Tolichowki Corridor',
    'Swiggy Loyalist', 'Other App / Unknown',
    'Blinkit Loyalist', 'Zepto Loyalist'
]
coefficients = [
    5.948166, 5.797463, 0.890007,
    -0.258802, -0.465969, -0.834916,
    -1.047683, -1.160126,
    -5.462120, -6.324176,
    -6.763787, -6.928364
]

df_coef = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
df_coef = df_coef.sort_values('Coefficient')

colors = ['#CC6666' if c < 0 else '#4682B4' for c in df_coef['Coefficient']]

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(df_coef['Feature'], df_coef['Coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
ax.set_title('Pure Predictive Drivers: High-Value Night Intent (No Leakage)', fontsize=14, weight='bold', pad=20)
ax.set_xlabel('Impact Weight (Coefficient)', fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'VISUAL_4_Business_Drivers.png'))
plt.close()

# --- Visual 5: TRUE Confusion Matrix (Pure Model) ---
# Based on the pure model classification report yielding N=2000 test set
cm = np.array([[940, 60], [0, 1000]])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'],
            annot_kws={"size": 16, "weight": "bold"})
ax.set_title('Pure Model Confusion Matrix (Unseen Test Set: N=2000)', fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'VISUAL_5_Confusion_Matrix.png'))
plt.close()

print("All enterprise visualizations compiled with pure model data and saved to docs/images/.")
