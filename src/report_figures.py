"""Generate report figures from main pipeline results.
Run this after src/train_xgb_cv.py to create all report charts.

Usage:
    python -m src.report_figures
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
FIGURE_DIR = PROJECT_ROOT / 'report_figures'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Report Figures Generator")
print("=" * 60)
print(f"Reading results from: {LOG_DIR}")
print(f"Saving figures to: {FIGURE_DIR}")

# ============================================================
# Load results
# ============================================================

summary_path = LOG_DIR / 'summary.json'
feat_imp_path = LOG_DIR / 'feature_importance.csv'
oof_path = LOG_DIR / 'oof_predictions.csv'

missing = []
if not summary_path.exists():
    missing.append(str(summary_path))
if not feat_imp_path.exists():
    missing.append(str(feat_imp_path))
if not oof_path.exists():
    missing.append(str(oof_path))

if missing:
    print("\n❌ Missing files:")
    for f in missing:
        print(f"   - {f}")
    print("\nPlease run first: python -m src.train_xgb_cv")
    exit(1)

print("\n✓ Loading results...")
with open(summary_path, 'r', encoding='utf-8') as f:
    summary = json.load(f)

feature_importance = pd.read_csv(feat_imp_path)
oof_df = pd.read_csv(oof_path)

fold_aucs = summary.get('fold_aucs', summary.get('fold_scores', []))
mean_auc = summary.get('mean_fold_auc', np.mean(fold_aucs))
std_auc = summary.get('std_fold_auc', np.std(fold_aucs))

y_true = oof_df['y_true'].values
y_pred = oof_df['y_pred'].values

print(f"✓ Fold AUCs: {fold_aucs}")
print(f"✓ Mean AUC: {mean_auc:.6f}")
print(f"✓ Std AUC: {std_auc:.6f}")
print(f"✓ OOF AUC: {summary.get('oof_auc', 0):.6f}")

# ============================================================
# Figure settings
# ============================================================

plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================
# Figure 1: Cross-Validation Fold AUCs
# ============================================================
print("\n[1/7] Generating CV fold AUC chart...")

fig, ax = plt.subplots(figsize=(10, 6))

folds = range(1, len(fold_aucs) + 1)
bars = ax.bar(folds, fold_aucs, color='steelblue', alpha=0.8, edgecolor='black')
ax.axhline(y=mean_auc, color='red', linestyle='--', linewidth=2, 
           label=f'Mean AUC = {mean_auc:.4f}')
ax.fill_between([0.5, len(fold_aucs) + 0.5], 
                [mean_auc - std_auc, mean_auc - std_auc], 
                [mean_auc + std_auc, mean_auc + std_auc], 
                alpha=0.2, color='red', label=f'±1 Std ({std_auc:.4f})')

for bar, auc_val in zip(bars, fold_aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0003,
            f'{auc_val:.4f}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('5-Fold Cross Validation AUC Scores', fontsize=14)
ax.set_xticks(folds)
ax.set_ylim(min(fold_aucs) - 0.002, max(fold_aucs) + 0.002)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / '01_cv_fold_aucs.png')
plt.savefig(FIGURE_DIR / '01_cv_fold_aucs.pdf')
print(f"   Saved: {FIGURE_DIR / '01_cv_fold_aucs.png'}")

# ============================================================
# Figure 2: Feature Importance
# ============================================================
print("\n[2/7] Generating feature importance chart...")

fig, ax = plt.subplots(figsize=(10, 8))

top_n = 15
top_features = feature_importance.head(top_n)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors[::-1])

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

for i, row in top_features.iterrows():
    ax.text(row['importance'] + 0.002, i, f'{row["importance"]:.3f}', 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURE_DIR / '02_feature_importance.png')
plt.savefig(FIGURE_DIR / '02_feature_importance.pdf')
print(f"   Saved: {FIGURE_DIR / '02_feature_importance.png'}")

# ============================================================
# Figure 3: Feature Importance Pie Chart
# ============================================================
print("\n[3/7] Generating feature importance pie chart...")

fig, ax = plt.subplots(figsize=(10, 8))

top_n_pie = 5
top_pie = feature_importance.head(top_n_pie)
others_sum = feature_importance.iloc[top_n_pie:]['importance'].sum()

pie_data = list(top_pie['importance']) + [others_sum]
pie_labels = list(top_pie['feature']) + ['Others']
colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))

ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
       colors=colors, startangle=90, textprops={'fontsize': 11})
ax.set_title(f'Feature Importance Distribution (Top {top_n_pie})', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURE_DIR / '03_feature_importance_pie.png')
plt.savefig(FIGURE_DIR / '03_feature_importance_pie.pdf')
print(f"   Saved: {FIGURE_DIR / '03_feature_importance_pie.png'}")

# ============================================================
# Figure 4: ROC Curve
# ============================================================
print("\n[4/7] Generating ROC curve...")

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Out-of-Fold Predictions', fontsize=14)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / '04_roc_curve.png')
plt.savefig(FIGURE_DIR / '04_roc_curve.pdf')
print(f"   Saved: {FIGURE_DIR / '04_roc_curve.png'}")

# ============================================================
# Figure 5: Prediction Distribution
# ============================================================
print("\n[5/7] Generating prediction distribution chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
for label, color, name in [(0, 'skyblue', 'No Diabetes'), (1, 'salmon', 'Diabetes')]:
    mask = y_true == label
    ax1.hist(y_pred[mask], bins=50, alpha=0.6, color=color, label=name, density=True)

ax1.set_xlabel('Predicted Probability', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Prediction Distribution by True Class', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.hist(y_pred, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Predicted Probability', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Overall Prediction Distribution', fontsize=14)
ax2.axvline(x=y_pred.mean(), color='red', linestyle='--', 
            label=f'Mean = {y_pred.mean():.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / '05_prediction_distribution.png')
plt.savefig(FIGURE_DIR / '05_prediction_distribution.pdf')
print(f"   Saved: {FIGURE_DIR / '05_prediction_distribution.png'}")

# ============================================================
# Figure 6: AUC Boxplot
# ============================================================
print("\n[6/7] Generating AUC boxplot...")

fig, ax = plt.subplots(figsize=(8, 6))

box_data = [fold_aucs]
bp = ax.boxplot(box_data, patch_artist=True, labels=['5-Fold CV AUC'])
bp['boxes'][0].set_facecolor('lightblue')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)

np.random.seed(42)
ax.scatter(np.random.normal(1, 0.04, len(fold_aucs)), fold_aucs, 
           alpha=0.6, color='darkblue', s=50)

ax.set_ylabel('AUC', fontsize=12)
ax.set_title('Distribution of Fold AUC Scores', fontsize=14)
ax.set_ylim(min(fold_aucs) - 0.002, max(fold_aucs) + 0.002)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURE_DIR / '06_auc_boxplot.png')
plt.savefig(FIGURE_DIR / '06_auc_boxplot.pdf')
print(f"   Saved: {FIGURE_DIR / '06_auc_boxplot.png'}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Report Figures Generated Successfully!")
print("=" * 60)
print(f"\nFigures saved to: {FIGURE_DIR}")
print("\nGenerated files:")
for f in sorted(FIGURE_DIR.glob('*.png')):
    print(f"  - {f.name}")

print("\n✅ Done! You can now use these images in your report.")
print("\n📌 Note: Experiment progress and learning curves are not included")
print("   because they require data not saved by the main pipeline.")
