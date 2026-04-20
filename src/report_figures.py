"""Generate final report figures from the selected 7-fold workflow.

Run this after ``src.train_xgb_cv`` to create the report-ready charts.

Usage:
    python -m src.report_figures
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
print("Report Figures Generator (Selected 7-Fold Workflow)")
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
oof_df = pd.read_csv(
    oof_path,
    usecols=['y_true', 'y_pred'],
    dtype={'y_true': 'uint8', 'y_pred': 'float32'},
)

fold_aucs = summary.get('fold_aucs', summary.get('fold_scores', []))
mean_auc = summary.get('mean_fold_auc', np.mean(fold_aucs))
std_auc = summary.get('std_fold_auc', np.std(fold_aucs))
fold_count = summary.get('n_splits', summary.get('N_splits'))
if fold_count is None:
    if fold_aucs:
        fold_count = len(fold_aucs)
    else:
        raise KeyError("summary.json must contain fold information via 'n_splits', 'N_splits', or 'fold_aucs'")

y_true = oof_df['y_true'].values
y_pred = oof_df['y_pred'].values

print(f"✓ Fold AUCs: {fold_aucs}")
print(f"✓ Mean AUC: {mean_auc:.6f}")
print(f"✓ Std AUC: {std_auc:.6f}")
print(f"✓ OOF AUC: {summary.get('oof_auc', 0):.6f}")
print(f"✓ Number of folds: {fold_count}")

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
# Figure 1: 7-Fold Cross-Validation AUCs
# ============================================================
print("\n[1/6] Generating 7-fold CV AUC chart...")

fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(10, 7),
    gridspec_kw={'height_ratios': [5, 1]},
)

fold_ids = np.arange(1, len(fold_aucs) + 1)
for ax in (ax_top, ax_bottom):
    bars = ax.bar(fold_ids, fold_aucs, color='#4c78a8', alpha=0.85, edgecolor='black')
    ax.axhline(y=mean_auc, color='#d62728', linestyle='--', linewidth=2,
               label=f'Mean AUC = {mean_auc:.6f}')
    ax.grid(True, alpha=0.3, axis='y')

for bar, auc_val in zip(bars, fold_aucs):
    ax_top.text(
        bar.get_x() + bar.get_width() / 2.0,
        auc_val + 0.00003,
        f'{auc_val:.4f}',
        ha='center',
        va='bottom',
        fontsize=10,
    )

upper_margin = 0.00015
lower_zoom = min(fold_aucs) - 0.0001
upper_zoom = max(fold_aucs) + upper_margin
ax_top.set_ylim(lower_zoom, upper_zoom)
ax_bottom.set_ylim(0.0, 0.08)

ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

kwargs = dict(
    marker=[(-1, -1), (1, 1)],
    markersize=12,
    linestyle='none',
    color='k',
    mec='k',
    mew=1,
    clip_on=False,
)
ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

ax_top.set_ylabel('Validation AUC', fontsize=12)
ax_bottom.set_xlabel('Fold', fontsize=12)
ax_top.set_title(f'{fold_count}-Fold Cross-Validation AUC by Fold (Broken Axis)', fontsize=14)
ax_bottom.set_xticks(fold_ids)
ax_top.legend(loc='upper left')

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'figure_1_7fold_cv_aucs.png')
plt.savefig(FIGURE_DIR / 'figure_1_7fold_cv_aucs.pdf')
plt.close(fig)
print(f"   Saved: {FIGURE_DIR / 'figure_1_7fold_cv_aucs.png'}")

# ============================================================
# Figure 2: Feature Importance
# ============================================================
print("\n[2/6] Generating feature importance chart...")

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
plt.savefig(FIGURE_DIR / 'figure_2_feature_importance.png')
plt.savefig(FIGURE_DIR / 'figure_2_feature_importance.pdf')
plt.close(fig)
print(f"   Saved: {FIGURE_DIR / 'figure_2_feature_importance.png'}")

# ============================================================
# Figure 3: Feature Importance Pie Chart
# ============================================================
print("\n[3/6] Generating feature importance pie chart...")

fig, ax = plt.subplots(figsize=(10, 8))

top_n_pie = 5
top_pie = feature_importance.head(top_n_pie)
colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2']

ax.pie(
    top_pie['importance'],
    labels=top_pie['feature'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1},
)
ax.set_title(f'Top {top_n_pie} Feature Importance Share', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'figure_3_feature_importance_pie.png')
plt.savefig(FIGURE_DIR / 'figure_3_feature_importance_pie.pdf')
plt.close(fig)
print(f"   Saved: {FIGURE_DIR / 'figure_3_feature_importance_pie.png'}")

# ============================================================
# Figure 4: ROC Curve
# ============================================================
print("\n[4/6] Generating ROC curve...")

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
plt.savefig(FIGURE_DIR / 'figure_4_roc_curve.png')
plt.savefig(FIGURE_DIR / 'figure_4_roc_curve.pdf')
plt.close(fig)
print(f"   Saved: {FIGURE_DIR / 'figure_4_roc_curve.png'}")

# ============================================================
# Figure 5: Prediction Distribution
# ============================================================
print("\n[5/6] Generating prediction distribution chart...")

fig, ax = plt.subplots(figsize=(11, 6))
negative_scores = y_pred[y_true == 0]
positive_scores = y_pred[y_true == 1]

bin_edges = np.linspace(0.0, 1.0, 51)
neg_counts, _ = np.histogram(negative_scores, bins=bin_edges)
pos_counts, _ = np.histogram(positive_scores, bins=bin_edges)
neg_prop = neg_counts / neg_counts.sum()
pos_prop = pos_counts / pos_counts.sum()
bin_widths = np.diff(bin_edges)

ax.bar(
    bin_edges[:-1],
    neg_prop,
    width=bin_widths,
    align='edge',
    alpha=0.55,
    label='True class 0',
    color='#4c78a8',
    edgecolor='white',
    linewidth=0.3,
)
ax.bar(
    bin_edges[:-1],
    pos_prop,
    width=bin_widths,
    align='edge',
    alpha=0.55,
    label='True class 1',
    color='#e45756',
    edgecolor='white',
    linewidth=0.3,
)
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Proportion within Class', fontsize=12)
ax.set_title('Prediction Distribution by True Class (OOF)', fontsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'figure_5_prediction_distribution.png')
plt.savefig(FIGURE_DIR / 'figure_5_prediction_distribution.pdf')
plt.close(fig)
print(f"   Saved: {FIGURE_DIR / 'figure_5_prediction_distribution.png'}")

# ============================================================
# Figure 6: AUC Dot Plot (Alternative - No Boxplot)
# ============================================================
print("\n[6/6] Generating AUC dot plot...")

fig, ax = plt.subplots(figsize=(10, 4))

fold_ids = np.arange(1, len(fold_aucs) + 1)

ax.scatter(fold_ids, fold_aucs, s=100, c='darkblue', alpha=0.7, zorder=5)

ax.plot(fold_ids, fold_aucs, 'b-', alpha=0.3, linewidth=1)

ax.axhline(y=mean_auc, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean AUC = {mean_auc:.5f}')
ax.axhline(y=mean_auc + std_auc, color='gray', linestyle=':', linewidth=1,
           alpha=0.7, label=f'±1 Std = {std_auc:.5f}')
ax.axhline(y=mean_auc - std_auc, color='gray', linestyle=':', linewidth=1, alpha=0.7)

for fold_id, auc_val in zip(fold_ids, fold_aucs):
    ax.annotate(f'{auc_val:.5f}', (fold_id, auc_val),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9)

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title(f'{fold_count}-Fold Cross-Validation AUC by Fold', fontsize=14)
ax.set_xticks(fold_ids)
ax.set_ylim(min(fold_aucs) - 0.002, max(fold_aucs) + 0.002)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'figure_6_auc_boxplot.png')
plt.savefig(FIGURE_DIR / 'figure_6_auc_boxplot.pdf')
plt.close(fig)
print(f"   Saved: {FIGURE_DIR / 'figure_6_auc_boxplot.png'}")

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
