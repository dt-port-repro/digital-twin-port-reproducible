"""
Fig3B: 子目标维度对比 — GA-RH vs 纯GA
四个维度：f₁翻箱成本、f₂装卸效率、fitness解质量、约束违反
分组柱状图 + 差异标注
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'output' / 'large_scale'
df = pd.read_parquet(OUT / 'large_scale_results_v2.parquet')

garh = df[df['mode']=='GA-RH'].set_index('berth_plan_no')
pure = df[df['mode']=='纯GA'].set_index('berth_plan_no')
common = garh.index.intersection(pure.index)

# ============================================================
# 1. 维度定义
# ============================================================
dimensions = [
    ('f1', 'Rehandle\nCost (f₁)', 'higher is better', True),
    ('f2', 'Loading\nEfficiency (f₂)', 'higher is better', True),
    ('fitness', 'Solution\nQuality (Fitness)', 'higher is better', True),
    ('penalty', 'Constraint\nViolation (Penalty)', 'lower is better', False),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))

# 分别对"常规问题"和"难题"做子图
scenarios = [
    ('All Ships (n=97)', common),
    ('Hard Problems (n=20)', garh.loc[common].nlargest(20, 'n_containers').index),
]

for col_idx, (scenario_name, ship_set) in enumerate(scenarios):
    ax = axes[col_idx]
    c = common.intersection(ship_set)
    
    x = np.arange(len(dimensions))
    width = 0.30
    
    garh_means, pure_means, garh_stds, pure_stds = [], [], [], []
    for col, _, higher_better, _ in dimensions:
        g = garh.loc[c, col]
        p = pure.loc[c, col]
        garh_means.append(g.mean())
        pure_means.append(p.mean())
        garh_stds.append(g.std())
        pure_stds.append(p.std())
    
    bars1 = ax.bar(x - width/2, garh_means, width, yerr=garh_stds,
                   color='#E74C3C', alpha=0.8, capsize=3, edgecolor='white',
                   linewidth=0.5, label='GA-RH', zorder=2)
    bars2 = ax.bar(x + width/2, pure_means, width, yerr=pure_stds,
                   color='#2E86C1', alpha=0.8, capsize=3, edgecolor='white',
                   linewidth=0.5, label='Pure GA', zorder=2)
    
    # 差异标注
    for i, (col, label, higher_better, _) in enumerate(dimensions):
        d = garh.loc[c, col].mean() - pure.loc[c, col].mean()
        better = (d > 0) if higher_better else (d < 0)
        sign = '+' if d >= 0 else ''
        color = '#27AE60' if better else '#E74C3C'
        ax.text(i, max(garh_means[i], pure_means[i]) + max(garh_stds[i], pure_stds[i]) + 0.02,
                f'Δ={sign}{d:.4f}', ha='center', fontsize=7.5,
                color=color, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l, _, _ in dimensions], fontsize=9)
    ax.set_title(scenario_name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')
    ax.tick_params(labelsize=9)

# ============================================================
# 2. 第二行：按不同规模分档看差异趋势
# ============================================================
bins = [400, 600, 800, 1000, 1500, 2000, 4000]
labels_bin = ['400–600', '600–800', '800–1k', '1k–1.5k', '1.5k–2k', '2k–4k']
bin_centers = np.arange(len(labels_bin))
df['bin'] = pd.cut(df['n_containers'], bins=bins, labels=labels_bin)

# 下方合并热力图：不同规模下各维度差异方向
fig2, ax2 = plt.subplots(figsize=(10, 4.5))

# 构建差异矩阵 (sign × magnitude)
dim_cols = ['f1', 'f2', 'fitness', 'penalty']
dim_labels_display = ['Rehandle (f₁)', 'Efficiency (f₂)', 'Fitness', 'Penalty']
diff_matrix = np.zeros((len(labels_bin), len(dim_cols)))

for i, b in enumerate(labels_bin):
    bpns = set(df[df['bin'] == b]['berth_plan_no'].unique())
    c = common.intersection(bpns)
    for j, col in enumerate(dim_cols):
        d = garh.loc[c, col].mean() - pure.loc[c, col].mean()
        diff_matrix[i, j] = d

# 差异热力图
norm = plt.Normalize(vmin=-max(abs(diff_matrix.min()), abs(diff_matrix.max())),
                     vmax=max(abs(diff_matrix.min()), abs(diff_matrix.max())))
im = ax2.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', norm=norm)

ax2.set_xticks(range(len(dim_cols)))
ax2.set_xticklabels(dim_labels_display, fontsize=10)
ax2.set_yticks(range(len(labels_bin)))
ax2.set_yticklabels(labels_bin, fontsize=10)
ax2.set_xlabel('Dimension', fontsize=12, fontweight='bold')
ax2.set_ylabel('Problem Scale (Containers)', fontsize=12, fontweight='bold')

# 格内数值
for i in range(len(labels_bin)):
    for j in range(len(dim_cols)):
        v = diff_matrix[i, j]
        text_color = 'white' if abs(v) > max(abs(diff_matrix.min()), abs(diff_matrix.max())) * 0.5 else 'black'
        ax2.text(j, i, f'{v:+.4f}', ha='center', va='center',
                fontsize=8, color=text_color, fontweight='bold')

plt.colorbar(im, ax=ax2, label='GA-RH − Pure GA', shrink=0.8)
ax2.set_title('Fig. GA-RH vs. Pure GA: Dimension-wise Advantage by Scale\n(green = GA-RH better; red = Pure GA better)',
              fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
png1 = OUT / 'fig3b_subobjective_comparison.png'
pdf1 = OUT / 'fig3b_subobjective_comparison.pdf'
fig.savefig(png1, dpi=300, bbox_inches='tight')
fig.savefig(pdf1, bbox_inches='tight')
plt.close(fig)

png2 = OUT / 'fig3b_heatmap.png'
pdf2 = OUT / 'fig3b_heatmap.pdf'
fig2.savefig(png2, dpi=300, bbox_inches='tight')
fig2.savefig(pdf2, bbox_inches='tight')
plt.close(fig2)

print(f'✅ Fig3B-1: {png1}')
print(f'✅ Fig3B-2: {png2}')
