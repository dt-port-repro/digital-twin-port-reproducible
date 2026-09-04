"""
Fig3A: GA-RH优势 vs 问题复杂度
横轴：问题复杂度（箱数×目的港数）
纵轴：GA-RH胜率(%) + 平均优势幅度
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
# 1. 定义复杂度 = n_containers × n_pod (箱数×目的港数)
# ============================================================
complexity = garh.loc[common, 'n_containers'] * garh.loc[common, 'n_pod']

# 分箱
bins_c = [0, 1500, 3000, 5000, 8000, 12000, 99999]
labels_c = ['≤1.5k', '1.5k–3k', '3k–5k', '5k–8k', '8k–12k', '>12k']
bin_centers_c = [750, 2250, 4000, 6500, 10000, 16000]

complexity_bin = pd.cut(complexity, bins=bins_c, labels=labels_c)

# ============================================================
# 2. 每档计算胜率和优势幅度
# ============================================================
delta_fit = garh.loc[common, 'fitness'] - pure.loc[common, 'fitness']

win_rates, adv_means, n_ships = [], [], []
for b in labels_c:
    mask = complexity_bin == b
    n = mask.sum()
    if n == 0:
        win_rates.append(np.nan)
        adv_means.append(np.nan)
        n_ships.append(0)
        continue
    d = delta_fit[mask]
    win_rates.append((d > 0).mean() * 100)
    adv_means.append(d.mean() * 100)  # 转为百分比
    n_ships.append(n)

# ============================================================
# 3. 绘图：双Y轴
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

bars = ax1.bar(bin_centers_c, win_rates, width=1200, color='#E74C3C',
               alpha=0.7, edgecolor='white', linewidth=0.5,
               label='GA-RH Win Rate (%)', zorder=2)

# 柱上标注百分比
for bar, wr, n in zip(bars, win_rates, n_ships):
    if not np.isnan(wr):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{wr:.0f}%\n(n={n})', ha='center', fontsize=9,
                 color='#333', fontweight='bold')

ax1.set_xlabel('Problem Complexity (Containers × Ports of Destination)',
               fontsize=13, fontweight='bold')
ax1.set_ylabel('GA-RH Win Rate (%)', fontsize=13, color='#E74C3C', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#E74C3C', labelsize=11)
ax1.set_ylim(0, 105)
ax1.set_xticks(bin_centers_c)
ax1.set_xticklabels(labels_c, fontsize=11)

# 第二Y轴：平均优势幅度
ax2 = ax1.twinx()
ax2.plot(bin_centers_c, adv_means, color='#2E86C1', linewidth=2.5,
         marker='s', markersize=9, label='Avg Advantage (×100)', zorder=3)
ax2.set_ylabel('Avg Fitness Advantage (GA-RH − Pure GA, ×10⁻²)',
               fontsize=13, color='#2E86C1', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#2E86C1', labelsize=11)
ax2.axhline(y=0, color='#888', linewidth=0.8, linestyle=':', alpha=0.6)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11,
           framealpha=0.9, edgecolor='#CCCCCC', loc='upper left')

ax1.set_title('Fig. GA-RH Advantage Grows with Problem Complexity\n(100 vessels, grouped by complexity)',
              fontsize=14, fontweight='bold', pad=12)
ax1.grid(True, axis='y', alpha=0.2, linestyle='--')

# 注释
ax1.text(0.98, 0.05,
         'Win rate = proportion of ships where\nGA-RH achieves higher fitness than Pure GA',
         transform=ax1.transAxes, fontsize=8, color='#666',
         ha='right', va='bottom', style='italic',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='#DDD', alpha=0.9))

plt.tight_layout()
png_path = OUT / 'fig3a_advantage_vs_complexity.png'
pdf_path = OUT / 'fig3a_advantage_vs_complexity.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
plt.close()
print(f'✅ Fig3A: {png_path}')

# ==================== 打印数据 ====================
print()
print(f'{"Complexity":<12} {"n":<4} {"WinRate":<10} {"AvgAdv(×10⁻²)":<15}')
print('-'*50)
for i, b in enumerate(labels_c):
    if n_ships[i] > 0:
        print(f'{b:<12} {n_ships[i]:<4} {win_rates[i]:<+9.0f}% {adv_means[i]:<+14.2f}')
