#!/usr/bin/env python3
"""Figure 5.4 (a)(b)(c) — 堆场选位对比图

读取 selection_results_v2.json 生成：
  (a) 平均惩罚值对比柱状图  FCFS | 三阶段 | 三阶段+虚拟占位
  (b) 惩罚值标准差对比柱状图
  (c) 提升率对比分组柱状图

用法: python gen_fig5_4.py
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 字体 ──────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.unicode_minus'] = False

# ── 路径 ──────────────────────────────────────
# 使用复现集内路径
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_OUT = os.path.join(PKG, 'output/chapter5')

# 读取数据：优先用 JSON（原始），回退到 CSV
json_path = os.path.join(PKG, 'output/chapter5/results/selection_results_v2.json')

data = None
if os.path.exists(json_path):
    with open(json_path) as f:
        data = json.load(f)
else:
    # fallback: 从 CSV 重建
    import csv
    csv_path = os.path.join(PKG, '03_results/tables/table_S11_selection_comparison.csv')
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append({
                "method": row["method"],
                "avg_penalty": float(row["avg_penalty"]),
                "std_penalty": float(row["std_penalty"]),
                "n_selected": int(row["n_selected"]),
                "total_time_s": float(row["total_time_s"])
            })

methods = [d['method'] for d in data]
avg_vals = [d['avg_penalty'] for d in data]
std_vals = [d['std_penalty'] for d in data]

# 提升率：以 FCFS 为基准
baseline_idx = 0  # FCFS
avg_base = avg_vals[baseline_idx]
std_base = std_vals[baseline_idx]
improve_avg = [(avg_base - v) / avg_base * 100 if avg_base > 0 else 0 for v in avg_vals]
improve_std = [(std_base - v) / std_base * 100 if std_base > 0 else 0 for v in std_vals]

# ── 配色 ──────────────────────────────────────
C_BLUE = '#4A7FB5'
C_ORANGE = '#E8A838'
C_GREEN = '#5A9E6F'

bar_colors = [C_BLUE, C_ORANGE, C_GREEN]

# ── (a) 平均惩罚值 ────────────────────────────
fig_a, ax_a = plt.subplots(1, 1, figsize=(6.5, 5))
fig_a.patch.set_facecolor('white')

x = np.arange(len(methods))
bars_a = ax_a.bar(x, avg_vals, width=0.50, color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)

for i, (b, v) in enumerate(zip(bars_a, avg_vals)):
    ax_a.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
             f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
             color=bar_colors[i], zorder=5)

ax_a.set_xticks(x)
ax_a.set_xticklabels(methods, fontsize=11)
ax_a.set_ylabel('平均惩罚值', fontsize=14, fontweight='bold')
ax_a.set_ylim(0, max(avg_vals) * 1.25)
ax_a.grid(axis='y', color='#D6E4F0', linestyle='-', linewidth=0.6, alpha=0.8)
ax_a.set_axisbelow(True)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

out_a = os.path.join(FIG_OUT, 'fig5_4a_penalty_comparison.png')
plt.savefig(out_a, dpi=200, bbox_inches='tight')
plt.close()
print(f'✅ (a) saved: {out_a}')


# ── (b) 标准差 ────────────────────────────────
fig_b, ax_b = plt.subplots(1, 1, figsize=(6.5, 5))
fig_b.patch.set_facecolor('white')

bars_b = ax_b.bar(x, std_vals, width=0.50, color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)

for i, (b, v) in enumerate(zip(bars_b, std_vals)):
    ax_b.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
             f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
             color=bar_colors[i], zorder=5)

ax_b.set_xticks(x)
ax_b.set_xticklabels(methods, fontsize=11)
ax_b.set_ylabel('惩罚值标准差', fontsize=14, fontweight='bold')
ax_b.set_ylim(0, max(std_vals) * 1.25)
ax_b.grid(axis='y', color='#D6E4F0', linestyle='-', linewidth=0.6, alpha=0.8)
ax_b.set_axisbelow(True)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

out_b = os.path.join(FIG_OUT, 'fig5_4b_std_comparison.png')
plt.savefig(out_b, dpi=200, bbox_inches='tight')
plt.close()
print(f'✅ (b) saved: {out_b}')


# ── (c) 提升率对比分组柱状图 ─────────────────
fig_c, ax_c = plt.subplots(1, 1, figsize=(7, 5))
fig_c.patch.set_facecolor('white')

categories = ['平均惩罚值降低', '标准差降低']
n_cats = len(categories)
n_methods = len(methods)  # FCFS, 三阶段, +虚拟
bar_width = 0.22
x_cats = np.arange(n_cats)

# 准备数据 [方法 × 指标]
vals_matrix = [
    improve_avg,   # 三个方法的 avg 降低率
    improve_std,   # 三个方法的 std 降低率
]

colors_c = [C_BLUE, C_ORANGE, C_GREEN]
hatches = ['', '//', '..']

for m in range(n_methods):
    vals = [vals_matrix[c][m] for c in range(n_cats)]
    offset = (m - (n_methods - 1) / 2) * bar_width
    bars = ax_c.bar(x_cats + offset, vals, width=bar_width, 
                    color=colors_c[m], edgecolor='white', linewidth=0.8,
                    label=methods[m], zorder=3)
    for b, v in zip(bars, vals):
        if v > 1:
            ax_c.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color=colors_c[m], zorder=5)

ax_c.set_xticks(x_cats)
ax_c.set_xticklabels(categories, fontsize=12)
ax_c.set_ylabel('相对于 FCFS 的降低率 (%)', fontsize=13, fontweight='bold')
ax_c.set_ylim(0, max(map(max, vals_matrix)) * 1.35)
ax_c.grid(axis='y', color='#D6E4F0', linestyle='-', linewidth=0.6, alpha=0.8)
ax_c.set_axisbelow(True)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='#cccccc')

out_c = os.path.join(FIG_OUT, 'fig5_4c_improvement_rates.png')
plt.savefig(out_c, dpi=200, bbox_inches='tight')
plt.close()
print(f'✅ (c) saved: {out_c}')

# 也生成一张合并版供快速预览
# 2×2 布局: (a)(b) 上排, (c) 下排
fig_all, axes = plt.subplots(2, 2, figsize=(14, 10))
fig_all.patch.set_facecolor('white')

# (a) - 左上
ax_a2 = axes[0, 0]
bars_a2 = ax_a2.bar(x, avg_vals, width=0.45, color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)
for i, (b, v) in enumerate(zip(bars_a2, avg_vals)):
    ax_a2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003, f'{v:.4f}',
              ha='center', va='bottom', fontsize=9, fontweight='bold', color=bar_colors[i])
ax_a2.set_xticks(x); ax_a2.set_xticklabels(methods, fontsize=9)
ax_a2.set_ylabel('平均惩罚值', fontsize=12, fontweight='bold')
ax_a2.set_ylim(0, max(avg_vals)*1.25)
ax_a2.grid(axis='y', color='#D6E4F0', linestyle='-', linewidth=0.6, alpha=0.8)
ax_a2.set_axisbelow(True)
ax_a2.spines['top'].set_visible(False); ax_a2.spines['right'].set_visible(False)

# (b) - 右上
ax_b2 = axes[0, 1]
bars_b2 = ax_b2.bar(x, std_vals, width=0.45, color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)
for i, (b, v) in enumerate(zip(bars_b2, std_vals)):
    ax_b2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002, f'{v:.4f}',
              ha='center', va='bottom', fontsize=9, fontweight='bold', color=bar_colors[i])
ax_b2.set_xticks(x); ax_b2.set_xticklabels(methods, fontsize=9)
ax_b2.set_ylabel('惩罚值标准差', fontsize=12, fontweight='bold')
ax_b2.set_ylim(0, max(std_vals)*1.25)
ax_b2.grid(axis='y', color='#D6E4F0', linestyle='-', linewidth=0.6, alpha=0.8)
ax_b2.set_axisbelow(True)
ax_b2.spines['top'].set_visible(False); ax_b2.spines['right'].set_visible(False)

# (c) - 下排，跨两列居中
gs = axes[1, 0].get_gridspec()
for ax in axes[1, :]:
    ax.remove()
ax_c2 = fig_all.add_subplot(gs[1, :])

for m in range(n_methods):
    vals = [vals_matrix[c][m] for c in range(n_cats)]
    offset = (m - (n_methods - 1) / 2) * bar_width
    bars = ax_c2.bar(x_cats + offset, vals, width=bar_width,
                     color=colors_c[m], edgecolor='white', linewidth=0.8,
                     label=methods[m], zorder=3)
    for b, v in zip(bars, vals):
        if v > 1:
            ax_c2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                      f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
                      fontweight='bold', color=colors_c[m])

ax_c2.set_xticks(x_cats); ax_c2.set_xticklabels(categories, fontsize=10)
ax_c2.set_ylabel('相对于FCFS的降低率 (%)', fontsize=12, fontweight='bold')
ax_c2.set_ylim(0, max(map(max, vals_matrix))*1.35)
ax_c2.grid(axis='y', color='#D6E4F0', linestyle='-', linewidth=0.6, alpha=0.8)
ax_c2.set_axisbelow(True)
ax_c2.spines['top'].set_visible(False); ax_c2.spines['right'].set_visible(False)
ax_c2.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#cccccc')

fig_all.suptitle('图5.4  堆场选位实验结果对比', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_all = os.path.join(FIG_OUT, 'fig5_4_selection_comparison.png')
plt.savefig(out_all, dpi=200, bbox_inches='tight')
plt.close()
print(f'✅ merged saved: {out_all}')
