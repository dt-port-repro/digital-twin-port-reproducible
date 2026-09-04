"""
Fig4: 算法可扩展性分析 — 对数求解时间 vs 问题规模
三条折线：GA-RH(混合算法)、纯GA(传统GA)、SA(模拟退火)
纵轴对数刻度，展示不同算法的扩展规律
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

# ============================================================
# 1. 分箱
# ============================================================
bins = [400, 600, 800, 1000, 1500, 2000, 4000]
labels = ['400–600', '600–800', '800–1k', '1k–1.5k', '1.5k–2k', '2k–4k']
bin_centers = [500, 700, 900, 1250, 1750, 3000]

df['bin'] = pd.cut(df['n_containers'], bins=bins, labels=labels)

# ============================================================
# 2. 计算均值
# ============================================================
modes = {
    'GA-RH': 'GA-RH（混合算法）',
    '纯GA': 'GA（传统遗传算法）',
    'SA': 'SA（模拟退火）',
}
colors = {'GA-RH': '#E74C3C', '纯GA': '#2E86C1', 'SA': '#28B463'}
markers = {'GA-RH': 'o', '纯GA': 's', 'SA': '^'}

fig, ax = plt.subplots(figsize=(10, 6.5))

# 同时绘制原始数据点（低透明度）和均值折线
for mode, label in modes.items():
    sub = df[df['mode'] == mode]
    means, stds = [], []
    for b in labels:
        d = sub[sub['bin'] == b]['time_s']
        means.append(d.mean() if len(d) > 0 else np.nan)
        stds.append(d.std() if len(d) > 0 else np.nan)
    means = np.array(means)
    stds = np.array(stds)

    # 散点：所有原始数据点（低透明度）
    # 加入少量水平抖动避免重叠
    for b_idx, b in enumerate(labels):
        vals = sub[sub['bin'] == b]['time_s'].values
        jitter = np.random.uniform(-30, 30, size=len(vals))
        ax.scatter([bin_centers[b_idx]] * len(vals) + jitter, vals,
                   color=colors[mode], alpha=0.12, s=12, zorder=1)

    # 均值折线
    ax.plot(bin_centers, means, color=colors[mode], linewidth=2.5,
            marker=markers[mode], markersize=9, label=label, zorder=3)

    # 标注每箱点数
    for i, b in enumerate(labels):
        n = len(sub[sub['bin'] == b])
        if n > 0 and not np.isnan(means[i]):
            ax.annotate(f'n={n}', (bin_centers[i], means[i]),
                        textcoords='offset points', xytext=(0, -22),
                        fontsize=7, color=colors[mode], ha='center',
                        alpha=0.6)

# ============================================================
# 3. 拟合趋势线：展示超线性/线性/次线性增长
# ============================================================
# 对GA-RH和纯GA拟合幂律曲线: t = a * N^b
for mode, color, marker in [('GA-RH', '#E74C3C', 'o'), ('纯GA', '#2E86C1', 's')]:
    sub = df[df['mode'] == mode]
    x_raw = sub['n_containers'].values
    y_raw = sub['time_s'].values
    # 对数域线性拟合
    A = np.vstack([np.log(x_raw), np.ones_like(x_raw)]).T
    coeffs, *_ = np.linalg.lstsq(A, np.log(y_raw), rcond=None)
    b_exp, ln_a = coeffs
    a = np.exp(ln_a)
    x_fit = np.linspace(400, 4000, 100)
    y_fit = a * x_fit ** b_exp
    # 用相同颜色但虚线表示趋势
    ax.plot(x_fit, y_fit, color=color, linewidth=1.2, linestyle='--',
            alpha=0.5, zorder=1)
    # 在图上标注指数
    mid_idx = len(x_fit) // 2
    ax.text(x_fit[mid_idx], y_fit[mid_idx] * 1.1,
            f't ∝ N^{b_exp:.2f}', fontsize=8, color=color, alpha=0.7,
            style='italic')

# ============================================================
# 4. 对数纵轴 + 美化
# ============================================================
ax.set_yscale('log')
ax.set_xlabel('Problem Scale (Containers)', fontsize=13, fontweight='bold')
ax.set_ylabel('Solving Time (seconds, log scale)', fontsize=13, fontweight='bold')
ax.set_title('Fig. Algorithm Scalability Analysis\n(100 vessels × 3 algorithms, log-scale y-axis)',
             fontsize=14, fontweight='bold', pad=12)

ax.set_xticks(bin_centers)
ax.set_xticklabels(labels, fontsize=11)
ax.tick_params(axis='both', labelsize=11)
ax.legend(fontsize=11, framealpha=0.9, edgecolor='#CCCCCC', loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--', which='both')
ax.set_xlim(350, 4100)
ax.set_ylim(0.5, 5000)

# SA近似常数时间标注
ax.annotate('SA: near-constant O(1)\n(fast but infeasible)',
            xy=(2200, 7), fontsize=9, color=colors['SA'],
            style='italic', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['SA'], alpha=0.8))

plt.tight_layout()
png_path = OUT / 'fig_scalability.png'
pdf_path = OUT / 'fig_scalability.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
plt.close()

print(f'✅ 图已保存:')
print(f'   {png_path}')
print(f'   {pdf_path}')

# 打印拟合系数
print()
print('='*50)
print('幂律拟合: t = a × N^b')
print('='*50)
for mode in ['GA-RH', '纯GA']:
    sub = df[df['mode'] == mode]
    x_raw = sub['n_containers'].values
    y_raw = sub['time_s'].values
    A = np.vstack([np.log(x_raw), np.ones_like(x_raw)]).T
    coeffs, *_ = np.linalg.lstsq(A, np.log(y_raw), rcond=None)
    b, ln_a = coeffs
    print(f'{mode:>6}:  a={np.exp(ln_a):.2e},  b={b:.3f}  =>  t ∝ N^{b:.2f}')
