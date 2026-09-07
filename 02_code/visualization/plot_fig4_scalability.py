"""
Fig4.6: 算法可扩展性分析 — 对数求解时间 vs 问题规模（论文图4.6）
100条船 × 3种算法；x 轴 6 档等距均衡；纵轴对数刻度
三条幂律拟合虚线 t = a·N^b（指数上标，a 比例系数，与论文文字 t=aNᵇ 一致），统一延伸至 2k-4k 档右缘
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patheffects import withStroke
from pathlib import Path

matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'output' / 'large_scale'
df = pd.read_parquet(OUT / 'large_scale_results_v2.parquet')

# ============================================================
# 1. 分箱 + 等距档位轴（x = 档位序号 0..5，每档等宽）
# ============================================================
bins = [400, 600, 800, 1000, 1500, 2000, 4000]
labels = ['400–600', '600–800', '800–1k', '1k–1.5k', '1.5k–2k', '2k–4k']
XPOS = np.arange(6)                 # 档位等距坐标
N_C = [500, 700, 900, 1250, 1750, 3000]   # 各档中心箱数（仅用于拟合曲线映射）
C_BND = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]   # 档边界 -> 箱数边界
N_BND = [400, 600, 800, 1000, 1500, 2000, 4000]
df['bin'] = pd.cut(df['n_containers'], bins=bins, labels=labels)

modes = {'GA-RH': 'GA-RH（混合算法）', '纯GA': 'GA（传统遗传算法）', 'SA': 'SA（模拟退火）'}
colors = {'GA-RH': '#D62728', '纯GA': '#1F77B4', 'SA': '#7F7F7F'}
markers = {'GA-RH': 'o', '纯GA': 's', 'SA': '^'}

fig, ax = plt.subplots(figsize=(10.8, 6.6))
n_by_bin = df[df['mode'] == 'GA-RH'].groupby('bin')['vessel_code'].count()

# ============================================================
# 2. 数据点 + 均值实线（等距档位坐标）
# ============================================================
for mode, label in modes.items():
    sub = df[df['mode'] == mode]
    means = []
    for b in labels:
        d = sub[sub['bin'] == b]['time_s']
        means.append(d.mean() if len(d) > 0 else np.nan)
    means = np.array(means)

    for b_idx, b in enumerate(labels):
        vals = sub[sub['bin'] == b]['time_s'].values
        rng = np.random.default_rng(42 + b_idx)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(XPOS[b_idx] + jitter, vals,
                   color=colors[mode], alpha=0.22 if mode != 'SA' else 0.15,
                   s=15 if mode != 'SA' else 10, zorder=2)

    ax.plot(XPOS, means, color=colors[mode], linewidth=3.0,
            marker=markers[mode], markersize=9, label=label, zorder=4,
            markerfacecolor=colors[mode], markeredgecolor='white', markeredgewidth=1.1,
            path_effects=[withStroke(linewidth=5, foreground='white')])

# ============================================================
# 3. 三条幂律拟合虚线 t = a·N^b，统一延伸至 2k-4k 档右缘 (c=5.5)
# ============================================================
fit_results = {}
# 档位坐标 -> 箱数：对数线性映射（log N 随档位线性），使幂律虚线在 log-y 轴上
# 接近笔直，超线性斜率(b>1)清晰可辨，与原图虚线形态一致
c_smooth = np.linspace(-0.5, 5.5, 300)
logN_s = np.interp(c_smooth, C_BND, np.log10(N_BND))
N_s = 10 ** logN_s
for mode, color in [('GA-RH', '#D62728'), ('纯GA', '#1F77B4'), ('SA', '#7F7F7F')]:
    sub = df[df['mode'] == mode]
    x_raw = sub['n_containers'].values
    y_raw = sub['time_s'].values
    A = np.vstack([np.log(x_raw), np.ones_like(x_raw)]).T
    b_exp, ln_a = np.linalg.lstsq(A, np.log(y_raw), rcond=None)[0]
    a = np.exp(ln_a)
    fit_results[mode] = (a, b_exp)
    ax.plot(c_smooth, a * N_s ** b_exp, color=color, linewidth=1.8,
            linestyle='--', alpha=0.95, zorder=3)

# 公式标注（右上空白区：2k-4k 档实测最高 ~1090s，y>1500 无数据）
ax.text(3.4, 3200, f'$t = aN^{{{fit_results["GA-RH"][1]:.2f}}}$',
        fontsize=14, color='#D62728', fontweight='bold', ha='center', va='center', zorder=6,
        path_effects=[withStroke(linewidth=3, foreground='white')])
ax.text(5.0, 3200, f'$t = aN^{{{fit_results["纯GA"][1]:.2f}}}$',
        fontsize=14, color='#1F77B4', fontweight='bold', ha='center', va='center', zorder=6,
        path_effects=[withStroke(linewidth=3, foreground='white')])
# SA 标注：SA 虚线末端上方（x=5.3 无点带；SA 档5 实测最高 26s）
ax.text(4.7, 80, f'$t = aN^{{{fit_results["SA"][1]:.2f}}}$',
        fontsize=12.5, color='#666666', fontweight='bold', ha='center', va='center', zorder=6,
        path_effects=[withStroke(linewidth=3, foreground='white')])

# ============================================================
# 4. 坐标与美化
# ============================================================
ax.set_yscale('log')
ax.set_xlabel('问题规模（集装箱数）', fontsize=14, fontweight='bold')
ax.set_ylabel('求解时间（秒，对数坐标）', fontsize=14, fontweight='bold')

ax.set_xticks(XPOS)
ax.set_xticklabels(labels, fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12.5, framealpha=0.95, edgecolor='#888888',
          loc='upper left', borderpad=0.8)
ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.7, which='both')
ax.set_xlim(-0.62, 5.8)
ax.set_ylim(0.5, 6000)

# n= 样本量标注：每箱一次（底部空白带 y≈0.85，SA 实测最低 1.5s 不重叠）
for b_idx in range(6):
    n = int(n_by_bin[labels[b_idx]])
    ax.text(XPOS[b_idx], 0.85, f'n={n}', fontsize=10, color='#666666',
            ha='center', va='center', zorder=6)

# SA 注释框：SA 实线下方靠右的空白带（x≈4.5 无点带，y 3~10 < SA 档4 最低 6.7 部分重叠规避，
# 取 y=5 中心，避开 SA 虚线(此处 ~12s)与散点）
ax.text(4.15, 3.6, 'SA: 快速但不可行', fontsize=11, color='#4A4A4A',
        ha='center', va='center', zorder=6,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#7F7F7F', alpha=0.95))

plt.tight_layout()
png_path = OUT / 'fig_scalability.png'
pdf_path = OUT / 'fig_scalability.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
plt.close()

print('图已保存:', png_path)
print('=' * 55)
print('幂律拟合 t = a × N^b (a 为比例系数，论文文字口径; 虚线均延伸至 2k-4k 档)')
print('=' * 55)
for mode in ['GA-RH', '纯GA', 'SA']:
    a, b = fit_results[mode]
    print(f'{mode:>6}:  a = {a:.2e},  b = {b:.3f}   =>   t = a·N^{b:.2f}')
