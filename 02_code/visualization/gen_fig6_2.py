"""Generate Fig 6.3 - Peak scenario config comparison
Data source: 03_results/canonical/sim_all_cfgall_d30r10.parquet
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / '03_results' / 'canonical'
OUT = ROOT / '03_results' / 'figures'
# Font（跨平台：Windows使用微软雅黑，Linux/Mac使用系统字体）
zh_font = None
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyhbd.ttc',
           '/System/Library/Fonts/PingFang.ttc', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
    if Path(fp).exists():
        zh_font = fm.FontProperties(fname=fp)
        break
if zh_font is None:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

sim = pd.read_parquet(DATA / 'sim_all_cfgall_d30r10.parquet')
sub = sim[sim.scenario=='高峰压力']

configs = ['A\n(基线)', 'B\n(纯配载)', 'C\n(组合)', 'D\n(协同)']
labels_short = ['A', 'B', 'C', 'D']

a = sub[sub.config=='A'][['turnaround_h','reshuffle_pct','equip_util_pct']].mean()
turnaround = [sub[sub.config==c]['turnaround_h'].mean() for c in ['A','B','C','D']]
reshuffle = [sub[sub.config==c]['reshuffle_pct'].mean() for c in ['A','B','C','D']]
equip_util = [sub[sub.config==c]['equip_util_pct'].mean() for c in ['A','B','C','D']]
imp_t = [(a.turnaround_h - t) / a.turnaround_h * 100 if t < a.turnaround_h else 0 for t in turnaround]
imp_r = [(a.reshuffle_pct - r) / a.reshuffle_pct * 100 if r < a.reshuffle_pct else 0 for r in reshuffle]
imp_e = [(e - a.equip_util_pct) / a.equip_util_pct * 100 if e > a.equip_util_pct else 0 for e in equip_util]

colors_cfg = {'A': '#8ECFC9', 'B': '#FFBE7A', 'C': '#FA7F6F', 'D': '#82B0D2'}
bar_colors = [colors_cfg[c] for c in labels_short]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.patch.set_facecolor('white')
x = np.arange(4)
bw = 0.55

for idx, (ax, vals, imps, ylabel, ylim, title) in enumerate([
    (axes[0], turnaround, imp_t, '船舶在港时间 (h)', (14.5, 17.5), '(a) 船舶在港时间'),
    (axes[1], reshuffle, imp_r, '堆场翻箱率 (%)', (4, 8.5), '(b) 堆场翻箱率'),
    (axes[2], equip_util, imp_e, '设备利用率 (%)', (55, 64), '(c) 设备利用率'),
]):
    bars = ax.bar(x, vals, bw, color=bar_colors, edgecolor='none', zorder=3)
    for i, (v, imp) in enumerate(zip(vals, imps)):
        if imp > 0:
            arrow = '↓' if idx < 2 else '↑'
            ax.annotate(f'{arrow}{imp:.1f}%', (x[i], v), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold',
                        color='#2d6a4f')
    ax.set_ylabel(ylabel, fontsize=11, fontproperties=zh_font if zh_font else None)
    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9, fontproperties=zh_font if zh_font else None)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title(title, fontsize=11, fontweight='bold', fontproperties=zh_font if zh_font else None, pad=8)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

plt.tight_layout(pad=2)
OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT / 'fig6_3_masdes_peak.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(OUT / 'fig6_3_masdes_peak.svg', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✅ fig6_3_masdes_peak.png')
