#!/usr/bin/env python3
"""生成图4.7收敛曲线配套表格 + 重新生成图4.7"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONV = ROOT / 'output' / 'convergence'
TABLES = ROOT / '03_results' / 'tables'
FIG = ROOT / 'output' / 'large_scale'
TABLES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

hist = pd.read_parquet(CONV / 'convergence_history.parquet')
results = pd.read_parquet(CONV / 'convergence_results.parquet')

# ========== 表 S6: 收敛曲线数据 ==========
print("生成表S6...")
# 每代每方法的均值±标准差
rows = []
for v in ['CNCT', 'CGAMV']:
    for m in ['GA-RH', '纯GA']:
        sub = hist[(hist['vessel_code']==v) & (hist['mode']==m)]
        grp = sub.groupby('generation')['fitness'].agg(['mean','std']).reset_index()
        for _, r in grp.iterrows():
            rows.append({'vessel': v, 'method': m, 'generation': int(r['generation']),
                         'fitness_mean': round(r['mean'], 6), 'fitness_std': round(r['std'], 6)})

table_hist = pd.DataFrame(rows)
table_hist.to_csv(TABLES / 'table_S6_convergence_history.csv', index=False, encoding='utf-8-sig')

# 汇总表: 5轮统计数据
rows2 = []
for v in ['CNCT', 'CGAMV']:
    for m in ['GA-RH', '纯GA']:
        sub = results[(results['vessel_code']==v) & (results['mode']==m)]
        rows2.append({
            'vessel': v, 'n_containers': sub['n_containers'].iloc[0],
            'method': m, 'n_runs': len(sub),
            'fitness_mean': round(sub['fitness'].mean(), 4),
            'fitness_std': round(sub['fitness'].std(), 4),
            'f1_mean': round(sub['f1'].mean(), 4),
            'f2_mean': round(sub['f2'].mean(), 4),
            'time_s_mean': round(sub['time_s'].mean(), 1),
        })
table_summary = pd.DataFrame(rows2)
table_summary.to_csv(TABLES / 'table_S6_convergence_summary.csv', index=False, encoding='utf-8-sig')
print(f"  ✅ table_S6_convergence_history.csv ({len(table_hist)} rows)")
print(f"  ✅ table_S6_convergence_summary.csv")

# ========== 新图4.7: 收敛曲线 ==========
print("生成新图4.7...")
fontsize = 12
colors = {'GA-RH': '#4A7FB5', '纯GA': '#E8A838'}
labels_map = {'GA-RH': 'GA-RH', '纯GA': '纯GA'}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for idx, vessel in enumerate(['CNCT', 'CGAMV']):
    ax = axes[idx]
    for mode in ['GA-RH', '纯GA']:
        sub = hist[(hist['vessel_code'] == vessel) & (hist['mode'] == mode)]
        grouped = sub.groupby('generation')['fitness'].agg(['mean', 'std']).reset_index()
        gens = grouped['generation'].values
        mean = grouped['mean'].values
        std = grouped['std'].values

        ax.plot(gens, mean, color=colors[mode], linewidth=1.5, label=labels_map[mode])
        ax.fill_between(gens, mean - std, mean + std, color=colors[mode], alpha=0.15)

    n_box = results[results['vessel_code']==vessel]['n_containers'].iloc[0]
    ax.set_title(f'{vessel} ({n_box}箱)', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('进化代数', fontsize=fontsize)
    if idx == 0:
        ax.set_ylabel('Fitness', fontsize=fontsize)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 32)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.0)
    ax.tick_params(labelsize=10)

plt.tight_layout()
path = FIG / 'fig4_7_convergence_curve_updated.png'
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✅ {path}")
print("完成")
