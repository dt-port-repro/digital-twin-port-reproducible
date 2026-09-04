#!/usr/bin/env python3
"""生成图4.8(参数敏感性热图)+图4.9(pop效应) + 配套表S7/S8"""
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
FS = 12
BLUE, ORANGE = '#4A7FB5', '#E8A838'

param = pd.read_parquet(CONV / 'param_sensitivity.parquet')
pops = sorted(param['pop_size'].unique())
gens = sorted(param['generations'].unique())

# ===================== 表S7: 网格搜索数据 =====================
print("生成表S7...")
rows = []
for v in ['CNCT','CGAMV']:
    sub = param[param['vessel_code']==v]
    for pop in pops:
        for gen in gens:
            r = sub[(sub['pop_size']==pop)&(sub['generations']==gen)].iloc[0]
            rows.append({
                'vessel': v, 'n_containers': int(r['n_containers']),
                'pop_size': pop, 'generations': gen,
                'fitness': round(r['fitness'], 6),
                'f1': round(r['f1'], 4),
                'f2': round(r['f2'], 4),
                'penalty': round(r['penalty'], 4),
                'time_s': round(r['time_s'], 1),
            })
pd.DataFrame(rows).to_csv(TABLES / 'table_S7_param_sensitivity.csv', index=False, encoding='utf-8-sig')
print(f"  ✅ table_S7_param_sensitivity.csv ({len(rows)} rows)")

# ===================== 表S8: pop效应数据 =====================
print("生成表S8...")
rows2 = []
for v in ['CNCT','CGAMV']:
    sub = param[param['vessel_code']==v]
    for pop in pops:
        s = sub[sub['pop_size']==pop]
        rows2.append({
            'vessel': v, 'pop_size': pop,
            'fitness_mean': round(s['fitness'].mean(), 6),
            'fitness_std': round(s['fitness'].std(), 6),
            'time_s_mean': round(s['time_s'].mean(), 1),
        })
pd.DataFrame(rows2).to_csv(TABLES / 'table_S8_pop_effect.csv', index=False, encoding='utf-8-sig')
print(f"  ✅ table_S8_pop_effect.csv ({len(rows2)} rows)")

# ===================== 图4.8: 热力图 =====================
print("生成图4.8...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for idx, vessel in enumerate(['CNCT', 'CGAMV']):
    ax = axes[idx]
    sub = param[param['vessel_code'] == vessel]
    pivot = sub.pivot_table(index='generations', columns='pop_size', values='fitness', aggfunc='mean')
    vmin, vmax = pivot.values.min(), pivot.values.max()
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=vmin*0.98, vmax=vmax*1.02)
    ax.set_xticks(range(len(pops)))
    ax.set_xticklabels([f'pop={p}' for p in pops], fontsize=9)
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels([f'gen={g}' for g in gens], fontsize=9)
    ax.set_xlabel('种群规模', fontsize=FS)
    ax.set_ylabel('迭代代数', fontsize=FS)
    for i in range(len(gens)):
        for j in range(len(pops)):
            v = pivot.values[i, j]
            tc = 'white' if v > pivot.values.mean() else 'black'
            ax.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
    best = np.unravel_index(pivot.values.argmax(), pivot.values.shape)
    ax.scatter(best[1], best[0], s=200, facecolors='none', edgecolors='#2ecc71', linewidth=2.5)
    ax.set_title(f'{vessel}', fontsize=FS, fontweight='bold')
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Fitness')
plt.tight_layout()
plt.savefig(FIG / 'fig4_8_param_sensitivity_updated.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ fig4_8")

# ===================== 图4.9: pop效应 =====================
print("生成图4.9...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for idx, vessel in enumerate(['CNCT', 'CGAMV']):
    ax = axes[idx]
    sub = param[param['vessel_code'] == vessel]
    pop_means = sub.groupby('pop_size')['fitness'].agg(['mean','std'])
    pop_time = sub.groupby('pop_size')['time_s'].mean()
    x = np.arange(len(pop_means))
    w = 0.3
    bars = ax.bar(x - w/2, pop_means['mean'].values, w, yerr=pop_means['std'].values,
                  capsize=3, color=BLUE, alpha=0.85, label='Fitness', edgecolor='white')
    ax2 = ax.twinx()
    ax2.plot(x, pop_time.values, 's-', color=ORANGE, linewidth=2.0, markersize=7, label='耗时')
    ax.set_xticks(x)
    ax.set_xticklabels([f'pop={int(p)}' for p in pop_means.index], fontsize=9)
    ax.set_xlabel('种群规模', fontsize=FS)
    ax.set_ylabel('Fitness', fontsize=FS, color=BLUE)
    ax2.set_ylabel('耗时 (s)', fontsize=FS, color=ORANGE)
    ax.set_title(f'{vessel}', fontsize=FS, fontweight='bold')
    ax.grid(alpha=0.2, axis='y')
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax2.spines['top'].set_visible(True)
    ax2.spines['top'].set_color('black')
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('black')
    for i, (m, s) in enumerate(zip(pop_means['mean'].values, pop_means['std'].values)):
        ax.text(i, m + s + 0.003, f'{m:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / 'fig4_9_pop_effect_updated.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ fig4_9")
print("全部完成")
