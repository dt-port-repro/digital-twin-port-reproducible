"""
生成双面板收敛曲线图（CNCT + CGAMV）
"""
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

R = Path('experiments/chapter4/results')

# === 加载数据 ===
cnct = pd.read_parquet(R/'convergence_cnct_series.parquet')
cgamv = pd.read_parquet(R/'convergence_cgamv_series.parquet')

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#1a1a2e')

datasets = [
    ('CNCT (795 TEU, 4 ports)', cnct),
    ('CGAMV (2,993 TEU, 8 ports)', cgamv)
]

for ax, (title, data) in zip(axes, datasets):
    ax.set_facecolor('#16213e')
    ax.grid(True, alpha=0.15)
    
    methods = ['GA-RH', 'PureGA']
    colors = ['#00d2ff', '#ff6b6b']
    
    for method, color in zip(methods, colors):
        sub = data[data['method']==method]
        gens = sorted(sub['gen'].unique())
        means, stds = [], []
        for g in gens:
            vals = sub[sub['gen']==g]['fitness'].values
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means); stds = np.array(stds)
        
        ax.plot(gens, means, label=method, color=color, linewidth=2)
        ax.fill_between(gens, means-stds, means+stds, color=color, alpha=0.12)
    
    ax.set_title(title, fontsize=13, color='white', pad=10)
    ax.set_xlabel('Generation', fontsize=11, color='#cccccc')
    ax.set_ylabel('Fitness (normalized)', fontsize=11, color='#cccccc')
    ax.tick_params(colors='#cccccc', labelsize=9)
    ax.legend(fontsize=10, framealpha=0.3, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    for spine in ax.spines.values(): spine.set_color('#444')

plt.tight_layout(pad=2)
out_path = 'output/convergence_curve.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
print(f"✅ {out_path}")

# Print stats
print("\n=== 收敛统计 ===")
for name, data in datasets:
    print(f"\n{name}:")
    for m in ['GA-RH','PureGA']:
        sub = data[data['method']==m]
        final = sub[sub['gen']==sub['gen'].max()]
        print(f"  {m}: {final['fitness'].mean():.4f}+-{final['fitness'].std():.4f}")
