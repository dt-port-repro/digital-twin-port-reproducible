#!/usr/bin/env python3
"""Figure 5.2 — 训练曲线 + 分步MAPE双轴图"""
import json, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.unicode_minus'] = False

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PKG, 'output/chapter5/results/lstm_results.json')
FIG_OUT = os.path.join(PKG, 'output/chapter5')

# 也支持从工作目录读
FALLBACK = os.path.join(PKG, 'experiments', 'chapter5', 'results', 'lstm_results.json')

fp = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK
with open(fp) as f:
    data = json.load(f)

C_BLUE = '#4A7FB5'
C_ORANGE = '#E8A838'
C_BROWN = '#8B5E3C'
C_GRID = '#D6E4F0'

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
fig.patch.set_facecolor('white')
windows = [('w1', 'W1 (1-6月→7-8月)'), ('w2', 'W2 (1-9月→10-12月)')]
steps = ['h+1','h+2','h+3','h+4','h+5','h+6','h+7']

for i, (wname, wlabel) in enumerate(windows):
    ax = axes[i]
    hist = data[wname]['train_history']
    epochs = [h['epoch']+1 for h in hist]
    train_loss = [h['train_loss'] for h in hist]
    val_loss = [h['val_loss'] for h in hist]
    
    # Loss curves
    ax.plot(epochs, train_loss, color=C_BLUE, linewidth=1.8, label='训练损失 (train loss)', zorder=5)
    ax.plot(epochs, val_loss, color=C_ORANGE, linewidth=1.8, label='验证损失 (val loss)', zorder=5)
    
    # Best epoch marker
    best_idx = int(np.argmin(val_loss))
    best_ep = epochs[best_idx]
    best_vl = val_loss[best_idx]
    ax.axvline(x=best_ep, color='#888888', linestyle='--', linewidth=0.8, alpha=0.6, zorder=3)
    ax.annotate(f'最优 epoch {best_ep}\n(val loss={best_vl:.4f})',
                xy=(best_ep, best_vl), fontsize=9,
                xytext=(best_ep+8, best_vl+0.10),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9),
                zorder=6)
    
    # Step MAPE bars (thin, on the left side, using secondary x-axis concept)
    ax2 = ax.twinx()
    metrics = data[wname]['metrics']
    step_map = [float(metrics[s]['mape']) for s in steps]
    
    # Place bars at positions 0-6 (left side of x-axis)
    bar_x = np.arange(len(steps))
    bars = ax2.bar(bar_x, step_map, width=0.55, color=C_BROWN, alpha=0.3, zorder=2,
                   label='分步MAPE (step MAPE %)')
    
    # Value labels ABOVE bars (vertical)
    for j, (bx, val) in enumerate(zip(bar_x, step_map)):
        ax2.text(bx, val + 0.1, f'{val:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold',
                 color=C_BROWN, zorder=7, rotation=90)
    
    # Axis labels
    ax.set_xlabel('训练轮数 (epoch)', fontsize=13, fontweight='bold')
    ax.set_ylabel('损失 (loss)', fontsize=13, color=C_BLUE, fontweight='bold')
    ax2.set_ylabel('MAPE (%)', fontsize=13, color=C_BROWN, fontweight='bold')
    ax.set_title(wlabel, fontsize=14, fontweight='bold', pad=12)
    
    # Grid: light blue
    ax.grid(True, color=C_GRID, linestyle='-', linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    
    # Limit x to show both loss curves and bars
    ax.set_xlim(-0.5, max(epochs)+3)
    
    # Tick sizes
    ax.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    
    # Combined legend in upper right, above bars
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, 
                       loc='upper right', framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout(pad=1.5)
outpath = os.path.join(FIG_OUT, 'fig5_2_training_curve_updated.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f'✅ v3 saved: {outpath}')
