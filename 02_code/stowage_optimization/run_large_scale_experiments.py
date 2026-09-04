#!/usr/bin/env python3
"""
大规模实验复现脚本 — 一键生成图4.2~4.6 + 配套表格S1~S5
数据源：reproduce/output/canonical/exp_large_scale.parquet (100船×3算法)
        experiments/chapter6/results/exp6_scalability.parquet (5船可扩展性)
输出：
  03_results/tables/table_S[1-5]_*.csv  +  output/large_scale/fig4_[2-6]_*.png

运行: python reproduce/run_large_scale_experiments.py
依赖: pandas, numpy, matplotlib, pyarrow
"""

import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL = ROOT / 'reproduce' / 'output' / 'canonical'
TABLES = ROOT / '03_results' / 'tables'
FIGURES = ROOT / 'output' / 'large_scale'
SCAL_DATA = ROOT / 'experiments' / 'chapter6' / 'results' / 'exp6_scalability.parquet'

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.unicode_minus'] = False

# Colors
BLUE, ORANGE, GREEN, BROWN = '#4A7FB5', '#E8A838', '#8B5E3C', '#8B5E3C'

def ensure_dir(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

print("=" * 60)
print("大规模实验复现脚本")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/6] 加载数据...")
df = pd.read_parquet(CANONICAL / 'exp_large_scale.parquet')
garh = df[df['mode']=='GA-RH'].set_index('berth_plan_no')
pure = df[df['mode']=='纯GA'].set_index('berth_plan_no')
sa = df[df['mode']=='SA'].set_index('berth_plan_no')
common = garh.index.intersection(pure.index)
print(f"  100船 × 3算法 = {len(df)} 条记录")
print(f"  配对对比: {len(common)} 个共同berth_plan")

# ============================================================
# 2. 表S1: 已有分箱统计
# ============================================================
print("\n[2/6] 表S1已存在 → 跳过")

# ============================================================
# 3. 表S2: 求解时间 vs 问题规模
# ============================================================
print("\n[3/6] 生成表S2 (时间分箱)...")
bins5 = [0, 600, 800, 1000, 2000, 99999]
labels5 = ['400-600', '600-800', '800-1k', '1k-2k', '2k-4k']
df['bin5'] = pd.cut(df['n_containers'], bins=bins5, labels=labels5, right=False)

rows_s2 = []
for b in labels5:
    sub = df[df['bin5'] == b]
    n = len(sub[sub['mode']=='GA-RH'])
    row = {'size_bin': b, 'n_ships': n}
    for m, m_lab in [('GA-RH','ga_rh'), ('纯GA','pure_ga'), ('SA','sa')]:
        s = sub[sub['mode']==m]['time_s']
        row[f'{m_lab}_time_min_mean'] = round(s.mean()/60, 2) if len(s)>0 else ''
        row[f'{m_lab}_time_min_std'] = round(s.std()/60, 2) if len(s)>0 else ''
        row[f'{m_lab}_fitness_mean'] = round(sub[sub['mode']==m]['fitness'].mean(), 4) if len(s)>0 else ''
        row[f'{m_lab}_fitness_std'] = round(sub[sub['mode']==m]['fitness'].std(), 4) if len(s)>0 else ''
    rows_s2.append(row)
pd.DataFrame(rows_s2).to_csv(ensure_dir(TABLES / 'table_S2_time_vs_scale.csv'), index=False)
print("  ✅ table_S2_time_vs_scale.csv")

# ============================================================
# 4. 表S3: 解质量 vs 问题规模
# ============================================================
print("[4/6] 生成表S3 (质量分箱)...")
rows_s3 = []
for b in labels5:
    sub = df[df['bin5'] == b]
    n = len(sub[sub['mode']=='GA-RH'])
    row = {'size_bin': b, 'n_ships': n}
    for m, m_lab in [('GA-RH','ga_rh'), ('纯GA','pure_ga')]:
        s = sub[sub['mode']==m]['fitness']
        row[f'{m_lab}_fitness'] = f'{s.mean():.4f}±{s.std():.4f}' if len(s)>0 else ''
        if m == '纯GA':
            gr = sub[sub['mode']=='GA-RH']['fitness'].mean()
            pg = sub[sub['mode']=='纯GA']['fitness'].mean()
            row['delta_ga_rh_vs_pure'] = f'{(gr-pg)*100:.2f}%'
    s_sa = sub[sub['mode']=='SA']['fitness']
    row['sa_fitness'] = f'{s_sa.mean():.4f}±{s_sa.std():.4f}' if len(s_sa)>0 else ''
    invalid = (s_sa < 0).sum()
    row['sa_invalid_rate'] = f'{invalid}/{n} ({invalid/n*100:.0f}%)'
    rows_s3.append(row)
pd.DataFrame(rows_s3).to_csv(ensure_dir(TABLES / 'table_S3_quality_vs_scale.csv'), index=False)
print("  ✅ table_S3_quality_vs_scale.csv")

# ============================================================
# 5. 表S4: 维度热力图数据
# ============================================================
print("[5/6] 生成表S4 (维度热力图)...")
bins6 = [400, 600, 800, 1000, 1500, 2000, 4000]
labels6 = ['400-600', '600-800', '800-1k', '1k-1.5k', '1.5k-2k', '2k-4k']
dim_cols = ['f1', 'f2', 'fitness', 'penalty']
dim_labels_cn = ['翻箱成本f1', '装卸效率f2', '解质量Fitness', '约束违反Penalty']
# lower is better for f1 and penalty
better_lower = [True, False, False, True]

df['bin6'] = pd.cut(df['n_containers'], bins=bins6, labels=labels6, right=False)

rows_s4 = []
for b in labels6:
    bpns = set(df[df['bin6']==b]['berth_plan_no'].unique())
    c = common.intersection(bpns)
    row = {'size_bin': b, 'n_ships': len(c)}
    for j, col in enumerate(dim_cols):
        g = garh.loc[c, col].mean()
        p = pure.loc[c, col].mean()
        d = g - p
        better = (d < 0) if better_lower[j] else (d > 0)
        row[f'{col}_GA-RH'] = f'{g:.4f}'
        row[f'{col}_纯GA'] = f'{p:.4f}'
        row[f'{col}_delta'] = f'{d:+.6f}'
        row[f'{col}_winner'] = 'GA-RH' if better else '纯GA'
    rows_s4.append(row)
pd.DataFrame(rows_s4).to_csv(ensure_dir(TABLES / 'table_S4_dimension_heatmap.csv'), index=False, encoding='utf-8-sig')
print("  ✅ table_S4_dimension_heatmap.csv")

# ============================================================
# 6. 表S5: 可扩展性数据
# ============================================================
print("[6/6] 生成表S5 (可扩展性)...")
if SCAL_DATA.exists():
    scal = pd.read_parquet(SCAL_DATA)
    s5 = scal[['vessel_code','n_containers','max_teu','n_pod','pop_size',
               'fitness','f1_rehandle','f2_efficiency','penalty','time_s']].copy()
    s5.columns = ['船型','箱数','TEU','目的港数','种群规模',
                  'Fitness','翻箱成本f1','装卸效率f2','约束违反','求解时间(s)']
    s5['TEU'] = s5['TEU'].astype(int)
    s5.to_csv(ensure_dir(TABLES / 'table_S5_scalability.csv'), index=False, encoding='utf-8-sig')
    print("  ✅ table_S5_scalability.csv")
else:
    print("  ⚠️ 可扩展性数据未找到，跳过")

# ============================================================
# 生成图4.2~4.6
# ============================================================
FIGURES.mkdir(parents=True, exist_ok=True)

# --- 图4.2: 求解时间 vs 规模 (柱状图+SA散点) ---
print("\n生成图4.2: 求解时间 vs 规模...")
x = np.arange(len(labels5))
w = 0.3
fig, ax = plt.subplots(figsize=(9, 5.5))
ga_means = [df[(df['bin5']==b)&(df['mode']=='GA-RH')]['time_s'].mean()/60 for b in labels5]
ga_stds = [df[(df['bin5']==b)&(df['mode']=='GA-RH')]['time_s'].std()/60 for b in labels5]
pu_means = [df[(df['bin5']==b)&(df['mode']=='纯GA')]['time_s'].mean()/60 for b in labels5]
pu_stds = [df[(df['bin5']==b)&(df['mode']=='纯GA')]['time_s'].std()/60 for b in labels5]
ax.bar(x-w/2, ga_means, w, yerr=ga_stds, capsize=4, color=BLUE, label='GA-RH', edgecolor='black', linewidth=0.5)
ax.bar(x+w/2, pu_means, w, yerr=pu_stds, capsize=4, color=ORANGE, label='纯GA', edgecolor='black', linewidth=0.5)
for i, b in enumerate(labels5):
    n = len(df[(df['bin5']==b)&(df['mode']=='GA-RH')])
    ax.text(i, max(ga_means[i], pu_means[i])+0.3, f'n={n}', ha='center', fontsize=9, fontweight='bold')
ax2 = ax.twinx()
sa_means_t = [df[(df['bin5']==b)&(df['mode']=='SA')]['time_s'].mean() for b in labels5]
sa_stds_t = [df[(df['bin5']==b)&(df['mode']=='SA')]['time_s'].std() for b in labels5]
ax2.errorbar(x, sa_means_t, yerr=sa_stds_t, fmt='D--', color=BROWN, markersize=6, capsize=3, linewidth=1.5, label='SA')
ax2.set_ylabel('SA求解时间 (s)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels5, fontsize=12)
ax.set_ylabel('求解时间 (min)', fontsize=12)
ax.set_xlabel('箱量范围', fontsize=12)
lines1, lbs1 = ax.get_legend_handles_labels()
lines2, lbs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbs1+lbs2, loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(True); ax.spines['top'].set_color('black'); ax.spines['top'].set_linewidth(1.0)
ax2.spines['top'].set_visible(True); ax2.spines['top'].set_color('black'); ax2.spines['top'].set_linewidth(1.0)
ax.spines['right'].set_visible(False); ax2.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig(FIGURES / 'fig4_2_time_vs_scale_updated.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ fig4_2")

# --- 图4.3: 解质量 vs 规模 ---
print("生成图4.3: 解质量 vs 规模...")
fig, ax = plt.subplots(figsize=(9, 5.5))
gr_f = [df[(df['bin5']==b)&(df['mode']=='GA-RH')]['fitness'].mean() for b in labels5]
gr_s = [df[(df['bin5']==b)&(df['mode']=='GA-RH')]['fitness'].std() for b in labels5]
pu_f = [df[(df['bin5']==b)&(df['mode']=='纯GA')]['fitness'].mean() for b in labels5]
pu_s = [df[(df['bin5']==b)&(df['mode']=='纯GA')]['fitness'].std() for b in labels5]
ax.bar(x-w/2, gr_f, w, yerr=gr_s, capsize=4, color=BLUE, label='GA-RH', edgecolor='black', linewidth=0.5)
ax.bar(x+w/2, pu_f, w, yerr=pu_s, capsize=4, color=ORANGE, label='纯GA', edgecolor='black', linewidth=0.5)
for i, b in enumerate(labels5):
    n = len(df[(df['bin5']==b)&(df['mode']=='GA-RH')])
    ax.text(i, max(gr_f[i], pu_f[i])+0.018, f'n={n}', ha='center', fontsize=9, fontweight='bold')
ax2 = ax.twinx()
sa_f = [df[(df['bin5']==b)&(df['mode']=='SA')]['fitness'].mean() for b in labels5]
sa_fs = [df[(df['bin5']==b)&(df['mode']=='SA')]['fitness'].std() for b in labels5]
ax2.errorbar(x, sa_f, yerr=sa_fs, fmt='D--', color=BROWN, markersize=6, capsize=3, linewidth=1.5, label='SA')
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.set_ylim(-1.2, 0.8); ax2.set_ylabel('SA Fitness', fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(labels5, fontsize=12)
ax.set_ylabel('Fitness', fontsize=12); ax.set_xlabel('箱量范围', fontsize=12)
ax.set_ylim(0.35, 1.05)
lines1, lbs1 = ax.get_legend_handles_labels()
lines2, lbs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbs1+lbs2, loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_axisbelow(True)
ax.spines['top'].set_visible(True); ax.spines['top'].set_color('black')
ax2.spines['top'].set_visible(True); ax2.spines['top'].set_color('black')
ax.spines['right'].set_visible(False); ax2.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig(FIGURES / 'fig4_3_quality_vs_scale_updated.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ fig4_3")

# --- 图4.4: GA-RH优势 vs 复杂度 ---
print("生成图4.4: 优势 vs 复杂度...")
complexity = garh.loc[common, 'n_containers'] * garh.loc[common, 'n_pod']
bins_c = [0, 1500, 3000, 5000, 8000, 12000, 99999]
labels_c = ['<=1.5k', '1.5k-3k', '3k-5k', '5k-8k', '8k-12k', '>12k']
bin_c = [750, 2250, 4000, 6500, 10000, 16000]
cp_bin = pd.cut(complexity, bins=bins_c, labels=labels_c, right=False)
delta_fit = garh.loc[common, 'fitness'] - pure.loc[common, 'fitness']
fig, ax1 = plt.subplots(figsize=(9, 5.5))
wins, ns = [], []
for b in labels_c:
    mask = cp_bin == b; n = mask.sum(); d = delta_fit[mask]
    wins.append((d>0).mean()*100); ns.append(n)
bars = ax1.bar(bin_c, wins, width=1100, color=BLUE, alpha=0.75, edgecolor='#333', linewidth=0.5, label='GA-RH胜率')
for bar, wr, n in zip(bars, wins, ns):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.8, f'{wr:.0f}%\nn={n}', ha='center', fontsize=9, fontweight='bold')
ax1.set_xlabel('问题复杂度'); ax1.set_ylabel('胜率 (%)', color=BLUE)
ax1.set_xticks(bin_c); ax1.set_xticklabels(labels_c, fontsize=12); ax1.set_ylim(0, 105); ax1.set_xlim(0, 18500)
ax2 = ax1.twinx()
advs = []
for b in labels_c:
    mask = cp_bin == b; d = delta_fit[mask]
    advs.append(d.mean()*100)
ax2.plot(bin_c, advs, color=ORANGE, linewidth=2.5, marker='s', markersize=8, label='平均优势(x1e-2)')
ax2.axhline(y=0, color='#888', linestyle=':', linewidth=0.8, alpha=0.6)
ax2.set_ylabel('平均适应度优势 (x1e-2)', color=ORANGE)
ax1.grid(axis='y', alpha=0.3, linestyle='--'); ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(True); ax2.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(False); ax2.spines['right'].set_visible(True)
lines1, lbs1 = ax1.get_legend_handles_labels()
lines2, lbs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lbs2, lbs1+lbs2, loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES / 'fig4_4_advantage_vs_complexity_updated.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ fig4_4")

# --- 图4.5: 热力图 ---
print("生成图4.5: 维度热力图...")
df['bin6'] = pd.cut(df['n_containers'], bins=bins6, labels=labels6, right=False)
diff_matrix = np.zeros((len(labels6), len(dim_cols)))
for i, b in enumerate(labels6):
    bpns = set(df[df['bin6']==b]['berth_plan_no'].unique())
    c = common.intersection(bpns)
    for j, col in enumerate(dim_cols):
        diff_matrix[i, j] = garh.loc[c, col].mean() - pure.loc[c, col].mean()

fig, ax = plt.subplots(figsize=(10, 5))
vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
im = ax.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
ax.set_xticks(range(len(dim_cols)))
ax.set_xticklabels(['翻箱成本 f1', '装卸效率 f2', '解质量 Fitness', '约束违反 Penalty'], fontsize=11)
ax.set_yticks(range(len(labels6)))
ax.set_yticklabels(labels6, fontsize=11)
ax.set_xlabel('评价维度', fontsize=12)
ax.set_ylabel('问题规模', fontsize=12)
for i in range(len(labels6)):
    for j in range(len(dim_cols)):
        v = diff_matrix[i, j]
        better = (v < 0) if better_lower[j] else (v > 0)
        mk = '~' if abs(v) < 0.0001 else ('▼' if better else '▲')
        tc = 'white' if abs(v) > vmax*0.5 else 'black'
        ax.text(j, i, f'{v:+.4f}\n{mk}', ha='center', va='center', fontsize=9, color=tc, fontweight='bold')
plt.colorbar(im, ax=ax, label='Delta', shrink=0.8)
ax.spines['top'].set_visible(True)
plt.tight_layout()
plt.savefig(FIGURES / 'fig4_5_heatmap_updated.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ fig4_5")

# --- 图4.6: 可扩展性 (对数) ---
print("生成图4.6: 可扩展性分析...")
bins6 = [400, 600, 800, 1000, 1500, 2000, 4000]
labels6 = ['400-600', '600-800', '800-1k', '1k-1.5k', '1.5k-2k', '2k-4k']
df['bin6'] = pd.cut(df['n_containers'], bins=bins6, labels=labels6, right=False)
x_idx = np.arange(len(labels6))
colors = {'GA-RH': BLUE, '纯GA': ORANGE, 'SA': GREEN}
markers = {'GA-RH': 'o', '纯GA': 's', 'SA': 'D'}
fig, ax = plt.subplots(figsize=(12, 5.5))
for mode in ['GA-RH', '纯GA', 'SA']:
    sub = df[df['mode'] == mode]
    for b_idx, b in enumerate(labels6):
        vals = sub[sub['bin6'] == b]['time_s'].values
        jitter = np.random.uniform(-0.2, 0.2, size=len(vals))
        ax.scatter([x_idx[b_idx]] * len(vals) + jitter, vals, color=colors[mode], alpha=0.12, s=15, zorder=1)
    means = [sub[sub['bin6']==b]['time_s'].mean() for b in labels6]
    ax.plot(x_idx, means, color=colors[mode], linewidth=2.5, marker=markers[mode], markersize=9, label=mode, zorder=3)
    for i, b in enumerate(labels6):
        n = len(sub[sub['bin6']==b])
        if n > 0:
            ax.annotate(f'n={n}', (x_idx[i], means[i]), textcoords='offset points', xytext=(0, -20), fontsize=7, color=colors[mode], ha='center', alpha=0.6)
    x_raw = sub['n_containers'].values; y_raw = sub['time_s'].values
    A = np.vstack([np.log(x_raw), np.ones_like(x_raw)]).T
    coeffs, *_ = np.linalg.lstsq(A, np.log(y_raw), rcond=None)
    b_exp, ln_a = coeffs; a = np.exp(ln_a)
    fl_x = np.linspace(400, 4000, 100)
    fl_y = a * fl_x ** b_exp
    fl_idx = np.interp(fl_x, [400, 4000], [0, 5])
    ax.plot(fl_idx, fl_y, color=colors[mode], linewidth=1.2, linestyle='--', alpha=0.4, zorder=1)
    y_at_mid = a * (1250) ** b_exp
    off = {'GA-RH': 2.5, '纯GA': 1.8, 'SA': 2.0}[mode]
    ax.text(3.0, y_at_mid*off, f't ∝ N^{b_exp:.2f}', fontsize=9, color=colors[mode], alpha=0.7, style='italic', ha='center')
ax.set_yscale('log')
ax.set_xlabel('问题规模 (箱数)', fontsize=13); ax.set_ylabel('求解时间 (秒, 对数坐标)', fontsize=13)
ax.set_xticks(x_idx); ax.set_xticklabels(labels6, fontsize=11)
ax.legend(fontsize=11, loc='upper left'); ax.grid(True, alpha=0.3, linestyle='--', which='both')
ax.set_xlim(-0.6, 5.6); ax.set_ylim(0.5, 5000)
ax.spines['top'].set_visible(True)
ax.annotate('SA: 快速但不可行', xy=(4.5, 4), fontsize=9, color=GREEN, style='italic', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREEN, alpha=0.8))
plt.tight_layout()
plt.savefig(FIGURES / 'fig4_6_scalability_updated.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ fig4_6")

# 打印拟合系数
print("\n幂律拟合系数:")
for mode in ['GA-RH', '纯GA']:
    sub = df[df['mode']==mode]
    xr = sub['n_containers'].values; yr = sub['time_s'].values
    A = np.vstack([np.log(xr), np.ones_like(xr)]).T
    coeffs, *_ = np.linalg.lstsq(A, np.log(yr), rcond=None)
    b, ln_a = coeffs
    t2k = np.exp(ln_a) * 2000**b
    print(f"  {mode}: a={np.exp(ln_a):.2e}, b={b:.3f}, t(2000箱)={t2k:.0f}s={t2k/60:.1f}min")

print("\n" + "=" * 60)
print("全部完成 ✅")
print("=" * 60)
