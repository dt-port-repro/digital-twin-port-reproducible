#!/usr/bin/env python3
"""生成第六章DES仿真三场景配套表 S14/S15/S16

数据来源: experiments/chapter6/results/exp6_sim_*.parquet

配套表:
  S14: 常规作业场景仿真结果
  S15: 高峰压力场景仿真结果
  S16: 异常情况场景仿真结果

用法: python gen_des_scenario_tables.py
"""
import pandas as pd, os, csv, shutil

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE_DIR = os.path.join(PKG, '03_results/tables')
DATA_DIR = os.path.join(PKG, '03_results/canonical')
SRC = os.path.join(PKG, 'experiments', 'chapter6', 'results')
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. 提取三场景数据 ──────────────────────────
scenarios = {
    '常规作业': 'exp6_sim_常规作业.parquet',
    '高峰压力': 'exp6_sim_高峰压力.parquet',
    '异常情况': 'exp6_sim_异常情况.parquet',
}

scenario_data = {}
for sname, sfile in scenarios.items():
    df = pd.read_parquet(os.path.join(SRC, sfile))
    avg = df.mean(numeric_only=True)
    scenario_data[sname] = avg
    print(f'  {sname}: {len(df)} runs')

# ── 指标映射 ──────────────────────────────────
# parquet列名 -> 显示名
METRICS = [
    ('turnaround_h', '船舶在港时间(h)', 'turnaround_improvement_pct', True),   # lower=better
    ('reshuffle_pct', '翻箱率(%)', 'reshuffle_improvement_pct', True),          # lower=better
    ('equip_util_pct', '设备利用率(%)', 'equip_util_improvement_pct', False),   # higher=better
]

# ── 2. 生成三场景表 S14/S15/S16 ──────────────
for table_idx, (sname, data) in enumerate(scenario_data.items()):
    table_id = f'S{14 + table_idx}'
    rows = []
    for base_key, cn, imp_key, lower_better in METRICS:
        base = data[f'baseline_{base_key}']
        opt = data[f'optimized_{base_key}']
        imp = data[imp_key]
        sign = '↓' if (imp > 0 and lower_better) or (imp < 0 and not lower_better) else '↑'
        rows.append({
            '指标': cn,
            '独立优化(基线)': f'{base:.1f}',
            '协同优化': f'{opt:.1f}',
            '改善幅度': f'{sign}{abs(imp):.1f}%'
        })
    out = os.path.join(TABLE_DIR, f'table_{table_id}_des_{sname}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['指标', '独立优化(基线)', '协同优化', '改善幅度'])
        w.writeheader()
        w.writerows(rows)
    print(f'  ✅ {table_id} saved ({len(rows)} rows)')

# ── 3. 三场景汇总表 ──────────────────────────
summary_rows = []
for sname, data in scenario_data.items():
    for base_key, cn, imp_key, lower_better in METRICS:
        base = data[f'baseline_{base_key}']
        opt = data[f'optimized_{base_key}']
        imp = data[imp_key]
        sign = '↓' if (imp > 0 and lower_better) or (imp < 0 and not lower_better) else '↑'
        summary_rows.append({
            '场景': sname,
            '指标': cn,
            '独立优化': f'{base:.1f}',
            '协同优化': f'{opt:.1f}',
            '改善幅度': f'{sign}{abs(imp):.1f}%'
        })
out_sum = os.path.join(TABLE_DIR, 'table_des_all_scenarios.csv')
with open(out_sum, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['场景', '指标', '独立优化', '协同优化', '改善幅度'])
    w.writeheader()
    w.writerows(summary_rows)
print(f'  ✅ 三场景汇总 saved ({len(summary_rows)} rows)')

# ── 4. 复制源parquet到复现包 ─────────────────
for sname, sfile in scenarios.items():
    src = os.path.join(SRC, sfile)
    dst = os.path.join(DATA_DIR, sfile)
    shutil.copy2(src, dst)
    print(f'  ✅ source: {sfile} ({os.path.getsize(dst)//1024}KB)')

print('\n✅ 全部完成')
