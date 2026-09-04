"""
补充脚本：读取MAS-DES真实输出，生成与下游兼容的实验结果文件
论文§6.3.2/§6.3.4

替换原有的公式估算数据为真实MAS-DES仿真输出
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # → project root
GR = ROOT / 'output' / 'ga_rh_results'
GR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════
# 1. 读取MAS-DES真实输出
# ════════════════════════════════════════════

masdes_dir = ROOT / 'output' / 'simulation_results'

# 三场景×四配置×多轮数据
scenario_map = {'常规作业':'常规作业', '高峰压力':'高峰压力', '异常情况':'异常情况'}
scenario_rounds = {'常规作业': 3, '高峰压力': 10, '异常情况': 10}

all_sim = []
for sc_name, sc_display in scenario_map.items():
    n_runs = scenario_rounds[sc_name]
    f = masdes_dir / f'sim_{sc_name}_cfgall_d30r{n_runs}.parquet'
    if not f.exists():
        print(f'  ⚠️ 未找到 {f}')
        continue
    df = pd.read_parquet(f)
    for cfg in ['A','B','C','D']:
        sub = df[df['config']==cfg]
        if len(sub)==0: continue
        all_sim.append({
            'scenario': sc_display,
            'config': cfg,
            'n_vessels': sub['n_vessels'].mean(),
            'turnaround_h': round(sub['turnaround_h'].mean(), 1),
            'reshuffle_pct': round(sub['reshuffle_pct'].mean(), 1),
            'equip_util_pct': round(sub['equip_util_pct'].mean(), 1),
            'n_runs': n_runs,
        })
        # 改善百分比
        if cfg == 'A':
            base_t = sub['turnaround_h'].mean()
            base_r = sub['reshuffle_pct'].mean()
            base_e = sub['equip_util_pct'].mean()
        elif cfg in ['B','C','D']:
            imp_t = (base_t - sub['turnaround_h'].mean()) / base_t * 100
            imp_r = (base_r - sub['reshuffle_pct'].mean()) / base_r * 100
            imp_e = (sub['equip_util_pct'].mean() - base_e) / base_e * 100
            all_sim[-1]['imp_t_pct'] = round(imp_t, 1)
            all_sim[-1]['imp_r_pct'] = round(imp_r, 1)
            all_sim[-1]['imp_e_pct'] = round(imp_e, 1)

df_sim = pd.DataFrame(all_sim)
df_sim.to_parquet(GR / 'masdes_simulation.parquet', index=False)
print(f'✅ MAS-DES仿真数据保存: {len(df_sim)}条')

# 打印验证
print('\n=== MAS-DES 真实输出 ===')
for _, r in df_sim.iterrows():
    suffix = ''
    if r['config'] in ['B','C','D']:
        suffix = f"  (↓{r['imp_t_pct']}% 船时 / ↓{r['imp_r_pct']}% 翻箱 / ↑{r['imp_e_pct']}% 设备)"
    print(f"  {r['scenario']} cfg={r['config']}: {r['turnaround_h']}h 翻箱{r['reshuffle_pct']}% 设备{r['equip_util_pct']}%{suffix}")

# ════════════════════════════════════════════
# 2. 生成消融实验数据（C=D）
# ════════════════════════════════════════════

# 使用常规作业MAS-DES数据构造消融表（4配置）
regular = df_sim[(df_sim['scenario']=='常规作业')]
abl_rows = []
for _, r in regular.iterrows():
    abl_rows.append({
        'config': f"{r['config']}: " + {
            'A':'纯配载', 'B':'纯堆场', 'C':'联合优化', 'D':'完整协同'
        }[r['config']],
        'turnaround_h': r['turnaround_h'],
        'reshuffle_pct': r['reshuffle_pct'],
        'equip_util_pct': r['equip_util_pct'],
        'imp_t': f"↓{r['imp_t_pct']}%" if r['config']!='A' else '—',
        'imp_r': f"↓{r['imp_r_pct']}%" if r['config']!='A' else '—',
        'imp_e': f"↑{r['imp_e_pct']}%" if r['config']!='A' else '—',
    })

# 添加C=D注释行
abl_rows.append({
    'config': '注',
    'turnaround_h': 'C与D一致（C=D）',
    'reshuffle_pct': '双向协议在当前参数化场景中',
    'equip_util_pct': '未产生额外增益',
    'imp_t': '',
    'imp_r': '',
    'imp_e': '',
})

df_abl = pd.DataFrame(abl_rows)
df_abl.to_csv(GR / 'masdes_ablation.csv', index=False)
print(f'\n✅ 消融实验数据保存: {len(df_abl)}条 (CSV)')

# ════════════════════════════════════════════
# 3. 打印论文用表
# ════════════════════════════════════════════
print('\n\n=== 表6.1 MAS-DES三场景实验数据 ===')
for sc in ['常规作业','高峰压力','异常情况']:
    print(f'\n{sc}:')
    print(f"{'配置':>6} | {'船时(h)':>8} {'翻箱(%)':>8} {'设备(%)':>8} {'船时改善':>8} {'翻箱改善':>8} {'设备改善':>8}")
    print('-'*60)
    sub = df_sim[df_sim['scenario']==sc]
    for _, r in sub.iterrows():
        imp_s = ''
        if r['config'] in ['B','C','D']:
            imp_s = f"{'↓'+str(r['imp_t_pct'])+'%':>8} {'↓'+str(r['imp_r_pct'])+'%':>8} {'↑'+str(r['imp_e_pct'])+'%':>8}"
        else:
            imp_s = f"{'—':>8} {'—':>8} {'—':>8}"
        print(f"{r['config']:>6} | {r['turnaround_h']:>8.1f} {r['reshuffle_pct']:>8.1f} {r['equip_util_pct']:>8.1f} {imp_s}")

print('\n\n=== 表6.4 消融实验 ===')
print(f"{'配置':>10} | {'船时(h)':>8} {'翻箱(%)':>8} {'设备(%)':>8} {'vs A改善':>20}")
print('-'*55)
for _, r in df_abl.iterrows():
    print(f"{r['config']:>10} | {str(r['turnaround_h']):>8} {str(r['reshuffle_pct']):>8} {str(r['equip_util_pct']):>8} {str(r['imp_t']+' / '+r['imp_r']+' / '+r['imp_e']):>20}")
