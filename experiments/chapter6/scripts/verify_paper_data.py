"""
第六章论文数据验证 — 逐表逐格核对脚本
从规范化parquet文件读取全部数据，输出论文所有表格的精确值
"""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # project root
SIM = ROOT / 'output' / 'simulation_results'
GR = ROOT / 'output' / 'ga_rh_results'

# 规范化数据文件（已冻结）
CANONICAL = {
    '常规作业': SIM / 'sim_常规作业_cfgall_d30r3.parquet',
    '高峰压力': SIM / 'sim_高峰压力_cfgall_d30r10.parquet',
    '异常情况': SIM / 'sim_异常情况_cfgall_d30r10.parquet',
}

def load(scenario):
    return pd.read_parquet(CANONICAL[scenario])

def hdr(text):
    print(f'\n{"="*60}\n{text}\n{"="*60}')

# ══════════════════════
# 表6.1-6.3
# ══════════════════════
hdr('表6.1-6.3 三场景MAS-DES实验数据')

scenarios = [
    ('常规作业', '表6.1 常规作业（30天*3轮）'),
    ('高峰压力', '表6.2 高峰压力（30天*10轮）'),
    ('异常情况', '表6.3 异常情况（30天*10轮）'),
]

for sc_name, sc_label in scenarios:
    df = load(sc_name)
    print(f'\n{sc_label}')
    print('  cfg   船时(h)  翻箱(%)  设备(%)  船时改善  翻箱改善  设备改善  船数')
    print('  ' + '-'*60)
    for cfg in ['A','B','C','D']:
        sub = df[df['config']==cfg]
        t = sub['turnaround_h'].mean()
        r = sub['reshuffle_pct'].mean()
        e = sub['equip_util_pct'].mean()
        imp_t = sub['turnaround_improvement_pct'].mean()
        imp_r = sub['reshuffle_improvement_pct'].mean()
        imp_e = sub['equip_util_improvement_pct'].mean()
        nv = sub['n_vessels'].mean()
        if cfg == 'A':
            print(f'  {cfg:<4} {t:>8.1f} {r:>8.1f} {e:>8.1f}     ---      ---      --- {nv:>6.0f}')
        else:
            print(f'  {cfg:<4} {t:>8.1f} {r:>8.1f} {e:>8.1f}     {imp_t:>5.1f}%  {imp_r:>5.1f}%  {imp_e:>5.1f}% {nv:>6.0f}')

# ══════════════════════
# 表6.4 统计检验
# ══════════════════════
hdr('表6.4 统计显著性检验')

for sc_name, sc_label in [('常规作业','常规'), ('高峰压力','高峰'), ('异常情况','异常')]:
    df = load(sc_name)
    a = df[df['config']=='A']['turnaround_h']
    d = df[df['config']=='D']['turnaround_h']
    t_stat, p_val = stats.ttest_ind(d, a, alternative='less')
    n1, n2 = len(a), len(d)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*d.var(ddof=1)) / (n1+n2-2))
    cd = (a.mean() - d.mean()) / sp
    imp = (a.mean()-d.mean())/a.mean()*100
    print(f'  {sc_label}: A={a.mean():.1f}h  D={d.mean():.1f}h  '
          f'd={imp:.1f}%  t={t_stat:.1f}  p={p_val:.6f}  d={cd:.2f}')

# ══════════════════════
# 表6.5 消融实验
# ══════════════════════
hdr('表6.5 消融实验（复用表6.1常规作业数据）')

df = load('常规作业')
print('  配置        船时(h)  翻箱(%)  设备(%)    vs A改善')
print('  ' + '-'*55)
for cfg in ['A','B','C','D']:
    sub = df[df['config']==cfg]
    t = sub['turnaround_h'].mean()
    r = sub['reshuffle_pct'].mean()
    e = sub['equip_util_pct'].mean()
    name = {'A':'A:纯配载','B':'B:纯堆场','C':'C:联合优化','D':'D:完整协同'}[cfg]
    if cfg == 'A':
        print(f'  {name:<10} {t:>8.1f} {r:>8.1f} {e:>8.1f}     ---')
    else:
        imp_t = sub['turnaround_improvement_pct'].mean()
        imp_r = sub['reshuffle_improvement_pct'].mean()
        imp_e = sub['equip_util_improvement_pct'].mean()
        s = f'd{imp_t:.1f}% / d{imp_r:.1f}% / u{imp_e:.1f}%'
        print(f'  {name:<10} {t:>8.1f} {r:>8.1f} {e:>8.1f}  {s:>20}')
print('  注: C与D一致')

# ══════════════════════
# 可扩展性
# ══════════════════════
hdr('可扩展性测试数据')
scal = pd.read_parquet(GR / 'exp6_scalability.parquet')
print('  船舶     箱数    时间(s)')
for _, r in scal.iterrows():
    print(f'  {r["vessel_code"]:<8} {r["n_containers"]:>6} {r["time_s"]:>8.1f}')

# ══════════════════════
# 输出CSV
# ══════════════════════
out_dir = ROOT / 'experiments' / 'chapter6' / 'output' / 'verified'
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for sc_name in ['常规作业', '高峰压力', '异常情况']:
    df = load(sc_name)
    for cfg in ['A','B','C','D']:
        sub = df[df['config']==cfg]
        rows.append({
            '场景': sc_name, '配置': cfg,
            '船时_h': round(sub['turnaround_h'].mean(), 1),
            '翻箱_pct': round(sub['reshuffle_pct'].mean(), 1),
            '设备_pct': round(sub['equip_util_pct'].mean(), 1),
            '船时改善_pct': round(sub['turnaround_improvement_pct'].mean(), 1),
            '翻箱改善_pct': round(sub['reshuffle_improvement_pct'].mean(), 1),
            '设备改善_pct': round(sub['equip_util_improvement_pct'].mean(), 1),
            '平均船数': round(sub['n_vessels'].mean(), 0),
            '运行轮次': len(sub),
        })

pd.DataFrame(rows).to_csv(out_dir / 'paper_tables_6_1_to_6_3.csv', index=False)
print(f'\nPaper tables saved: {out_dir / "paper_tables_6_1_to_6_3.csv"}')
scal.to_csv(out_dir / 'scalability_data.csv', index=False)
print(f'Scalability saved: {out_dir / "scalability_data.csv"}')
print('\nDone. All data traceable to canonical parquet files.')
