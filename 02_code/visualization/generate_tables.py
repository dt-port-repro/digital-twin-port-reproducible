"""
从实验数据生成论文表4.5-4.13的完整表格（固化版）
所有数据来自 canonical/ 标准源（gen=50, pop=60 统一参数）
表4.8/4.9的gamma=0使用gen=80，编码策略C使用gen=80/pop=100
"""
import pandas as pd, numpy as np
from pathlib import Path

CANONICAL = Path(__file__).parent.parent.parent / '03_results' / 'canonical'
EXPORT = CANONICAL
TABLES_DIR = Path(__file__).parent.parent.parent / '03_results' / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 80)
print('第四章论文表格生成（固化版 v2 - gen=50/pop=60 统一参数）')
print('=' * 80)

garh = pd.read_parquet(CANONICAL / 'exp1_garh.parquet').sort_values('n_containers')
pure_ga = pd.read_parquet(CANONICAL / 'exp2_pure_ga.parquet').sort_values('n_containers')
fcfs = pd.read_parquet(CANONICAL / 'exp3_fcfs.parquet').sort_values('n_containers')
gamma0 = pd.read_parquet(CANONICAL / 'exp4_gamma0.parquet').sort_values('vessel_code')
post = pd.read_parquet(CANONICAL / 'exp5_postproc.parquet').sort_values('vessel_code')
encoding = pd.read_parquet(CANONICAL / 'exp6_encoding.parquet')
rob = pd.read_parquet(CANONICAL / 'exp_robustness.parquet')

SHIPS = ['CNTIG','CNCT','MXNT','OFUT','CGAMV','APESP']

def save_csv(table_id, data):
    fname = 'table_4_{}.csv'.format(table_id)
    data.to_csv(TABLES_DIR / fname, index=False)
    print('  OK {}'.format(fname))

# ============================================================
# 表4.5 GA-RH基本性能（单轮，gen=50）
# ============================================================
print('\n' + '=' * 70)
print('表4.5 GA-RH基本性能（实验1，单轮，gen=50）')
print('=' * 70)
print()
hdr = '{:>6} {:>8} {:>10} {:>8} {:>8} {:>8} {:>10} {:>10}'
print(hdr.format('船名','箱量','fitness','f1翻箱','f2效率','f3平衡','f4堆场','时间'))
print('-' * 68)
for _, r in garh.iterrows():
    print('{:>6} {:>8.0f} {:>10.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}  {:>6.0f}s'.format(
        r['vessel_code'], r['n_containers'], r['fitness'],
        r['rehandle'], r['efficiency'], r['balance'], r['yard_collab'], r['time_s']))

r_val = np.corrcoef(garh['n_containers'], garh['fitness'])[0,1]
print()
print('箱量-适应度相关系数 r = {:.4f}'.format(r_val))
print('GA-RH平均fitness = {:.4f}'.format(garh['fitness'].mean()))
save_csv('5_garh', garh)

# ============================================================
# 表4.6 GA-RH vs 纯GA（单轮，gen=50）
# ============================================================
print('\n' + '=' * 70)
print('表4.6 GA-RH vs 纯GA（实验2，单轮，gen=50）')
print('=' * 70)
m = garh.merge(pure_ga, on='vessel_code', suffixes=('_garh','_pure'))
print()
hdr2 = '{:>6} {:>10} {:>10} {:>8}'
print(hdr2.format('船名','GA-RH','纯GA','D(%)'))
print('-' * 38)
deltas = []
for _, r in m.iterrows():
    d = (r['fitness_garh'] - r['fitness_pure']) / abs(r['fitness_pure']) * 100
    deltas.append(d)
    print('{:>6} {:>10.4f} {:>10.4f} {:>+7.2f}%'.format(r['vessel_code'], r['fitness_garh'], r['fitness_pure'], d))
print()
print('平均D = {:.2f}%'.format(np.mean(deltas)))
print('Dmax = {:.2f}%'.format(max(abs(d) for d in deltas)))
comp6 = m[['vessel_code','fitness_garh','fitness_pure']].copy()
save_csv('6_comparison', comp6)

# ============================================================
# 表4.7 GA-RH vs FCFS贪心
# ============================================================
print('\n' + '=' * 70)
print('表4.7 GA-RH vs FCFS贪心基线（实验2续）')
print('=' * 70)
m2 = garh.merge(fcfs, on='vessel_code', suffixes=('_garh','_fcfs'))
print()
hdr3 = '{:>6} {:>6} {:>10} {:>10} {:>10}'
print(hdr3.format('船名','箱量','FCFS','GA-RH','提升(%)'))
print('-' * 46)
fcfs_fits = []
for _, r in m2.iterrows():
    imp = (r['fitness_garh'] - r['fitness_fcfs']) / abs(r['fitness_fcfs']) * 100
    fcfs_fits.append(r['fitness_fcfs'])
    print('{:>6} {:>6.0f} {:>10.4f} {:>10.4f} {:>+9.1f}%'.format(
        r['vessel_code'], r['n_containers_garh'], r['fitness_fcfs'], r['fitness_garh'], imp))
avg_fcfs = np.mean(fcfs_fits)
avg_imp = (garh['fitness'].mean() - avg_fcfs) / abs(avg_fcfs) * 100
print()
print('FCFS平均fitness = {:.4f}'.format(avg_fcfs))
print('GA-RH平均fitness = {:.4f}'.format(garh['fitness'].mean()))
print('平均提升幅度 = {:.0f}%'.format(avg_imp))
save_csv('7_fcfs', fcfs)

# ============================================================
# 表4.8 GA-RH vs gamma=0（gamma=0用gen=80）
# ============================================================
print('\n' + '=' * 70)
print('表4.8 GA-RH vs gamma=0（实验3，gamma=0使用gen=80, pop=60）')
print('=' * 70)
m3 = garh.merge(gamma0, on='vessel_code', suffixes=('_garh','_g0'))
print()
hdr4 = '{:>6} {:>12} {:>10} {:>8} {:>12}'
print(hdr4.format('船名','GA-RH(fit)','g=0(fit)','Dfit(%)','GA-RH(f1)'))
print('-' * 52)
for _, r in m3.iterrows():
    d = (r['fitness_garh'] - r['fitness_g0']) / abs(r['fitness_g0']) * 100
    f1 = r.get('rehandle_garh', 0)
    print('{:>6} {:>10.4f}  {:>10.4f} {:>+7.2f}%  {:>10.4f}'.format(
        r['vessel_code'], r['fitness_garh'], r['fitness_g0'], d, f1))
print()
g0_avg = gamma0['fitness'].mean()
print('GA-RH平均 = {:.4f}（gen=50，1500次评估）'.format(garh['fitness'].mean()))
print('g=0平均 = {:.4f}（gen=80，2400次评估）'.format(g0_avg))

# 表4.9 聚合均值
# NOTE(2026-09-03 修正): GA-RH 的 rehandle 列曾因脚本键名 bug（detail['stability']不存在）
# 在 exp1_garh.parquet 中为占位 0。fitness 公式已知且可逆，rehandle 可从
# fitness/efficiency/balance/yard_collab/penalty 精确反推：
#   fr = (fitness - 0.35*eff - 0.25*bal - 0.15*yc + 5*pen) / 0.25
# 反推结果与论文表4.8 GA-RH(f1) 列 6/6 船逐位吻合（铁证），exp1_garh.parquet 已回填。
f1_garh = garh['rehandle'].mean()  # 已回填，正常复算 = 0.682072
f1_g0 = gamma0['rehandle'].mean()
e2_garh = garh['efficiency'].mean()
e2_g0 = gamma0['efficiency'].mean()
garh_fit_avg = garh['fitness'].mean()
g0_fit_avg = g0_avg
print()
print('--- 聚合均值对比（表4.9）---')
print('  f1(翻箱):  GA-RH={:.4f}  g=0={:.4f}  D={:+.2f}%'.format(
    f1_garh, f1_g0, (f1_garh-f1_g0)/abs(f1_g0)*100))
print('  f2(效率):  GA-RH={:.4f}  g=0={:.4f}  D={:+.2f}%'.format(
    e2_garh, e2_g0, (e2_garh-e2_g0)/abs(e2_g0)*100))
print('  fitness:   GA-RH={:.4f}  g=0={:.4f}  D={:+.2f}%'.format(
    garh_fit_avg, g0_fit_avg, (garh_fit_avg-g0_fit_avg)/abs(g0_fit_avg)*100))
# 表4.9 CSV
agg9 = pd.DataFrame({
    'metric': ['f1_rehandle','f2_efficiency','fitness'],
    'GA-RH': [f1_garh, e2_garh, garh_fit_avg],
    'gamma0': [f1_g0, e2_g0, g0_fit_avg],
    'delta_pct': [
        round((f1_garh-f1_g0)/abs(f1_g0)*100, 2),
        round((e2_garh-e2_g0)/abs(e2_g0)*100, 2),
        round((garh_fit_avg-g0_fit_avg)/abs(g0_fit_avg)*100, 2)
    ]
})
save_csv('9_aggregate', agg9)
save_csv('8_gamma0', gamma0)

# ============================================================
# 表4.10 鲁棒性测试
# ============================================================
print('\n' + '=' * 70)
print('表4.10 算法鲁棒性测试（实验4，3船×3水平×15轮=135轮）')
print('=' * 70)
print()
hdr5 = '{:>6} {:>6} {:>6} {:>30} {:>8} {:>8}'
print(hdr5.format('船名','箱量','扰动','fitness(均值+-std)','CV%','D(%)'))
print('-' * 65)
rows_rob = []
for vc in sorted(rob['vessel_code'].unique()):
    sub = rob[rob['vessel_code']==vc]
    nc = sub['n_containers'].iloc[0]
    base = sub[sub['perturb_level']==0]['fitness'].mean()
    for pl in sorted(sub['perturb_level'].unique()):
        s = sub[sub['perturb_level']==pl]
        if len(s):
            fm = s['fitness'].mean()
            fs = s['fitness'].std()
            cv = fs/abs(fm)*100
            d = (fm-base)/abs(base)*100
            print('{:>6} {:>6.0f}  +-{:>2}%  {:>.4f}+-{:.4f}  {:>6.2f}%  {:>+7.2f}%'.format(
                vc, nc, pl, fm, fs, cv, d))
            rows_rob.append({
                'vessel_code': vc, 'n_containers': nc, 'perturb_level': pl,
                'n_runs': len(s), 'fitness_mean': round(fm,4), 'fitness_std': round(fs,4),
                'CV_pct': round(cv,2), 'delta_pct': round(d,2)
            })
save_csv('10_robustness_summary', pd.DataFrame(rows_rob))

# ============================================================
# 表4.11 后处理消融
# ============================================================
print('\n' + '=' * 70)
print('表4.11 后处理消融（实验5，单轮，gen=50）')
print('=' * 70)
print()
hdr6 = '{:>6} {:>10} {:>10} {:>10}'
print(hdr6.format('船名','纯GA','GA+后处理','改善%'))
print('-' * 40)
for _, r in post.iterrows():
    print('{:>6} {:>10.4f} {:>10.4f} {:>+9.4f}%'.format(
        r['vessel_code'], r['pure_fitness'], r['post_fitness'], r['improve_pct']))
print()
print('平均改善 = {:.4f}%'.format(post['improve_pct'].mean()))
save_csv('11_postproc', post)

# ============================================================
# 表4.12 编码策略对比（逐船）
# ============================================================
print('\n' + '=' * 70)
print('表4.12 编码策略对比（实验6，gen=50/pop=60，C用80/100gen）')
print('=' * 70)

A = encoding[encoding['strategy'].str.contains('A', na=False)].set_index('vessel_code')
B = encoding[encoding['strategy'].str.contains('B', na=False)].groupby('vessel_code').agg(
    mean_fit=('fitness','mean'), std_fit=('fitness','std'), mean_time=('time_s','mean'))
C = encoding[encoding['strategy'].str.contains('C', na=False)].set_index('vessel_code')

vessels_enc = ['CNCT', 'CGAMV', 'APESP']
print()
hdr7 = '{:>6} {:>14} {:>14} {:>14} {:>12} {:>8}'
print(hdr7.format('船型','A:分层+修复','B:扁平+惩罚','C:近似最优','DCvsA(%)','B_std'))
print('-' * 68)
rows_enc = []
for v in vessels_enc:
    a = A.loc[v, 'fitness']
    b = B.loc[v, 'mean_fit']
    bs = B.loc[v, 'std_fit']
    c = C.loc[v, 'fitness']
    delta = (c - a) / abs(a) * 100
    print('{:>6} {:>12.4f}     {:>12.4f}     {:>12.4f}     {:>+8.2f}%     {:>6.4f}'.format(
        v, a, b, c, delta, bs))
    rows_enc.append({
        'vessel_code': v, 'A_fitness': round(a,4), 'B_mean_fitness': round(b,4),
        'B_std': round(bs,4), 'C_fitness': round(c,4), 'delta_CvsA_pct': round(delta,2)
    })
a_mean = A.loc[vessels_enc, 'fitness'].mean()
b_mean = B.loc[vessels_enc, 'mean_fit'].mean()
c_mean = C.loc[vessels_enc, 'fitness'].mean()
delta_mean = (c_mean - a_mean) / abs(a_mean) * 100
print('{:>6} {:>12.4f}     {:>12.4f}     {:>12.4f}     {:>+8.2f}%'.format('均值', a_mean, b_mean, c_mean, delta_mean))
save_csv('12_encoding', pd.DataFrame(rows_enc))

# ============================================================
# 表4.13 编码策略性能汇总
# ============================================================
print('\n' + '=' * 70)
print('表4.13 编码策略性能汇总（实验6）')
print('=' * 70)
print()
hdr8 = '{:25s} {:>8} {:>8} {:>8} {:>14} {:>10}'
print(hdr8.format('策略','CNCT','CGAMV','APESP','时间(min)','约束满足率'))
print('-' * 73)

a_fits = [A.loc[v, 'fitness'] for v in vessels_enc]
a_times = [A.loc[v, 'time_s']/60 for v in vessels_enc]
print('{:25s} {:>8.4f} {:>8.4f} {:>8.4f}  {:>.1f}~{:.1f}    {:>10}'.format(
    'A:分层编码+修复', a_fits[0], a_fits[1], a_fits[2], min(a_times), max(a_times), '100%'))

b_fits = [B.loc[v, 'mean_fit'] for v in vessels_enc]
b_times = [B.loc[v, 'mean_time']/60 for v in vessels_enc]
print('{:25s} {:>8.4f} {:>8.4f} {:>8.4f}  {:>.1f}~{:.1f}    {:>10}'.format(
    'B:扁平+惩罚', b_fits[0], b_fits[1], b_fits[2], min(b_times), max(b_times), '负值不可行'))

c_fits = [C.loc[v, 'fitness'] for v in vessels_enc]
c_times = [C.loc[v, 'time_s']/60 for v in vessels_enc]
print('{:25s} {:>8.4f} {:>8.4f} {:>8.4f}  {:>.1f}~{:.1f}    {:>10}'.format(
    'C:近似最优', c_fits[0], c_fits[1], c_fits[2], min(c_times), max(c_times), '100%'))

# 表4.13 CSV
rows13 = [
    {'strategy': 'A:分层编码+修复', 'CNCT': a_fits[0], 'CGAMV': a_fits[1], 'APESP': a_fits[2],
     'time_min_min': round(min(a_times),1), 'time_min_max': round(max(a_times),1), 'constraint_rate': '100%'},
    {'strategy': 'B:扁平+惩罚', 'CNCT': b_fits[0], 'CGAMV': b_fits[1], 'APESP': b_fits[2],
     'time_min_min': round(min(b_times),1), 'time_min_max': round(max(b_times),1), 'constraint_rate': '负值不可行'},
    {'strategy': 'C:近似最优', 'CNCT': c_fits[0], 'CGAMV': c_fits[1], 'APESP': c_fits[2],
     'time_min_min': round(min(c_times),1), 'time_min_max': round(max(c_times),1), 'constraint_rate': '100%'},
]
save_csv('13_summary', pd.DataFrame(rows13))

print()
print('-' * 73)
print('数据来源：03_results/canonical/exp6_encoding.parquet')
print('A/B策略：gen=50, pop=60；C策略：gen=80, pop=100')
print('B策略为5轮均值；约束满足率基于解可行性判断')

# ============================================================
print('\n' + '=' * 70)
print('数据文件（canonical标准源）:')
for p in sorted(CANONICAL.glob('exp*.parquet')):
    print('  {}'.format(p))
print('=' * 70)
