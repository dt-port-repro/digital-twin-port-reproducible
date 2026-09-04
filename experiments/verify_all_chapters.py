"""
四五六章论文数据综合验证脚本
读取规范化数据文件，输出论文全部表格的精确值
"""
import json, pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent  # project root

def hdr(text):
    print(f'\n{"="*70}\n{text}\n{"="*70}')

def safe_mean(arr):
    arr = arr.dropna()
    return arr.mean() if len(arr) > 0 else np.nan

# ══════════════════════════════════════════════════════════
# 第四章 船舶智能配载优化
# ══════════════════════════════════════════════════════════
hdr('第四章 — 船舶智能配载优化')

CH4_RESULTS = ROOT / 'experiments' / 'chapter4' / 'results'

# 实验1：基线测试（单次）
test_files = {
    'CNTIG': CH4_RESULTS / 'test_cntig.parquet',
    'CNCT': CH4_RESULTS / 'test_cnct_795.parquet',
    'MXNT': CH4_RESULTS / 'test_mxnt.parquet',
    'OFUT': CH4_RESULTS / 'test_ofut.parquet',
    'CGAMV': CH4_RESULTS / 'test_cgamv.parquet',
    'APESP': CH4_RESULTS / 'test_apesp.parquet',
}

print('\n表4.x GA-RH在核心船舶上的可行性验证结果')
print(f'{"船名":<8} {"箱量":>6} {"挂港":>4} {"fitness":>10} {"f₁翻箱":>8} {"时间(s)":>8}')
print('-'*50)
fitness_vals = []
teu_vals = []
for ship, fpath in test_files.items():
    if fpath.exists():
        df = pd.read_parquet(fpath)
        fit = df['fitness'].iloc[0]
        fitness_vals.append(fit)
        teu = df['n_containers'].iloc[0]
        teu_vals.append(teu)
        f1 = df.get('rehandle', pd.Series([np.nan])).iloc[0]
        t = df.get('time_s', pd.Series([0])).iloc[0]
        print(f'{ship:<8} {teu:>6.0f} {"?":>4} {fit:>10.4f} {f1:>8.4f} {t:>8.0f}')
    else:
        print(f'{ship:<8} 文件不存在')

# 相关性：CNCT 有 penalty 违反（负 fitness），计算时排除
pos_fit = [f for f in fitness_vals if f > 0]
pos_teu = [t for t, f in zip(teu_vals, fitness_vals) if f > 0]
if len(pos_fit) >= 3:
    r, p = stats.pearsonr(pos_teu, pos_fit)
    print(f'\n  fitness与箱量相关性（排除CNCT）: r={r:.3f}, p={p:.6f}')
    print(f'  范围从 {min(pos_fit):.4f} 到 {max(pos_fit):.4f}')
else:
    print('\n  不能计算相关性（正fitness样本不足）')

# 实验2：GA-RH vs 纯GA（消融实验5轮均值）
print('\n\n表4.x GA-RH与纯GA对比（5轮均值）')
exp5 = pd.read_parquet(CH4_RESULTS / 'all_experiments.parquet')
exp5 = exp5[exp5['experiment'] == 'exp5_ablation']
print(f'{"船名":<8} {"GA-RH均值":>10} {"GA-纯均值":>10} {"method列":>8}')
for ship in ['CNCT','CGAMV','APESP']:
    sub = exp5[exp5['vessel_code'] == ship]
    if len(sub) > 0:
        print(f'{ship:<8} {sub["fitness_before"].dropna().mean():>10.4f} {sub["fitness_after"].dropna().mean():>10.4f} {sub["method"].iloc[0]}')
        print(f'        GA-RH前={sub["fitness_before"].mean():.4f} GA-RH后={sub["fitness_after"].mean():.4f} 改善={sub["improvement_pct"].mean():.2f}%')

# 表4.14 协同效果
print('\n\n表4.x 完整GA-RH与独立优化对比')
exp3 = ae = pd.read_parquet(CH4_RESULTS / 'all_experiments.parquet')
exp3 = exp3[exp3['experiment'] == 'exp3_gamma0']
print(f'{"船名":<8} {"共同优化":>10} {"独立(γ=0)":>12} {"差异":>8}')
for ship in ['CNTIG','CNCT','MXNT','OFUT','CGAMV','APESP']:
    sub = exp3[exp3['vessel_code'] == ship]
    if len(sub) > 0:
        full_fit = sub['fitness'].mean()
        print(f'{ship:<8} {full_fit:>10.4f}')

# ══════════════════════════════════════════════════════════
# 第五章 堆场作业预测与优化
# ══════════════════════════════════════════════════════════
hdr('第五章 — 堆场作业预测与优化')

# 表5.1 预测模型
cmp = json.load(open(ROOT / 'output/lstm_results/full_comparison_results.json'))
print('\n表5.1 预测模型对比')
print(f'{"模型":<25} {"MAE":>8} {"RMSE":>8} {"MAPE":>6} {"PICP":>6}')
print('-'*55)
MODEL_MAP = {
    'ARIMA': 'ARIMA(2,1,2)', 'Prophet': 'STL+ARIMA',
    'LSTM': 'LSTM', 'Transformer': 'Transformer',
    '本文模型': '本文模型(LSTM+Attention)',
}
for key, name in MODEL_MAP.items():
    if key in cmp['summary']:
        m = cmp['summary'][key]
        print(f'{name:<25} {m["mae"]:>8.1f} {m["rmse"]:>8.1f} {m["mape"]:>6.2f}% {m["picp"]:>6.1f}%')

# 选位结果
print('\n表5.X 堆场选位对比')
sel = json.load(open(ROOT / 'output/yard_selection_results/selection_results_v2.json'))
print(f'{"方法":<20} {"平均惩罚":>10} {"标准差":>8}')
for r in sel[:3]:
    print(f'{r["method"]:<20} {r["avg_penalty"]:>10.4f} {r["std_penalty"]:>8.4f}')
fcfs = sel[0]['avg_penalty']
stage3 = sel[1]['avg_penalty']
print(f'  改善: {(fcfs-stage3)/fcfs*100:.1f}% ({fcfs:.4f}→{stage3:.4f})')

# PPO
print('\n表5.X PPO协调器对比')
ppo = json.load(open(ROOT / 'output/ppo_results/ppo_results.json'))
print(f'{"窗口":<6} {"PPO奖励":>10} {"基线":>10} {"提升":>8}')
for w in ['w1','w2']:
    r = ppo[w]
    print(f'{w:<6} {r["test_reward"]:>10.1f} {r["baseline_reward"]:>10.1f} {r["test_reward"]-r["baseline_reward"]:>+8.1f}')
print(f'{"平均":<6} {ppo["avg"]["test_reward"]:>10.1f} {ppo["avg"]["baseline_reward"]:>10.1f} {ppo["avg"]["test_reward"]-ppo["avg"]["baseline_reward"]:>+8.1f}')

# ══════════════════════════════════════════════════════════
# 第六章 数字孪生验证
# ══════════════════════════════════════════════════════════
hdr('第六章 — 数字孪生系统验证')

CANONICAL = {
    '常规': 'sim_常规作业_cfgall_d30r3.parquet',
    '高峰': 'sim_高峰压力_cfgall_d30r10.parquet',
    '异常': 'sim_异常情况_cfgall_d30r10.parquet',
}

SIM_DIR = ROOT / 'output' / 'simulation_results'

for sc, fname in CANONICAL.items():
    df = pd.read_parquet(SIM_DIR / fname)
    print(f'\n{sc}场景')
    print(f'{"配置":<5} {"船时":>8} {"翻箱":>8} {"设备":>8} {"船时改善":>8} {"翻箱改善":>8} {"设备改善":>8}')
    for cfg in ['A','B','C','D']:
        sub = df[df['config']==cfg]
        t = sub['turnaround_h'].mean()
        r = sub['reshuffle_pct'].mean()
        e = sub['equip_util_pct'].mean()
        if cfg == 'A':
            print(f'{cfg:<5} {t:>8.1f} {r:>8.1f} {e:>8.1f} {"---":>8} {"---":>8} {"---":>8}')
        else:
            imp_t = (df[df['config']=='A']['turnaround_h'].mean() - t) / df[df['config']=='A']['turnaround_h'].mean() * 100
            imp_r = (df[df['config']=='A']['reshuffle_pct'].mean() - r) / df[df['config']=='A']['reshuffle_pct'].mean() * 100
            imp_e = (e - df[df['config']=='A']['equip_util_pct'].mean()) / df[df['config']=='A']['equip_util_pct'].mean() * 100
            print(f'{cfg:<5} {t:>8.1f} {r:>8.1f} {e:>8.1f} {imp_t:>7.1f}% {imp_r:>7.1f}% {imp_e:>7.1f}%')

# 统计检验
print('\n统计显著性检验')
print(f'{"场景":<6} {"A均值":>8} {"D均值":>8} {"改善":>8} {"t值":>8} {"p值":>8} {"Cohen d":>8}')
for sc, fname in CANONICAL.items():
    df = pd.read_parquet(SIM_DIR / fname)
    a = df[df['config']=='A']['turnaround_h']
    d = df[df['config']=='D']['turnaround_h']
    t_stat, p_val = stats.ttest_ind(d, a, alternative='less')
    n1, n2 = len(a), len(d)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*d.var(ddof=1)) / (n1+n2-2))
    cd = (a.mean() - d.mean()) / sp
    imp = (a.mean()-d.mean())/a.mean()*100
    print(f'{sc:<6} {a.mean():>8.1f} {d.mean():>8.1f} {imp:>7.1f}% {t_stat:>8.1f} {p_val:>8.6f} {cd:>8.2f}')

print('\n\n✅ 验证完成！')
print('注意：')
print('  - 第四章论文数据与parquet不匹配（CNTIG 0.7921 vs 0.7537等）')
print('  - 第五章表5.2的翻箱率/设备利用率数据无脚本出处')
print('  - 第六章文字"3.1%"与表格"↓2.9%"矛盾')
