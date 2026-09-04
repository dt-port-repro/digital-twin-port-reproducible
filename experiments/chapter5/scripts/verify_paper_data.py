"""
第五章论文数据验证 — 读取规范化结果文件，输出论文全部表格
§5.1 预测模型 → 表5.1 / 图5.1-5.2
§5.2 堆场选位 → 表5.2(选位对比)
§5.3 PPO协调器 → 图5.4
§5.4 预测-优化协同 → 表5.2(协同效果)
"""
import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

def read_json(rel_path):
    with open(ROOT / rel_path) as f:
        return json.load(f)

def hdr(text):
    print()
    print('=' * 60)
    print(text)
    print('=' * 60)

# ════════════════════════════════════════
# 表5.1 预测模型对比
# ════════════════════════════════════════
hdr('表5.1 预测模型性能对比')

comparison = read_json('output/lstm_results/full_comparison_results.json')
summary = comparison['summary']

MODEL_MAP = {
    'ARIMA':        'ARIMA(2,1,2)',
    'Prophet':      'STL+ARIMA',
    'LSTM':         'LSTM',
    'Transformer':  'Transformer',
    '本文模型':       '本文模型(LSTM+Attention)',
}

print('  {:<25s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s}'.format(
    '模型', 'MAE(TEU)', 'RMSE(TEU)', 'MAPE(%)', 'PICP(%)', '时间(s)'))
print('  ' + '-' * 70)
for key, name in MODEL_MAP.items():
    if key in summary:
        m = summary[key]
        print('  {:<25s} {:>10.1f} {:>10.1f} {:>8.2f} {:>8.1f} {:>8.2f}'.format(
            name, m['mae'], m['rmse'], m['mape'], m['picp'], m['time_s']))

# 计算本文模型 vs LSTM基础模型
lstm = summary.get('LSTM', {})
ours = summary.get('LSTM_Attention', {})
if lstm and ours:
    mape_imp = (lstm['mape'] - ours['mape']) / lstm['mape'] * 100
    print('  LSTM+Attention vs LSTM基础: MAPE改善 {:.1f}% ({:.2f}% -> {:.2f}%)'.format(
        mape_imp, lstm['mape'], ours['mape']))

# ════════════════════════════════════════
# 表5.2 堆场选位优化
# ════════════════════════════════════════
hdr('表5.2 堆场选位优化对比')

selection = read_json('output/yard_selection_results/selection_results_v2.json')
print('  {:<20s} {:>10s} {:>8s} {:>10s}'.format(
    '方法', '平均惩罚', '标准差', '总时间(s)'))
print('  ' + '-' * 50)
for r in selection:
    print('  {:<20s} {:>10.4f} {:>8.4f} {:>10.2f}'.format(
        r['method'], r['avg_penalty'], r['std_penalty'], r['total_time_s']))

# 计算改善
fcfs_rec = [r for r in selection if 'FCFS' in r['method']]
stage3_rec = [r for r in selection if '三阶段惩罚' in r['method']]
if fcfs_rec and stage3_rec:
    fcfs = fcfs_rec[0]['avg_penalty']
    stage3 = stage3_rec[0]['avg_penalty']
    imp = (fcfs - stage3) / fcfs * 100
    std_imp = (fcfs_rec[0]['std_penalty'] - stage3_rec[0]['std_penalty']) / fcfs_rec[0]['std_penalty'] * 100
    print('  {} {:.1f}% ({:.4f} -> {:.4f})'.format('惩罚降低:', imp, fcfs, stage3))
    print('  {} {:.1f}% ({:.4f} -> {:.4f})'.format('标准差降低:', std_imp, fcfs_rec[0]['std_penalty'], stage3_rec[0]['std_penalty']))

# ════════════════════════════════════════
# 表5.3 PPO协调器
# ════════════════════════════════════════
hdr('表5.3 PPO协调器对比')

ppo = read_json('output/ppo_results/ppo_results.json')
print('  {:<6s} {:<20s} {:>10s} {:>12s}'.format(
    '窗口', '方法', '累积奖励', 'vs基线提升'))
print('  ' + '-' * 50)
for w in ['w1', 'w2']:
    r = ppo[w]
    bl = r['baseline_reward']
    print('  {:<6s} {:<20s} {:>10.1f} {:>+12.1f}'.format(
        w, 'PPO动态', r['test_reward'], r['test_reward'] - bl))
    print('  {:<6s} {:<20s} {:>30s}'.format(
        '', '动作分布', str(r['action_dist'])))
    print('  {:<6s} {:<20s} {:>10.1f}'.format(
        '', '静态基线(平衡)', bl))

avg = ppo['avg']
print('  {} {:>10.1f} {:>+12.1f}'.format(
    '两窗平均 PPO:', avg['test_reward'], avg['improvement']))
imp_pct = abs(avg['improvement'] / avg['baseline_reward']) * 100
print('  {} {:.1f}%'.format('相对基线提升:', imp_pct))

# ════════════════════════════════════════
# 保存CSV
# ════════════════════════════════════════
out_dir = ROOT / 'experiments' / 'chapter5' / 'output' / 'verified'
out_dir.mkdir(parents=True, exist_ok=True)

# 5.1 预测模型
with open(out_dir / 'table_5_1_prediction.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['模型', 'MAE(TEU)', 'RMSE(TEU)', 'MAPE(%)', 'PICP(%)', '时间(s)'])
    for key, name in MODEL_MAP.items():
        if key in summary:
            m = summary[key]
            w.writerow([name, m['mae'], m['rmse'], m['mape'], m['picp'], m['time_s']])

# 5.2 堆场选位
with open(out_dir / 'table_5_2_yard.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['方法', '平均惩罚', '标准差', '总时间(s)'])
    for r in selection:
        w.writerow([r['method'], r['avg_penalty'], r['std_penalty'], r['total_time_s']])

# 5.3 PPO
with open(out_dir / 'table_5_2_ppo.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['窗口', '方法', '累积奖励', 'vs基线提升'])
    for w_name in ['w1', 'w2']:
        r = ppo[w_name]
        w.writerow([w_name, 'PPO动态', r['test_reward'], round(r['test_reward'] - r['baseline_reward'], 1)])
        w.writerow([w_name, '静态基线(平衡)', r['baseline_reward'], 0])

print()
print('输出文件:')
print('  ' + str(out_dir / 'table_5_1_prediction.csv'))
print('  ' + str(out_dir / 'table_5_2_yard.csv'))
print('  ' + str(out_dir / 'table_5_2_ppo.csv'))
print('Done.')

# ════════════════════════════════════════
# 表5.2 预测-优化协同效果
# ════════════════════════════════════════
hdr('论文表5.2 预测-优化协同效果评估')

integration = read_json('output/lstm_results/prediction_yard_integration.json')
print('  基于MCT 2024年实际数据的预测-优化协同效果：')
for w in ['w1', 'w2']:
    r = integration[w]
    print(f'  {w}: MAPE={r["prediction"]["mape"]:.2f}%, PICP={r["prediction"]["picp"]:.1f}%')
    bl = r['baseline']
    wp = r['with_prediction']
    imp_r = (bl['reshuffle_pct'] - wp['reshuffle_pct']) / bl['reshuffle_pct'] * 100
    imp_e = (wp['equip_util_pct'] - bl['equip_util_pct']) / bl['equip_util_pct'] * 100
    print(f'    翻箱 {bl["reshuffle_pct"]:.1f}% -> {wp["reshuffle_pct"]:.1f}%')
    print(f'    设备 {bl["equip_util_pct"]:.1f}% -> {wp["equip_util_pct"]:.1f}%')

print()
print('  [注] 论文表5.2引用值: 翻箱33.9%(8.5%->5.6%), 设备+12.3pp')
print('  来自第五章独立实验，与本处预测-协同实验数值不同。')

# ════════════════════════════════════════
# 图表文件完整性检查
# ════════════════════════════════════════
hdr('第五章图表文件对照')

FIGS_DIR = ROOT / 'output' / 'chapter5_figures'
ALL_FIGS = {
    'fig5_1_model_comparison.png': '图5.1 预测模型对比(§5.1.3)',
    'fig5_2_training_curve.png': '图5.2 训练损失与MAPE(§5.1.3)',
    'fig5_3_picp_comparison.png': '图5.3 分位数PICP(§5.1.3)',
    'fig5_4_ppo_reward.png': '图5.4 PPO奖励曲线(§5.3.2)',
}
print(f'  {"文件":40s} {"论文对应":35s} {"状态":>6s}')
print('  ' + '-' * 83)
for f, desc in ALL_FIGS.items():
    status = 'OK' if (FIGS_DIR / f).exists() else 'MISSING'
    print(f'  {f:40s} {desc:35s} [{status:>6s}]')

print()
print('  (图5.2算法流程图, 图5.3协同框架为设计图, 见论文正文)')

# 输出CSV：协同效果
with open(out_dir / 'table_5_2_collaborative.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['窗口', 'MAPE(%)', 'PICP(%)', '基线翻箱(%)', '协同翻箱(%)', '基线设备(%)', '协同设备(%)'])
    for w_name in ['w1', 'w2']:
        r = integration[w_name]
        bl = r['baseline']
        wp = r['with_prediction']
        w.writerow([w_name, r['prediction']['mape'], r['prediction']['picp'],
                     bl['reshuffle_pct'], wp['reshuffle_pct'],
                     bl['equip_util_pct'], wp['equip_util_pct']])

print(f'\n保存: {out_dir}/table_5_2_collaborative.csv')
print('All done.')
