#!/usr/bin/env python3
"""生成配套表 S13: PPO训练奖励历史"""
import json, os, csv

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fp = os.path.join(PKG, 'output/chapter5/results/ppo_results.json')
fallback = os.path.join(PKG, 'output', 'chapter5', 'results', 'ppo_results.json')
path = fp if os.path.exists(fp) else fallback

with open(path) as f:
    data = json.load(f)

w1 = data['w1']
w2 = data['w2']

rows = []
max_ep = max(len(w1['train_rewards_history']), len(w2['train_rewards_history']),
             len(w1['test_rewards_history']), len(w2['test_rewards_history']))

for i in range(max_ep):
    rows.append({
        '轮次': i + 1,
        'W1训练奖励': w1['train_rewards_history'][i] if i < len(w1['train_rewards_history']) else '',
        'W2训练奖励': w2['train_rewards_history'][i] if i < len(w2['train_rewards_history']) else '',
        'W1测试奖励': w1['test_rewards_history'][i] if i < len(w1['test_rewards_history']) else '',
        'W2测试奖励': w2['test_rewards_history'][i] if i < len(w2['test_rewards_history']) else '',
    })

out = os.path.join(PKG, '03_results/tables/table_S13_ppo_training_history.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['轮次', 'W1训练奖励', 'W2训练奖励', 'W1测试奖励', 'W2测试奖励'])
    w.writeheader()
    w.writerows(rows)

print(f'✅ table_S13 saved: {out} ({len(rows)} rows)')
