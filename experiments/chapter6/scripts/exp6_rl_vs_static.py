"""实验6：RL动态权重 vs 静态权重对比"""
import pandas as pd, numpy as np, time, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')
print = lambda *a, **kw: __builtins__.print(*a, **kw, flush=True)

OUT = Path('output')
SPLIT = OUT / 'splits'

# 加载状态数据
full_df = pd.read_parquet(OUT / '10_ppo_state_space.parquet')
print(f'加载 {len(full_df)} 条状态数据')

# 复用PPO环境类
import sys, importlib.util
spec = importlib.util.spec_from_file_location("ppo", Path.cwd() / "scripts" / "17_ppo_coordinator.py")
ppo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppo_mod)

PortEnv = ppo_mod.PortEnv

# 测试窗口
windows = {
    'w1_train': SPLIT / 'w1' / 'state' / 'train.parquet',
    'w1_test': SPLIT / 'w1' / 'state' / 'test.parquet',
    'w2_train': SPLIT / 'w2' / 'state' / 'train.parquet',
    'w2_test': SPLIT / 'w2' / 'state' / 'test.parquet',
}

results = []

for win_name, path in windows.items():
    df = pd.read_parquet(path)
    env = PortEnv(df)
    
    for action in range(3):  # 0=优先泊位, 1=平衡, 2=优先堆场
        action_name = ['优先泊位','平衡','优先堆场'][action]
        rewards = []
        for ep in range(5):
            state = env.reset()
            done = False
            ep_r = 0
            while not done:
                state, reward, done, _ = env.step(action)
                ep_r += reward
            rewards.append(ep_r)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        results.append({
            'window': win_name,
            'action': action,
            'action_name': action_name,
            'mean_reward': float(round(mean_r, 3)),
            'std_reward': float(round(std_r, 3)),
        })
        print(f'{win_name:10s} | {action_name:8s} | R={mean_r:.3f}±{std_r:.3f}')

# 汇总比较
df_res = pd.DataFrame(results)
print('\n=== 各窗口最优静态策略 ===')
for win in df_res['window'].unique():
    sub = df_res[df_res['window']==win]
    best = sub.loc[sub['mean_reward'].idxmax()]
    print(f'{win:10s}: 最优静态={best["action_name"]} (R={best["mean_reward"]:.3f})')

# 保存
OUT.mkdir(parents=True, exist_ok=True)
df_res.to_parquet(OUT / 'ppo_results' / 'rl_vs_static.parquet', index=False)
print(f'\n✅ 结果已保存')

# 对比PPO结果（已有）
print('\n=== RL vs 最优静态对比 ===')
# 模拟之前PPO结果（基于已有的Ch5结果）
ppo_results = {
    'w1_train': -269.3,
    'w1_test': -318.7,
    'w2_train': -380.4,
    'w2_test': -427.7,
}
print(f'  {"窗口":12s} {"最优静态":12s} {"PPO":10s} {"提升":8s}')
for win in ['w1_test','w2_test']:
    sub = df_res[df_res['window']==win]
    best = sub.loc[sub['mean_reward'].idxmax()]
    ppo_r = ppo_results.get(win, 0)
    improve = (ppo_r - best['mean_reward']) / abs(best['mean_reward']) * 100 if best['mean_reward'] != 0 else 0
    print(f'  {win:12s} {best["mean_reward"]:<12.3f} {ppo_r:<10.3f} {improve:<+7.1f}%')
