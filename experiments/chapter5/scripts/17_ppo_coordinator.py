"""
Step 17: PPO 港口协调器（论文第五章下半）
===========================================
状态：41维港口状态（6h粒度）
动作：3个离散协调决策
奖励：泊位拥堵成本 + 堆场拥挤成本 + 停留时间惩罚
评估：两窗walk-forward累计reward
"""
import pandas as pd, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json, time, warnings
warnings.filterwarnings('ignore')

OUT = Path('output')
SPLIT = OUT / 'splits'
RESULT = OUT / 'ppo_results'
RESULT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')


# ════════════════════════════════════════════
# 1. 港口环境
# ════════════════════════════════════════════

class PortEnv:
    """港口协调环境——基于历史数据的仿真"""
    
    # 动作空间: 0=优先泊位, 1=平衡, 2=优先堆场
    N_ACTIONS = 3
    # 关键状态索引
    COL_BERTH = 1   # berth_occupancy
    COL_DURATION = 2 # avg_duration_h
    COL_DELAY = 3   # avg_delay_h
    COL_YARD = 7    # utilization_rate (12.35 avg)
    COL_BOXES = 5   # total_boxes
    COL_DWELL = 10  # dwell_mean_h
    
    def __init__(self, states_df, action_effects=None):
        """
        states_df: DataFrame with time_bin + 41 state columns
        action_effects: dict, 每个动作对rewards的乘数影响
        """
        self.data = states_df.reset_index(drop=True)
        self.n_steps = len(self.data)
        self.action_effects = action_effects or {
            0: {'berth_weight': 0.6, 'yard_weight': 1.2, 'dwell_weight': 1.0},  # 优先泊位
            1: {'berth_weight': 1.0, 'yard_weight': 1.0, 'dwell_weight': 1.0},  # 平衡
            2: {'berth_weight': 1.4, 'yard_weight': 0.6, 'dwell_weight': 0.8},  # 优先堆场
        }
        
        # 提取状态（去掉time_bin和所有datetime列）
        time_cols = [c for c in self.data.columns if self.data[c].dtype == 'datetime64[us]' or self.data[c].dtype == 'datetime64[ns]']
        feature_cols = [c for c in self.data.columns if c not in time_cols]
        self.states = self.data[feature_cols].values.astype(np.float32)
        # 填充NaN（lag列开头可能有NaN）
        col_means = np.nanmean(self.states, axis=0)
        self.states = np.where(np.isnan(self.states), col_means, self.states)
        self.feature_cols = feature_cols
        
        # 计算各维度的归一化参数（从全部数据获取）
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0) + 1e-8
        
        # 提取成本相关列索引
        self.col_berth_delay = self._find_col('avg_delay_h')
        self.col_yard_occ = self._find_col('utilization_rate')
        self.col_dwell = self._find_col('dwell_mean_h')
        self.col_berth_occ = self._find_col('berth_occupancy')
        self.col_boxes = self._find_col('total_boxes')
        
        # 成本归一化常数
        self.delay_max = self.states[:, self.col_berth_delay].max()
        self.yard_max = self.states[:, self.col_yard_occ].max()
        self.dwell_max = self.states[:, self.col_dwell].max()
        self.berth_occ_max = self.states[:, self.col_berth_occ].max()
        
        self.current_idx = 0
    
    def _find_col(self, name):
        return self.feature_cols.index(name)
    
    def reset(self):
        """重置到序列开始"""
        self.current_idx = 0
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态（归一化）"""
        s = self.states[self.current_idx]
        return (s - self.state_mean) / self.state_std
    
    def step(self, action):
        """
        执行动作，转移到下一时间步
        返回: next_state, reward, done, info
        """
        idx = self.current_idx
        effects = self.action_effects[action]
        
        # 使用当前状态计算成本
        delay = self.states[idx, self.col_berth_delay]
        yard = self.states[idx, self.col_yard_occ]
        dwell = self.states[idx, self.col_dwell]
        berth_occ = self.states[idx, self.col_berth_occ]
        
        # 归一化成本
        cost_delay = delay / self.delay_max
        cost_yard = yard / self.yard_max
        cost_dwell = dwell / self.dwell_max
        
        # 动作影响的加权总成本（越小越好）
        total_cost = (
            effects['berth_weight'] * cost_delay +
            effects['yard_weight'] * cost_yard +
            effects['dwell_weight'] * cost_dwell
        )
        
        # 额外惩罚：高泊位占用（超过5个）+ 超长停留
        if berth_occ > 0.6 * self.berth_occ_max:
            total_cost += 0.1
        if dwell > 0.8 * self.dwell_max:
            total_cost += 0.1
        
        # Reward = 负成本，范围大致[-2, 0]
        reward = -total_cost
        
        # 转移到下一步
        self.current_idx += 1
        done = self.current_idx >= self.n_steps - 1
        
        if not done:
            next_state = self._get_state()
        else:
            next_state = self._get_state()  # 最后一帧
        
        info = {
            'cost_delay': float(cost_delay),
            'cost_yard': float(cost_yard),
            'cost_dwell': float(cost_dwell),
            'berth_occ': float(berth_occ),
        }
        
        return next_state, reward, done, info
    
    @property
    def state_dim(self):
        return len(self.feature_cols)


# ════════════════════════════════════════════
# 2. PPO网络
# ════════════════════════════════════════════

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 共享特征提取
        self.feat = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor: 策略网络
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic: 价值网络
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.feat(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, state, deterministic=False):
        """采样动作"""
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        
        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, action_log_prob, value.squeeze(-1), probs


# ════════════════════════════════════════════
# 3. PPO训练
# ════════════════════════════════════════════

class PPO:
    """PPO算法实现"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5):
        self.net = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
    
    def collect_trajectory(self, env, max_steps=None):
        """收集一条轨迹"""
        states, actions, rewards, dones = [], [], [], []
        log_probs, values = [], []
        
        state = env.reset()
        step = 0
        
        while True:
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = self.net(s_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
                log_prob = F.log_softmax(logits, dim=-1)
                act_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            
            next_state, reward, done, info = env.step(action.item())
            
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(act_log_prob.item())
            values.append(value.item())
            
            state = next_state
            step += 1
            
            if done or (max_steps and step >= max_steps):
                break
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
        }
    
    def compute_gae(self, rewards, values, dones, next_value=0):
        """GAE优势估计"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns
    
    def update(self, trajectory, epochs=10, batch_size=64):
        """PPO策略更新"""
        states = torch.FloatTensor(trajectory['states']).to(device)
        actions = torch.LongTensor(trajectory['actions']).to(device)
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).to(device)
        
        # 计算GAE
        advantages, returns = self.compute_gae(
            trajectory['rewards'], trajectory['values'], trajectory['dones'])
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n = len(states)
        total_loss = 0
        
        for _ in range(epochs):
            # Mini-batch
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]
                
                # 当前策略
                logits, values = self.net(batch_states)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_probs = log_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clip loss
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_ret)
                
                # Total loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return {
            'loss': total_loss / (epochs * (n // batch_size + 1)),
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
    
    def evaluate(self, env, n_episodes=1):
        """评估策略（确定性）"""
        total_rewards = []
        actions_hist = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            ep_reward = 0
            ep_actions = []
            
            while not done:
                s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _, _, _ = self.net.get_action(s_tensor, deterministic=True)
                next_state, reward, done, info = env.step(action.item())
                ep_reward += reward
                ep_actions.append(action.item())
                state = next_state
            
            total_rewards.append(ep_reward)
            actions_hist.extend(ep_actions)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'action_dist': pd.Series(actions_hist).value_counts(normalize=True).to_dict() if actions_hist else {},
        }


# ════════════════════════════════════════════
# 4. 主流程
# ════════════════════════════════════════════

def run():
    print('=' * 60)
    print('PPO 港口协调器（论文第五章）')
    print('=' * 60)
    
    # 加载数据
    print('\n[1/3] 加载PPO状态空间...')
    full_df = pd.read_parquet(OUT / '10_ppo_state_space.parquet')
    print(f'  总数据: {len(full_df)}步, 41维状态')
    
    # 两窗训练+评估
    windows = {
        'w1': (SPLIT / 'w1' / 'state' / 'train.parquet', 
               SPLIT / 'w1' / 'state' / 'test.parquet',
               'Jan-Jun→Jul/Aug'),
        'w2': (SPLIT / 'w2' / 'state' / 'train.parquet', 
               SPLIT / 'w2' / 'state' / 'test.parquet',
               'Jan-Sep→Oct/Dec'),
    }
    
    all_results = {}
    
    for win_name, (train_path, test_path, desc) in windows.items():
        print(f'\n[2/3] {win_name}: {desc}')
        
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        print(f'  训练: {len(train_df)}步')
        print(f'  测试: {len(test_df)}步')
        
        # 构建环境
        train_env = PortEnv(train_df)
        test_env = PortEnv(test_df)
        
        print(f'  状态维度: {train_env.state_dim}')
        print(f'  动作空间: {PortEnv.N_ACTIONS}个离散动作')
        print(f'  动作说明: 0=优先泊位, 1=平衡, 2=优先堆场')
        
        # 初始化PPO
        ppo = PPO(
            state_dim=train_env.state_dim,
            action_dim=PortEnv.N_ACTIONS,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
        )
        
        # 训练
        print('\n  训练中...')
        t0 = time.time()
        n_iterations = 50
        train_rewards = []
        test_rewards = []
        
        for i in range(n_iterations):
            # 收集轨迹
            traj = ppo.collect_trajectory(train_env)
            
            # 更新策略
            update_info = ppo.update(traj, epochs=10, batch_size=64)
            
            # 评估
            train_eval = ppo.evaluate(train_env)
            test_eval = ppo.evaluate(test_env)
            
            train_rewards.append(train_eval['mean_reward'])
            test_rewards.append(test_eval['mean_reward'])
            
            if i % 10 == 0:
                print(f'  Iter {i:3d}: train_R={train_eval["mean_reward"]:.3f}±{train_eval["std_reward"]:.3f}, '
                      f'test_R={test_eval["mean_reward"]:.3f}±{test_eval["std_reward"]:.3f}, '
                      f'entropy={update_info["entropy"]:.3f}')
        
        elapsed = time.time() - t0
        print(f'  完成: {elapsed:.0f}s')
        
        # 最终评估
        final_train = ppo.evaluate(train_env)
        final_test = ppo.evaluate(test_env)
        
        print(f'\n  最终训练集: R={final_train["mean_reward"]:.3f}±{final_train["std_reward"]:.3f}')
        print(f'  最终测试集: R={final_test["mean_reward"]:.3f}±{final_test["std_reward"]:.3f}')
        print(f'  动作分布: {final_test["action_dist"]}')
        
        # 基准策略对比（始终选"平衡"动作）
        baseline_rewards = []
        for _ in range(3):
            state = test_env.reset()
            done = False
            ep_r = 0
            while not done:
                state, reward, done, _ = test_env.step(1)  # 动作1=平衡
                ep_r += reward
            baseline_rewards.append(ep_r)
        baseline_r = np.mean(baseline_rewards)
        print(f'  基准策略(始终平衡): R={baseline_r:.3f}')
        
        all_results[win_name] = {
            'train_reward': float(final_train['mean_reward']),
            'test_reward': float(final_test['mean_reward']),
            'baseline_reward': float(baseline_r),
            'action_dist': {str(k): float(v) for k, v in final_test['action_dist'].items()},
            'improvement_over_baseline': float(final_test['mean_reward'] - baseline_r),
            'train_rewards_history': [float(r) for r in train_rewards],
            'test_rewards_history': [float(r) for r in test_rewards],
            'n_train_steps': len(train_df),
            'n_test_steps': len(test_df),
            'elapsed_s': elapsed,
        }
    
    # 汇总
    print('\n[3/3] 汇总')
    print('=' * 60)
    
    for win_name, res in all_results.items():
        print(f'\n{win_name}:')
        print(f'  训练R={res["train_reward"]:.3f}, 测试R={res["test_reward"]:.3f}, 基准={res["baseline_reward"]:.3f}')
        print(f'  提升: {res["improvement_over_baseline"]:.3f}')
        print(f'  动作分布: {res["action_dist"]}')
    
    # 两窗平均
    if len(all_results) >= 2:
        avg_train = np.mean([r['train_reward'] for r in all_results.values()])
        avg_test = np.mean([r['test_reward'] for r in all_results.values()])
        avg_baseline = np.mean([r['baseline_reward'] for r in all_results.values()])
        avg_improve = np.mean([r['improvement_over_baseline'] for r in all_results.values()])
        print(f'\n两窗平均:')
        print(f'  训练R={avg_train:.3f}, 测试R={avg_test:.3f}, 基准={avg_baseline:.3f}')
        print(f'  平均提升: {avg_improve:.3f}')
        
        all_results['avg'] = {
            'train_reward': avg_train,
            'test_reward': avg_test,
            'baseline_reward': avg_baseline,
            'improvement': avg_improve,
        }
    
    # 保存
    with open(RESULT / 'ppo_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n📄 结果保存: {RESULT}/ppo_results.json')


if __name__ == '__main__':
    run()
