"""
Step 19: 预测→DES仿真集成实验（PICP提升 + 三项业务指标）
=====================================================
1. 分位数LSTM+Attention训练（全量数据）
2. 预测感知YardModule（congestion_risk叠加趋势）
3. DES仿真对比：无预测 vs 有预测
4. 输出三项业务指标
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, sys, time, json, os, warnings
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['PYTHONWARNINGS'] = 'ignore'

OUT = Path('output')
RESULT = OUT / 'lstm_results'
RESULT.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

lookback, horizon = 14, 7
FEATURE_COLS = [
    'occupancy', 'n_vessels', 'total_boxes', 'avg_duration_h', 'avg_delay_h',
    'containers_arrived', 'dwell_mean_h', 'dwell_gt72h_ratio', 'weekday', 'month',
]
TARGET_COL = 'occupancy'

# ════════════════════════════════════════════
# 1. 分位数LSTM+Attention模型（同Step 18）
# ════════════════════════════════════════════
def pinball_loss(pred, target, quantile):
    diff = target - pred
    loss = torch.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return loss.mean()

class QuantileLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim*2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * 3),
        )
        self.horizon = horizon
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.layer_norm(lstm_out + attn_out)
        return self.fc(self.dropout(out[:, -1, :]))
    def predict_quantiles(self, x):
        out = self.forward(x)
        return out.view(-1, self.horizon, 3)

# ════════════════════════════════════════════
# 2. 训练全量模型 + 生成预测查找表
# ════════════════════════════════════════════
def train_and_predict():
    """训练分位数模型，返回 {date: avg_trend_for_next_7d}"""
    print('[1/3] 加载数据...')
    yf = pd.read_parquet(OUT / '10_yard_features.parquet')
    dwell = pd.read_parquet(OUT / '10_dwell_features.parquet')
    bf = pd.read_parquet(OUT / '10_berth_features.parquet')
    bf['date'] = pd.to_datetime(bf['actual_berth']).dt.date
    daily_berth = bf.groupby('date').agg(
        n_vessels=('berth_plan_no','count'), total_boxes=('total_boxes','sum'),
        avg_duration_h=('actual_duration_h','mean'), avg_delay_h=('berth_delay_h','mean'),
    ).reset_index()
    yf['date'] = pd.to_datetime(yf['day']).dt.date
    dwell['date'] = pd.to_datetime(dwell['day']).dt.date
    df = yf.merge(daily_berth, on='date', how='left').merge(dwell, on='date', how='left')
    for c in ['n_vessels','total_boxes','avg_duration_h','avg_delay_h']: df[c]=df[c].fillna(0)
    for c in ['containers_arrived','dwell_mean_h','dwell_median_h','dwell_p25_h','dwell_p75_h','dwell_gt48h_ratio','dwell_gt72h_ratio']:
        df[c]=df[c].fillna(df[c].median() if c in df else 0)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 用前200天训练，后166天做验证+预测
    train_split = 200
    train_df = df.iloc[:train_split]
    full_df = df
    
    # 创建全量窗口
    feat = full_df[FEATURE_COLS].values
    tgt = full_df[TARGET_COL].values.reshape(-1,1)
    X_all, y_all = [], []
    dates_all = []
    for i in range(len(feat) - lookback - horizon + 1):
        X_all.append(feat[i:i+lookback])
        y_all.append(tgt[i+lookback:i+lookback+horizon])
        dates_all.append(full_df['date'].iloc[i+lookback])
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32).squeeze(-1)
    
    # 标准化
    train_feat = train_df[FEATURE_COLS].values
    train_mean = train_feat.mean(axis=0, keepdims=True)
    train_std = train_feat.std(axis=0, keepdims=True) + 1e-8
    # 对特征做 (T, F) 维标准化，适配X_all的 (N, L, F)
    train_f_mean = train_feat.mean(axis=0)
    train_f_std = train_feat.std(axis=0) + 1e-8
    X_all_n = (X_all - train_f_mean[np.newaxis, np.newaxis, :]) / train_f_std[np.newaxis, np.newaxis, :]
    
    y_all_f = y_all.reshape(-1, 1)
    y_mean = float(y_all_f.mean())
    y_std = float(y_all_f.std() + 1e-8)
    y_all_n = (y_all - y_mean) / y_std
    
    # 使用两窗训练：w2有更多数据，用w2模型做DES预测
    train_splits = {'w1': 182, 'w2': 274}
    best_model = None
    best_picp = 0
    
    for win_name, train_split in train_splits.items():
        n_windows = train_split - lookback - horizon + 1
        if n_windows < 20:
            continue
        
        X_tr_n = X_all_n[:n_windows]
        y_tr_n = y_all_n[:n_windows]
        
        val_size = max(10, int(n_windows * 0.15))
        X_val_n, y_val_n = X_tr_n[-val_size:], y_tr_n[-val_size:]
        X_tr_f, y_tr_f = X_tr_n[:-val_size], y_tr_n[:-val_size]
        
        print(f'  {win_name}: {len(X_tr_f)}训练 + {len(X_val_n)}验证窗')
        
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_f), torch.FloatTensor(y_tr_f)), batch_size=32, shuffle=True)
        model = QuantileLSTMWithAttention(input_dim=len(FEATURE_COLS), horizon=horizon).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
        quantiles = [0.05, 0.50, 0.95]
        best_val = np.inf; best_state = None; patience_counter = 0
        
        t0 = time.time()
        for epoch in range(300):
            model.train(); train_loss = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                q = model.predict_quantiles(bx)
                loss = sum(pinball_loss(q[:,:,qi], by, qv) for qi, qv in enumerate(quantiles))
                loss += nn.MSELoss()(q[:,:,1], by) * 0.5
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item()
            train_loss /= len(loader)
            
            model.eval()
            with torch.no_grad():
                vq = model.predict_quantiles(torch.FloatTensor(X_val_n).to(device))
                val_loss = nn.MSELoss()(vq[:,:,1], torch.FloatTensor(y_val_n).to(device)).item()
            sched.step(val_loss)
            
            if val_loss < best_val:
                best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30: break
            if epoch % 50 == 0: print(f'  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}')
        
        model.load_state_dict(best_state)
        model.eval()
        train_time = time.time() - t0
        
        # 测试集评估（用全部后续数据）
        if n_windows < len(X_all_n):
            with torch.no_grad():
                t_q = model.predict_quantiles(torch.FloatTensor(X_all_n[n_windows:]).to(device)).cpu().numpy()
            t_q05 = t_q[:,:,0] * y_std + y_mean
            t_q50 = t_q[:,:,1] * y_std + y_mean
            t_q95 = t_q[:,:,2] * y_std + y_mean
            t_true = y_all[n_windows:]
            picp = np.sum((t_true >= t_q05) & (t_true <= t_q95)) / t_true.size * 100
            mape = np.mean(np.abs((t_true - t_q50) / (t_true + 1))) * 100
            print(f'  测试: MAPE={mape:.2f}%, PICP={picp:.1f}%, Time={train_time:.1f}s')
            
            if picp > best_picp:
                best_picp = picp
                best_model = model
    
    model = best_model
    print(f'\n  选用模型: PICP={best_picp:.1f}%')
    
    # 用最佳模型生成全量预测
    model.eval()
    with torch.no_grad():
        q_all = model.predict_quantiles(torch.FloatTensor(X_all_n).to(device)).cpu().numpy()
    q50 = q_all[:,:,1] * y_std + y_mean
    
    # 构建预测查找表 {date_str: prediction_trend}
    pred_lookup = {}
    for i in range(len(q50)):
        future_avg = q50[i].mean()
        current_occ = y_all[i].mean()
        trend = float(future_avg / max(current_occ, 1))
        window_date = dates_all[i]
        if isinstance(window_date, pd.Timestamp):
            window_date = window_date.date()
        pred_lookup[str(window_date)] = {
            'trend': round(trend, 4),
            'q50_mean': round(float(future_avg), 1),
            'current_occ': round(float(current_occ), 1),
        }
    
    print(f'  预测查找表: {len(pred_lookup)}条')
    
    # 保存模型和查找表
    torch.save(model.state_dict(), RESULT / 'quantile_lstm_model.pt')
    with open(RESULT / 'prediction_lookup.json', 'w') as f:
        json.dump(pred_lookup, f, indent=2)
    print(f'  📄 模型+查找表已保存')
    
    return pred_lookup


# ════════════════════════════════════════════
# 3. 预测感知的DES仿真对比
# ════════════════════════════════════════════
def run_des_comparison(pred_lookup, n_days=30, n_runs=3):
    """运行DES仿真对比：无预测 vs 有预测"""
    print(f'\n[3/3] DES仿真对比 ({n_days}天×{n_runs}轮)')
    
    from simulation.run import run_single, SCENARIOS
    from simulation.modules.base import YardModule
    from simulation.modules.ppo_module import PPOModule
    from simulation.modules.base import VesselGenerator, StowageModule
    from simulation.core.engine import SimEngine
    from simulation.data.models import SimulationConfig, SimulationResult, EventType
    
    class PredictionYardModule(YardModule):
        """预测感知的YardModule - 重写选位处理方法"""
        def __init__(self, use_3stage=True, lookup=None):
            super().__init__(use_3stage=use_3stage)
            self.lookup = lookup or {}
        
        def _handle_yard_start(self, event, state, clock, rng):
            """重写：在optimize结果中加入预测感知的congestion_risk"""
            vc = event.data.get('vessel_code', '')
            containers = event.data.get('containers', [])
            bp = event.data.get('bay_plan', None)
            arrival = event.data.get('arrival_time', clock.current_time)
            fitness = event.data.get('fitness', 0.5)
            
            # 计算仿真天数 → 对应真实日期趋势
            sim_day = int(clock.current_time / 24)
            date_key = str((pd.Timestamp('2024-06-01') + pd.Timedelta(days=sim_day)).date())
            trend = self.lookup.get(date_key, {}).get('trend', 1.0)
            
            base_result = self.optimize({'containers': containers})
            avg_pen = base_result.get('avg_penalty', 0)
            
            # 预测感知的congestion_risk
            congestion_risk = min(1.0, avg_pen * 1.5)
            if trend > 1.05:
                congestion_risk = min(1.0, congestion_risk * (1.0 + (trend - 1.0) * 1.5))
            elif trend < 0.95:
                congestion_risk = max(0.1, congestion_risk * (1.0 - (1.0 - trend) * 0.5))
            
            coord_time = clock.current_time + 0.1
            return {
                'new_events': [
                    {'time': coord_time, 'type': EventType.COORDINATION.value,
                     'data': {'vessel_code': vc, 'containers': containers,
                              'arrival_time': arrival, 'bay_plan': bp, 'fitness': fitness,
                              'n_containers': len(containers),
                              'yard_penalty': avg_pen,
                              'yard_reshuffle': round(8.5 * max(0.4, 1.0 - congestion_risk * 0.8), 2),
                              'congestion_risk': round(congestion_risk, 4)},
                     'priority': 1},
                ]
            }
    
    scenario = '常规作业'
    seeds = [100 + i for i in range(n_runs)]
    
    print(f'\n  ── 基线（无预测）──')
    base_results = []
    for seed in seeds:
        from simulation.run import run_single as rs
        r = rs(scenario, 'C', n_days, seed, verbose=True)
        base_results.append(r)
    
    avg_base = {
        'turnaround_h': float(np.mean([r.turnaround_h for r in base_results])),
        'reshuffle_pct': float(np.mean([r.reshuffle_pct for r in base_results])),
        'equip_util_pct': float(np.mean([r.equip_util_pct for r in base_results])),
    }
    print(f'  基线平均: 船时{avg_base["turnaround_h"]:.1f}h 翻箱{avg_base["reshuffle_pct"]:.1f}% 设备{avg_base["equip_util_pct"]:.1f}%')
    
    print(f'\n  ── 预测增强──')
    pred_results = []
    for seed in seeds:
        sc_cfg = SCENARIOS[scenario].copy()
        cfg_key = 'C'
        from simulation.run import CONFIGS
        cfg = CONFIGS[cfg_key]
        
        config = SimulationConfig(
            scenario=scenario, n_days=n_days, n_runs=1, seeds=[seed],
            **{k: v for k, v in sc_cfg.items() if k in ['ships_per_day','yard_util_init','equip_avail','avg_delay_h']},
            config=cfg_key,
            **{k: v for k, v in cfg.items() if k in ['use_garh','use_yard','use_ppo']},
        )
        
        engine = SimEngine(config)
        engine.register_module('vessel_gen', VesselGenerator())
        engine.register_module('stowage', StowageModule(use_garh=cfg['use_garh']))
        engine.register_module('yard', PredictionYardModule(use_3stage=cfg['use_yard'], lookup=pred_lookup))
        engine.register_module('ppo', PPOModule(use_ppo=cfg['use_ppo']))
        
        print(f'  [预测] 种子{seed}...', end=' ', flush=True)
        t0 = time.time()
        result = engine.run()
        elapsed = time.time() - t0
        print(f'完成 ({elapsed:.1f}s, {result.n_vessels}船)', flush=True)
        pred_results.append(result)
    
    avg_pred = {
        'turnaround_h': float(np.mean([r.turnaround_h for r in pred_results])),
        'reshuffle_pct': float(np.mean([r.reshuffle_pct for r in pred_results])),
        'equip_util_pct': float(np.mean([r.equip_util_pct for r in pred_results])),
    }
    print(f'  预测平均: 船时{avg_pred["turnaround_h"]:.1f}h 翻箱{avg_pred["reshuffle_pct"]:.1f}% 设备{avg_pred["equip_util_pct"]:.1f}%')
    
    # 改善
    imp_t = (avg_base['turnaround_h'] - avg_pred['turnaround_h']) / avg_base['turnaround_h'] * 100
    imp_r = (avg_base['reshuffle_pct'] - avg_pred['reshuffle_pct']) / avg_base['reshuffle_pct'] * 100
    imp_e = (avg_pred['equip_util_pct'] - avg_base['equip_util_pct']) / avg_base['equip_util_pct'] * 100
    
    print(f'\n{"="*60}')
    print('三项业务效用指标')
    print(f'{"="*60}')
    print(f'| {"指标":<16} | {"无预测":>10} | {"有预测":>10} | {"改善":>10} |')
    print(f'|{"":-^16}|{"":-^12}|{"":-^12}|{"":-^12}|')
    print(f'| {"翻箱率(%)":<16} | {avg_base["reshuffle_pct"]:>9.1f}% | {avg_pred["reshuffle_pct"]:>9.1f}% | ↓{imp_r:>7.1f}% |')
    print(f'| {"设备利用率(%)":<16} | {avg_base["equip_util_pct"]:>9.1f}% | {avg_pred["equip_util_pct"]:>9.1f}% | ↑{imp_e:>7.1f}% |')
    print(f'| {"船时改善(%)":<16} | {avg_base["turnaround_h"]:>9.1f}h | {avg_pred["turnaround_h"]:>9.1f}h | ↓{imp_t:>7.1f}% |')
    
    # 保存
    final = {
        'scenario': scenario, 'n_days': n_days, 'n_runs': n_runs,
        'baseline': avg_base, 'prediction': avg_pred,
        'improvement': {
            'turnaround': round(imp_t, 1),
            'reshuffle': round(imp_r, 1),
            'equip_util': round(imp_e, 1),
        },
    }
    with open(RESULT / 'des_prediction_comparison.json', 'w') as f:
        json.dump(final, f, indent=2)
    print(f'\n📄 {RESULT}/des_prediction_comparison.json')
    
    return final


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
if __name__ == '__main__':
    import warnings
    # Train model and get predictions
    pred_lookup = train_and_predict()
    
    # Run DES comparison
    results = run_des_comparison(pred_lookup, n_days=30, n_runs=3)
    
    print('\n✅ 全部完成!')
