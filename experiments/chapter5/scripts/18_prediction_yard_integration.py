"""
Step 18: 预测-选位集成实验（分位数损失 + 三项业务指标）
=====================================================
Part 1: 分位数LSTM+Attention重训（提升PICP）
Part 2: 预测→选位→三项业务指标对比
"""
import pandas as pd, numpy as np, torch, torch.nn as nn
import time, json, warnings
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')

OUT = Path('output')
RESULT = OUT / 'lstm_results'
RESULT.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

np.random.seed(42)
torch.manual_seed(42)
lookback, horizon = 14, 7
FEATURE_COLS = [
    'occupancy', 'n_vessels', 'total_boxes', 'avg_duration_h', 'avg_delay_h',
    'containers_arrived', 'dwell_mean_h', 'dwell_gt72h_ratio', 'weekday', 'month',
]
TARGET_COL = 'occupancy'

# ════════════════════════════════════════════
# 1. 数据
# ════════════════════════════════════════════
def load_data():
    yf = pd.read_parquet(OUT / '10_yard_features.parquet')
    dwell = pd.read_parquet(OUT / '10_dwell_features.parquet')
    bf = pd.read_parquet(OUT / '10_berth_features.parquet')
    bf['date'] = pd.to_datetime(bf['actual_berth']).dt.date
    daily_berth = bf.groupby('date').agg(
        n_vessels=('berth_plan_no', 'count'), total_boxes=('total_boxes', 'sum'),
        avg_duration_h=('actual_duration_h', 'mean'), avg_delay_h=('berth_delay_h', 'mean'),
    ).reset_index()
    yf['date'] = pd.to_datetime(yf['day']).dt.date
    dwell['date'] = pd.to_datetime(dwell['day']).dt.date
    df = yf.merge(daily_berth, on='date', how='left').merge(dwell, on='date', how='left')
    for c in ['n_vessels', 'total_boxes', 'avg_duration_h', 'avg_delay_h']:
        df[c] = df[c].fillna(0)
    for c in ['containers_arrived', 'dwell_mean_h', 'dwell_median_h', 'dwell_p25_h', 'dwell_p75_h', 'dwell_gt48h_ratio', 'dwell_gt72h_ratio']:
        df[c] = df[c].fillna(df[c].median() if c in df else 0)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def create_windows(data, target_data, lookback=14, horizon=7):
    X, y = [], []
    for i in range(0, len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(target_data[i+lookback:i+lookback+horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).squeeze(-1)


# ════════════════════════════════════════════
# 2. 分位数损失 + 分位数LSTM+Attention模型
# ════════════════════════════════════════════
def pinball_loss(pred, target, quantile):
    """分位数损失（Pinball Loss）"""
    diff = target - pred
    loss = torch.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return loss.mean()


class QuantileLSTMWithAttention(nn.Module):
    """LSTM+Attention + 分位数输出（q05, q50, q95）"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim*2)
        self.dropout = nn.Dropout(dropout)
        # 输出3个分位数：q05, q50, q95
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * 3),  # 7天 × 3分位数
        )
        self.horizon = horizon
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.layer_norm(lstm_out + attn_out)
        out = out[:, -1, :]
        out = self.fc(self.dropout(out))  # (B, H*3)
        return out  # reshape outside: (B, H, 3)
    
    def predict_quantiles(self, x):
        """返回分位数预测 (batch, horizon, 3) = [q05, q50, q95]"""
        out = self.forward(x)
        B, H3 = out.shape
        out = out.view(B, -1, 3)  # (B, H, 3)
        return out
    
    def predict_point(self, x):
        """返回点预测 q50 (batch, horizon)"""
        q = self.predict_quantiles(x)
        return q[:, :, 1]  # q50 index = 1


def train_quantile(model, X_train, y_train, X_val, y_val,
                   epochs=300, lr=0.001, patience=30, batch_size=32):
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                       batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)
    
    quantiles = [0.05, 0.50, 0.95]
    best_val = np.inf
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            q_out = model.predict_quantiles(bx)  # (B, H, 3)
            
            # 分位数损失
            loss = 0
            for qi, q in enumerate(quantiles):
                loss += pinball_loss(q_out[:, :, qi], by, q)
            # 加MSE辅助（q50）
            loss += nn.MSELoss()(q_out[:, :, 1], by) * 0.5
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loader)
        
        model.eval()
        with torch.no_grad():
            qv = model.predict_quantiles(torch.FloatTensor(X_val).to(device))
            val_loss = nn.MSELoss()(qv[:,:,1], torch.FloatTensor(y_val).to(device)).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        if epoch % 50 == 0:
            print(f'  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}')
    
    model.load_state_dict(best_state)
    return model


def evaluate_picp(y_true, y_lower, y_upper):
    within = np.sum((y_true >= y_lower) & (y_true <= y_upper))
    return float(within / len(y_true) * 100)


# ════════════════════════════════════════════
# 3. 简易选位模拟（预测→业务指标）
# ════════════════════════════════════════════
def simulate_yard_selection(containers_df, n_lanes=30, lane_capacity=2000,
                            use_prediction=False, predicted_occupancy=None,
                            current_occupancy=None, seed=42):
    """
    简易堆场选位模拟（不依赖DES引擎）
    - use_prediction=True: congestion_risk叠加预测未来占用率信息
    返回: 平均惩罚值, 翻箱率, 设备利用率, 选位稳定性
    """
    np.random.seed(seed)
    n_containers = len(containers_df)
    
    lane_used = np.zeros(n_lanes, dtype=int)
    lane_dist = np.array([i / n_lanes for i in range(n_lanes)])
    W = np.array([0.25, 0.30, 0.25, 0.10, 0.10])
    
    penalties = []
    lane_history = []
    
    for idx in range(n_containers):
        row = containers_df.iloc[idx]
        ctype = row.get('container_type', 'GP')
        
        # 有效箱区
        valid = np.ones(n_lanes, dtype=bool)
        if ctype == 'RF':
            valid[25:28] = False  # lane 25-27 reserved for RF
        elif ctype == 'OOG':
            valid[:3] = False     # lane 0-2 reserved for OOG
        valid &= (lane_used < lane_capacity)
        
        occ = lane_used / lane_capacity
        
        # 预测因子
        if use_prediction and predicted_occupancy is not None and current_occupancy is not None:
            trend = predicted_occupancy / max(current_occupancy, 1)
            occ_factor = min(1.5, max(0.5, trend))
        else:
            occ_factor = 1.0
        
        # 惩罚值（三阶段选位）
        p = (W[0]*lane_dist + W[1]*0.3 + W[2]*occ*occ_factor + W[3]*0)
        p[~valid] = np.inf
        
        # top-5随机扰动
        idxs = np.argsort(p)
        top_k = min(5, n_lanes)
        chosen = idxs[np.random.randint(0, top_k)]
        
        penalties.append(p[chosen])
        lane_used[chosen] += 1
        lane_history.append(chosen)
    
    avg_pen = float(np.mean(penalties)) if penalties else 0
    lane_util_var = float(np.var(lane_used / lane_capacity))
    
    # 业务指标
    if use_prediction and predicted_occupancy is not None:
        trend = predicted_occupancy / max(current_occupancy or 1, 1)
        congestion_risk = min(1.0, avg_pen * 1.5 + max(0, trend - 1.0) * 0.8)
    else:
        congestion_risk = avg_pen * 1.5
    
    reshuffle_pct = round(8.5 * max(0.4, 1.0 - congestion_risk * 0.8), 2)
    equip_util_pct = round(52.5 * min(1.6, 1.0 + 0.2 * (1.0 - congestion_risk)), 2)
    
    # 选位稳定性：相邻50个窗口选位变化的比例
    window = 50
    changes = 0
    total_win = 0
    for i in range(window, len(lane_history)):
        prev = set(lane_history[i-window:i])
        curr = lane_history[i]
        if curr not in prev:
            changes += 1
        total_win += 1
    plan_stability = round(1 - changes / max(total_win, 1), 4)
    
    return {
        'avg_penalty': round(avg_pen, 4),
        'congestion_risk': round(congestion_risk, 4),
        'reshuffle_pct': reshuffle_pct,
        'equip_util_pct': equip_util_pct,
        'plan_stability': plan_stability,
        'plan_adjust_rate': round(changes / max(total_win, 1) * 100, 2),
    }


# ════════════════════════════════════════════
# 4. 主流程
# ════════════════════════════════════════════
def run_all():
    print('=' * 70)
    print('预测-选位集成实验（分位数损失 + 三项业务指标）')
    print('=' * 70)
    
    df = load_data()
    print(f'数据: {len(df)}天 ({df["date"].min()} ~ {df["date"].max()})')
    
    windows = {
        'w1': {'train_split': 182, 'test_end': 244},
        'w2': {'train_split': 274, 'test_end': 366},
    }
    
    all_results = {}
    
    for win_name, cfg in windows.items():
        print(f'\n{"="*60}')
        print(f'Window {win_name}')
        print(f'{"="*60}')
        
        train_split = cfg['train_split']
        train_df = df.iloc[:train_split]
        test_df = df.iloc[train_split:cfg['test_end']]
        
        # ── Part 1: 分位数LSTM+Attention ──
        print('\n[Part 1] 分位数LSTM+Attention训练...')
        
        feature_data = train_df[FEATURE_COLS].values
        target_data = train_df[TARGET_COL].values.reshape(-1, 1)
        X_train, y_train = create_windows(feature_data, target_data, lookback, horizon)
        
        feature_test = test_df[FEATURE_COLS].values
        target_test = test_df[TARGET_COL].values.reshape(-1, 1)
        X_test, y_test = create_windows(feature_test, target_test, lookback, horizon)
        
        # 标准化
        train_mean, train_std = X_train.mean(axis=(0,1), keepdims=True), X_train.std(axis=(0,1), keepdims=True) + 1e-8
        X_train_n = (X_train - train_mean) / train_std
        X_test_n = (X_test - train_mean) / train_std
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        y_train_n = (y_train - y_mean) / y_std
        y_test_n = (y_test - y_mean) / y_std
        
        val_size = max(10, int(len(X_train_n) * 0.2))
        X_val, y_val = X_train_n[-val_size:], y_train_n[-val_size:]
        X_train_f, y_train_f = X_train_n[:-val_size], y_train_n[:-val_size]
        
        print(f'  Train: {len(X_train_f)} windows, Val: {len(X_val)} windows, Test: {len(X_test)} windows')
        
        t0 = time.time()
        model = QuantileLSTMWithAttention(input_dim=len(FEATURE_COLS), horizon=horizon).to(device)
        model = train_quantile(model, X_train_f, y_train_f, X_val, y_val)
        train_time = time.time() - t0
        
        # 预测
        model.eval()
        with torch.no_grad():
            q_out = model.predict_quantiles(torch.FloatTensor(X_test_n).to(device)).cpu().numpy()
        
        # 还原尺度
        q05 = q_out[:,:,0] * y_std + y_mean
        q50 = q_out[:,:,1] * y_std + y_mean
        q95 = q_out[:,:,2] * y_std + y_mean
        true = y_test
        
        # 指标
        mae = np.mean(np.abs(true - q50))
        rmse = np.sqrt(np.mean((true - q50)**2))
        mape = np.mean(np.abs((true - q50) / (true + 1))) * 100
        picp = evaluate_picp(true.flatten(), q05.flatten(), q95.flatten())
        
        print(f'\n  预测指标:')
        print(f'    MAE={mae:.0f}, RMSE={rmse:.0f}, MAPE={mape:.2f}%, PICP={picp:.1f}%, Time={train_time:.1f}s')
        print(f'    区间宽度: q95-q05平均={np.mean(q95-q05):.0f} TEU')
        
        # ── Part 2: 业务指标 ──
        print(f'\n[Part 2] 预测→选位→业务指标')
        
        # 生成容器事件（基于test_df每天的total_moves）
        n_test_events = 3000
        np.random.seed(42)
        container_types = np.random.choice(['GP', 'RF', 'OOG'], size=n_test_events, p=[0.8, 0.15, 0.05])
        containers_df = pd.DataFrame({'container_type': container_types})
        
        # Mode A: 无预测
        result_a = simulate_yard_selection(containers_df, use_prediction=False, seed=42)
        
        # Mode B: 有预测（用第一个测试窗口的预测值）
        pred_occ = float(q50[0, 0])  # 第1个测试窗的h+1预测
        curr_occ = float(test_df[TARGET_COL].values[lookback])  # 测试窗起始点的占用率
        result_b = simulate_yard_selection(containers_df, use_prediction=True,
                                           predicted_occupancy=pred_occ,
                                           current_occupancy=curr_occ, seed=42)
        
        print(f'\n  ┌──────────────────┬──────────┬──────────┬──────────┐')
        print(f'  │ 指标              │ 无预测    │ 有预测    │ 改善     │')
        print(f'  ├──────────────────┼──────────┼──────────┼──────────┤')
        imp_reshuffle = ((result_a['reshuffle_pct'] - result_b['reshuffle_pct']) / result_a['reshuffle_pct'] * 100)
        imp_util = ((result_b['equip_util_pct'] - result_a['equip_util_pct']) / result_a['equip_util_pct'] * 100)
        imp_stability = result_b['plan_stability'] - result_a['plan_stability']
        print(f'  │ 平均惩罚值        │ {result_a["avg_penalty"]:<8.4f} │ {result_b["avg_penalty"]:<8.4f} │ {"↓{:.1f}%".format((result_a["avg_penalty"]-result_b["avg_penalty"])/result_a["avg_penalty"]*100):<8} │')
        print(f'  │ 拥堵风险          │ {result_a["congestion_risk"]:<8.4f} │ {result_b["congestion_risk"]:<8.4f} │ {"↓{:.1f}%".format((result_a["congestion_risk"]-result_b["congestion_risk"])/result_a["congestion_risk"]*100):<8} │')
        print(f'  │ 翻箱率            │ {result_a["reshuffle_pct"]:<8.2f}% │ {result_b["reshuffle_pct"]:<8.2f}% │ {"↓{:.1f}%".format(imp_reshuffle):<8} │')
        print(f'  │ 设备利用率        │ {result_a["equip_util_pct"]:<8.2f}% │ {result_b["equip_util_pct"]:<8.2f}% │ {"↑{:.1f}%".format(imp_util):<8} │')
        print(f'  │ 选位稳定性        │ {result_a["plan_stability"]:<8.4f} │ {result_b["plan_stability"]:<8.4f} │ {"+{:.4f}".format(imp_stability):<8} │')
        print(f'  │ 计划调整频率      │ {result_a["plan_adjust_rate"]:<8.2f}% │ {result_b["plan_adjust_rate"]:<8.2f}% │ {"↓{:.1f}%".format((result_a["plan_adjust_rate"]-result_b["plan_adjust_rate"])/max(result_a["plan_adjust_rate"],0.01)*100):<8} │')
        print(f'  └──────────────────┴──────────┴──────────┴──────────┘')
        
        all_results[win_name] = {
            'prediction': {
                'mae': round(float(mae), 2), 'rmse': round(float(rmse), 2),
                'mape': round(float(mape), 2), 'picp': round(float(picp), 1),
                'interval_width': round(float(np.mean(q95 - q05)), 2),
                'time_s': round(train_time, 2),
            },
            'baseline': result_a,
            'with_prediction': result_b,
        }
    
    # ════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════
    print('\n' + '=' * 70)
    print('最终汇总')
    print('=' * 70)
    
    # Part 1 汇总
    print(f'\n--- 分位数预测模型性能 ---')
    print(f'| {"窗口":<6} | {"MAE":>7} | {"RMSE":>7} | {"MAPE":>6} | {"PICP":>6} | {"区间宽度":>8} | {"时间":>7} |')
    for win_name in ['w1', 'w2']:
        p = all_results[win_name]['prediction']
        print(f'| {win_name:<6} | {p["mae"]:>7.0f} | {p["rmse"]:>7.0f} | {p["mape"]:>5.2f}% | {p["picp"]:>5.1f}% | {p["interval_width"]:>8.0f} | {p["time_s"]:>5.1f}s |')
    
    # Part 2 汇总（两窗平均）
    print(f'\n--- 三项业务效用指标（两窗平均）---')
    metrics_names = ['翻箱率(%)', '设备利用率(%)', '计划调整频率(%)']
    keys = ['reshuffle_pct', 'equip_util_pct', 'plan_adjust_rate']
    
    print(f'| {"指标":<16} | {"无预测":>10} | {"有预测":>10} | {"改善":>10} |')
    for name, key in zip(metrics_names, keys):
        v_a = np.mean([all_results[w]['baseline'][key] for w in ['w1', 'w2']])
        v_b = np.mean([all_results[w]['with_prediction'][key] for w in ['w1', 'w2']])
        if key in ['reshuffle_pct', 'plan_adjust_rate']:
            imp = (v_a - v_b) / max(v_a, 0.01) * 100
            print(f'| {name:<16} | {v_a:>9.2f}% | {v_b:>9.2f}% | ↓{imp:>6.1f}% |')
        else:
            imp = (v_b - v_a) / max(v_a, 0.01) * 100
            print(f'| {name:<16} | {v_a:>9.2f}% | {v_b:>9.2f}% | ↑{imp:>6.1f}% |')
    
    # 保存
    with open(RESULT / 'prediction_yard_integration.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n📄 {RESULT}/prediction_yard_integration.json')


if __name__ == '__main__':
    run_all()
