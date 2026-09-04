"""
Step 16: LSTM多步堆场占用率预测模型（论文第五章上半）
=====================================================
预测目标：未来N天堆场占用率（occupancy）
输入：历史占用率时序 + 靠泊活动 + 停留时间
评估：MAE/MSE/MAPE，两窗walk-forward平均
"""
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings, json, time
warnings.filterwarnings('ignore')

OUT = Path('output')
SPLIT = OUT / 'splits'
RESULT = OUT / 'lstm_results'
RESULT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# ════════════════════════════════════════════
# 1. 数据加载与特征工程
# ════════════════════════════════════════════

def load_features():
    """加载并合并所有日频特征"""
    yf = pd.read_parquet(OUT / '10_yard_features.parquet')
    dwell = pd.read_parquet(OUT / '10_dwell_features.parquet')
    bf = pd.read_parquet(OUT / '10_berth_features.parquet')
    
    # 靠泊特征 -> 每日聚合
    bf['date'] = pd.to_datetime(bf['actual_berth']).dt.date
    daily_berth = bf.groupby('date').agg(
        n_vessels=('berth_plan_no', 'count'),
        total_boxes=('total_boxes', 'sum'),
        avg_duration_h=('actual_duration_h', 'mean'),
        avg_delay_h=('berth_delay_h', 'mean'),
        large_vessels=('is_large', 'sum'),
    ).reset_index()
    
    # 合并
    yf['date'] = pd.to_datetime(yf['day']).dt.date
    dwell['date'] = pd.to_datetime(dwell['day']).dt.date
    
    df = yf.merge(daily_berth, on='date', how='left')
    df = df.merge(dwell, on='date', how='left')
    
    # 填充缺失（无靠泊活动日）
    fill_cols = ['n_vessels', 'total_boxes', 'avg_duration_h', 'avg_delay_h', 'large_vessels']
    for c in fill_cols:
        df[c] = df[c].fillna(0)
    
    # 停留时间特征（dwell在2024年前后有扩展数据）
    dwell_cols = ['containers_arrived', 'dwell_mean_h', 'dwell_median_h', 'dwell_p25_h', 'dwell_p75_h', 'dwell_gt48h_ratio', 'dwell_gt72h_ratio']
    for c in dwell_cols:
        df[c] = df[c].fillna(df[c].median() if c in df else 0)
    
    df = df.sort_values('date').reset_index(drop=True)
    print(f'  合并后: {len(df)}天, {df["date"].min()} ~ {df["date"].max()}')
    return df


def prepare_sequences(df, feature_cols, target_col, lookback=14, horizon=7, stride=1):
    """创建滑动窗口序列"""
    data = df[feature_cols].values
    targets = df[target_col].values
    
    X, y = [], []
    for i in range(0, len(data) - lookback - horizon + 1, stride):
        X.append(data[i:i+lookback])
        y.append(targets[i+lookback:i+lookback+horizon])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ════════════════════════════════════════════
# 2. LSTM模型定义
# ════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    """LSTM多步预测模型"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, horizon)
    
    def forward(self, x):
        # x: (batch, lookback, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, lookback, hidden_dim)
        last_out = lstm_out[:, -1, :]  # 取最后一步
        last_out = self.dropout(last_out)
        return self.fc(last_out)  # (batch, horizon)


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=200, lr=0.001, patience=20, batch_size=32):
    """训练早停"""
    train_loader = DataLoader(TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = np.inf
    best_state = None
    patience_counter = 0
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.FloatTensor(y_val).to(device)
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  早停 @ epoch {epoch}, val_loss={val_loss:.4f}')
                break
        
        if epoch % 50 == 0:
            print(f'  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}')
    
    # 恢复最佳模型
    model.load_state_dict(best_state)
    return model, history


# ════════════════════════════════════════════
# 3. 评估指标
# ════════════════════════════════════════════

def evaluate(y_true, y_pred, horizon, target_name='occupancy'):
    """计算各预测步长的MAE/MSE/MAPE"""
    metrics = {}
    for h in range(horizon):
        true_h = y_true[:, h]
        pred_h = y_pred[:, h]
        mae = np.mean(np.abs(true_h - pred_h))
        mse = np.mean((true_h - pred_h) ** 2)
        mape = np.mean(np.abs((true_h - pred_h) / (true_h + 1))) * 100
        metrics[f'h+{h+1}'] = {'mae': round(mae, 2), 'mse': round(mse, 2), 'mape': round(mape, 2)}
    
    metrics['avg'] = {
        'mae': round(np.mean([m['mae'] for m in metrics.values()]), 2),
        'mse': round(np.mean([m['mse'] for m in metrics.values()]), 2),
        'mape': round(np.mean([m['mape'] for m in metrics.values()]), 2),
    }
    return metrics


# ════════════════════════════════════════════
# 4. 主流程
# ════════════════════════════════════════════

def run():
    print('=' * 60)
    print('LSTM 堆场占用率预测（论文第五章）')
    print('=' * 60)
    
    # 加载数据
    print('\n[1/3] 加载特征...')
    df = load_features()
    
    # 选择特征和目标
    feature_cols = [
        'occupancy', 'n_vessels', 'total_boxes', 'avg_duration_h', 'avg_delay_h',
        'containers_arrived', 'dwell_mean_h', 'dwell_gt72h_ratio',
        'weekday', 'month',
    ]
    target_col = 'occupancy'
    
    print(f'  特征: {feature_cols}')
    print(f'  目标: {target_col}')
    
    lookback, horizon = 14, 7  # 用14天预测未来7天
    all_results = {}
    
    # 两窗训练
    windows = {'w1': (0, 244, 'Jan-Jun→Jul/Aug'), 
               'w2': (0, 366, 'Jan-Sep→Oct/Dec')}
    
    for win_name, (train_end, val_end, desc) in windows.items():
        print(f'\n[2/3] {win_name}: {desc}')
        
        train_df = df.iloc[:train_end] if win_name == 'w1' else df.iloc[:train_end+92]
        # W1: train=182, test=62 (total 244)
        # W2: train=274, test=92 (total 366)
        if win_name == 'w1':
            train_split = 182
        else:
            train_split = 274
        
        train_data = df.iloc[:train_split]
        test_data = df.iloc[train_split:train_split+92] if win_name == 'w2' else df.iloc[train_split:244]
        
        print(f'  Train: {len(train_data)}天 ({train_data["date"].min()}~{train_data["date"].max()})')
        print(f'  Test:  {len(test_data)}天 ({test_data["date"].min()}~{test_data["date"].max()})')
        
        # 准备序列
        X_train, y_train = prepare_sequences(train_data, feature_cols, target_col, lookback, horizon)
        X_test, y_test = prepare_sequences(test_data, feature_cols, target_col, lookback, horizon)
        
        print(f'  X_train: {X_train.shape}, y_train: {y_train.shape}')
        print(f'  X_test:  {X_test.shape}, y_test: {y_test.shape}')
        
        if len(X_train) < 10 or len(X_test) < 2:
            print(f'  ⏭ 数据不足，跳过')
            continue
        
        # 标准化
        train_mean = X_train.mean(axis=(0, 1), keepdims=True)
        train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train - train_mean) / train_std
        X_test_norm = (X_test - train_mean) / train_std
        
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8
        y_train_norm = (y_train - y_mean) / y_std
        y_test_norm = (y_test - y_mean) / y_std
        
        # 验证集（从训练集末尾取20%）
        val_size = max(10, int(len(X_train_norm) * 0.2))
        X_val = X_train_norm[-val_size:]
        y_val = y_train_norm[-val_size:]
        X_train_final = X_train_norm[:-val_size]
        y_train_final = y_train_norm[:-val_size]
        
        # 构建模型
        model = LSTMPredictor(
            input_dim=len(feature_cols),
            hidden_dim=64,
            num_layers=2,
            horizon=horizon,
        ).to(device)
        
        print(f'  模型参数: {sum(p.numel() for p in model.parameters()):,}')
        
        # 训练
        t0 = time.time()
        model, history = train_model(model, X_train_final, y_train_final,
                                      X_val, y_val, epochs=300, patience=30)
        elapsed = time.time() - t0
        print(f'  训练完成: {elapsed:.0f}s')
        
        # 预测
        model.eval()
        with torch.no_grad():
            pred_norm = model(torch.FloatTensor(X_test_norm).to(device)).cpu().numpy()
            pred = pred_norm * y_std + y_mean
            true = y_test
        
        # 评估
        metrics = evaluate(true, pred, horizon)
        print(f'  预测结果: MAE={metrics["avg"]["mae"]:.0f}, MSE={metrics["avg"]["mse"]:.0f}, MAPE={metrics["avg"]["mape"]:.1f}%')
        
        all_results[win_name] = {
            'metrics': metrics,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'elapsed_s': elapsed,
            'y_mean': float(y_mean),
            'y_std': float(y_std),
        }
    
    # ════════════════════════════════════════════
    # 5. 汇总输出
    # ════════════════════════════════════════════
    
    print('\n[3/3] 结果汇总')
    print('=' * 60)
    
    for win_name, res in all_results.items():
        m = res['metrics']
        print(f'\n{win_name}:')
        print(f'  平均 MAE={m["avg"]["mae"]}, MSE={m["avg"]["mse"]}, MAPE={m["avg"]["mape"]}%')
        print(f'  按步长:')
        for h in sorted([k for k in m if k.startswith('h+')]):
            print(f'    {h}: MAE={m[h]["mae"]}, MAPE={m[h]["mape"]}%')
    
    # 两窗平均
    if len(all_results) >= 2:
        m1, m2 = all_results['w1']['metrics'], all_results['w2']['metrics']
        avg_mae = (m1['avg']['mae'] + m2['avg']['mae']) / 2
        avg_mse = (m1['avg']['mse'] + m2['avg']['mse']) / 2
        avg_mape = (m1['avg']['mape'] + m2['avg']['mape']) / 2
        print(f'\n两窗平均: MAE={avg_mae:.2f}, MSE={avg_mse:.2f}, MAPE={avg_mape:.2f}%')
        
        all_results['avg'] = {'mae': avg_mae, 'mse': avg_mse, 'mape': avg_mape}
    
    # 保存
    with open(RESULT / 'lstm_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n📄 结果保存: {RESULT}/lstm_results.json')


if __name__ == '__main__':
    run()
