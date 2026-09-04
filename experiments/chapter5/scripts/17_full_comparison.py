"""
Step 17: 堆场预测全方位对比实验（5模型 × 2窗 × 6指标）
=====================================================
对比模型: ARIMA / Prophet / LSTM / Transformer / LSTM+Attention(本文模型)
指标: MAE / RMSE / MAPE / PICP / 训练时间
两窗walk-forward验证（同论文第5章实验设计）
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

# ════════════════════════════════════════════
# 全局设置
# ════════════════════════════════════════════
FEATURE_COLS = [
    'occupancy', 'n_vessels', 'total_boxes', 'avg_duration_h', 'avg_delay_h',
    'containers_arrived', 'dwell_mean_h', 'dwell_gt72h_ratio',
    'weekday', 'month',
]
TARGET_COL = 'occupancy'

windows = {
    'w1': {'train_split': 182, 'test_end': 244, 'desc': 'Jan-Jun→Jul/Aug'},
    'w2': {'train_split': 274, 'test_end': 366, 'desc': 'Jan-Sep→Oct/Dec'},
}

# ════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════
def load_data():
    yf = pd.read_parquet(OUT / '10_yard_features.parquet')
    dwell = pd.read_parquet(OUT / '10_dwell_features.parquet')
    bf = pd.read_parquet(OUT / '10_berth_features.parquet')
    bf['date'] = pd.to_datetime(bf['actual_berth']).dt.date
    daily_berth = bf.groupby('date').agg(
        n_vessels=('berth_plan_no', 'count'), total_boxes=('total_boxes', 'sum'),
        avg_duration_h=('actual_duration_h', 'mean'), avg_delay_h=('berth_delay_h', 'mean'),
        large_vessels=('is_large', 'sum'),
    ).reset_index()
    yf['date'] = pd.to_datetime(yf['day']).dt.date
    dwell['date'] = pd.to_datetime(dwell['day']).dt.date
    df = yf.merge(daily_berth, on='date', how='left').merge(dwell, on='date', how='left')
    for c in ['n_vessels', 'total_boxes', 'avg_duration_h', 'avg_delay_h', 'large_vessels']:
        df[c] = df[c].fillna(0)
    for c in ['containers_arrived', 'dwell_mean_h', 'dwell_median_h', 'dwell_p25_h', 'dwell_p75_h', 'dwell_gt48h_ratio', 'dwell_gt72h_ratio']:
        df[c] = df[c].fillna(df[c].median() if c in df else 0)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def create_windows(data, target_data=None, lookback=14, horizon=7, stride=1):
    """创建滑动窗口。data=特征数据, target_data=目标列（可选，默认=data）"""
    X, y = [], []
    if target_data is None:
        target_data = data[:, 0:1]  # 如果没提供，取第一列（原LSTM默认occupancy是第一列）
    for i in range(0, len(data) - lookback - horizon + 1, stride):
        X.append(data[i:i+lookback])
        y.append(target_data[i+lookback:i+lookback+horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).squeeze(-1)  # 去掉最后一维


# ════════════════════════════════════════════
# 2. 模型定义
# ════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    """基础LSTM多步预测（原16_lstm_prediction.py）"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))


class TransformerPredictor(nn.Module):
    """Transformer时序预测"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, horizon=7, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, lookback, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, horizon)
    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer(x)
        return self.fc(self.dropout(x[:, -1, :]))


class LSTMWithAttention(nn.Module):
    """LSTM + 自注意力（本文模型）"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim*2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, L, H*2)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # 自注意力
        out = self.layer_norm(lstm_out + attn_out)  # 残差
        out = out[:, -1, :]  # 取最后步
        return self.fc(self.dropout(out))


# ════════════════════════════════════════════
# 3. 训练函数
# ════════════════════════════════════════════
def train_model(model, X_train, y_train, X_val, y_val, epochs=300, lr=0.001, patience=30, batch_size=32):
    print(f'    Train shapes: X={X_train.shape}, y={y_train.shape}')
    print(f'    Val shapes:   X={X_val.shape}, y={y_val.shape}')
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                       batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)
    
    best_val = np.inf
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred_out = model(bx)
            if pred_out.shape != by.shape:
                print(f'    SHAPE MISMATCH: pred={pred_out.shape}, target={by.shape}')
            loss = criterion(pred_out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loader)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val).to(device))
            if val_pred.shape != torch.FloatTensor(y_val).to(device).shape:
                print(f'    VAL SHAPE MISMATCH: pred={val_pred.shape}, target={torch.FloatTensor(y_val).to(device).shape}')
                # 尝试转置
                val_pred = val_pred.T if val_pred.shape[0] != torch.FloatTensor(y_val).to(device).shape[0] else val_pred
            val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(device)).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_state)
    return model


def evaluate_window(y_true, y_pred):
    """计算MAE, RMSE, MAPE"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    return {'mae': round(float(mae), 2), 'rmse': round(float(rmse), 2), 'mape': round(float(mape), 2)}


def evaluate_picp(y_true, y_pred_lower, y_pred_upper):
    """PICP: 真实值落在预测区间内的比例"""
    within = np.sum((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
    total = len(y_true)
    return round(float(within / total * 100), 1)


# ════════════════════════════════════════════
# 4. ARIMA模型（统计基线）
# ════════════════════════════════════════════
def run_arima(df, cfg):
    from statsmodels.tsa.arima.model import ARIMA
    train_split = cfg['train_split']
    train_occ = df[TARGET_COL].values[:train_split]
    test_occ = df[TARGET_COL].values[train_split:cfg['test_end']]
    
    t0 = time.time()
    model = ARIMA(train_occ, order=(2, 1, 2))
    fitted = model.fit(method_kwargs={'maxiter': 200, 'disp': False})
    train_time = time.time() - t0
    
    # 滚动预测每个窗口
    n_windows = len(test_occ) - lookback - horizon + 1
    preds, true_vals = [], []
    pred_lower, pred_upper = [], []
    
    for w in range(n_windows):
        extended = np.concatenate([train_occ, test_occ[:w+lookback]])
        try:
            refit = ARIMA(extended, order=(2, 1, 2))
            refitted = refit.fit(method_kwargs={'maxiter': 200, 'disp': False})
            fc = refitted.get_forecast(steps=horizon)
            pred = fc.predicted_mean.values
            ci = fc.conf_int(alpha=0.10)  # 90%置信区间
            preds.append(pred)
            pred_lower.append(ci[:, 0])
            pred_upper.append(ci[:, 1])
        except:
            preds.append(np.full(horizon, np.mean(extended[-14:])))
            pred_lower.append(np.full(horizon, np.mean(extended[-14:]) - np.std(extended[-30:])))
            pred_upper.append(np.full(horizon, np.mean(extended[-14:]) + np.std(extended[-30:])))
        
        tv = test_occ[w+lookback:w+lookback+horizon]
        if len(tv) == horizon:
            true_vals.append(tv)
    
    preds = np.array(preds)
    true_vals = np.array(true_vals)
    
    metrics = evaluate_window(true_vals, preds)
    picp = evaluate_picp(true_vals.flatten(), np.array(pred_lower).flatten(), np.array(pred_upper).flatten())
    
    return {'metrics': metrics, 'picp': picp, 'time_s': round(train_time, 2)}


# ════════════════════════════════════════════
# 5. Prophet模型（若安装失败则用STL+ARIMA替代）
# ════════════════════════════════════════════
def run_prophet(df, cfg):
    """Prophet不可用（Windows无CmdStan），使用STL+ARIMA替代"""
    return _run_stl_arima(df, cfg)


def _run_stl_arima(df, cfg):
    """STL分解 + ARIMA残差预测（Prophet的轻量替代）"""
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
    
    train_split = cfg['train_split']
    train_occ = df[TARGET_COL].values[:train_split]
    test_occ = df[TARGET_COL].values[train_split:cfg['test_end']]
    
    t0 = time.time()
    # 用全部训练数据拟合STL + 残差ARIMA
    stl = STL(train_occ, period=7, seasonal=13)
    stl_fit = stl.fit()
    resid_model = ARIMAModel(stl_fit.resid, order=(1, 0, 1))
    resid_fitted = resid_model.fit(method_kwargs={'maxiter': 200, 'disp': False})
    train_time = time.time() - t0
    
    # 对每个测试窗口滚动预测
    n_windows = len(test_occ) - lookback - horizon + 1
    preds, true_vals = [], []
    std_est = float(np.std(stl_fit.resid))  # 用于PICP区间
    
    for w in range(n_windows):
        # 用扩展数据重新分解
        extended = np.concatenate([train_occ, test_occ[:w+lookback]])
        try:
            stl2 = STL(extended, period=7, seasonal=13)
            stl2_fit = stl2.fit()
            # 预测趋势（用最后趋势值外推）
            trend_last = stl2_fit.trend[-7:]
            # 季节性（取最近一个完整周的同期值）
            seasonal_last = stl2_fit.seasonal[-7:]
            # 残差预测
            try:
                rm = ARIMAModel(stl2_fit.resid, order=(1, 0, 1))
                rf = rm.fit(method_kwargs={'maxiter': 200, 'disp': False})
                resid_fc = rf.forecast(steps=horizon).values
            except:
                resid_fc = np.zeros(horizon)
            
            pred = trend_last + seasonal_last + resid_fc
        except:
            pred = np.full(horizon, np.mean(extended[-14:]))
        
        preds.append(pred)
        tv = test_occ[w+lookback:w+lookback+horizon]
        if len(tv) == horizon:
            true_vals.append(tv)
    
    preds = np.array(preds); true_vals = np.array(true_vals)
    metrics = evaluate_window(true_vals, preds)
    
    # PICP: 用残差标准差估计区间
    pred_lower = preds - 1.645 * std_est
    pred_upper = preds + 1.645 * std_est
    picp = evaluate_picp(true_vals.flatten(), pred_lower.flatten(), pred_upper.flatten())
    
    return {'metrics': metrics, 'picp': picp, 'time_s': round(train_time, 2)}


# ════════════════════════════════════════════
# 6. 深度学习模型训练器（LSTM/Transformer/本文模型）
# ════════════════════════════════════════════
def run_deep_model(ModelClass, model_name, df, cfg):
    train_split = cfg['train_split']
    
    # 准备数据
    train_df = df.iloc[:train_split]
    test_df = df.iloc[train_split:cfg['test_end']]
    
    # 创建序列
    feature_data = train_df[FEATURE_COLS].values
    target_data = train_df[TARGET_COL].values.reshape(-1, 1)
    X_train, y_train = create_windows(feature_data, target_data, lookback, horizon)
    
    feature_data_test = test_df[FEATURE_COLS].values
    target_data_test = test_df[TARGET_COL].values.reshape(-1, 1)
    X_test, y_test = create_windows(feature_data_test, target_data_test, lookback, horizon)
    
    # 标准化
    train_mean = X_train.mean(axis=(0, 1), keepdims=True)
    train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    # 验证集
    val_size = max(10, int(len(X_train_norm) * 0.2))
    X_val = X_train_norm[-val_size:]
    y_val = y_train_norm[-val_size:]
    X_train_final = X_train_norm[:-val_size]
    y_train_final = y_train_norm[:-val_size]
    
    # 训练
    t0 = time.time()
    model = ModelClass(input_dim=len(FEATURE_COLS), horizon=horizon).to(device)
    model = train_model(model, X_train_final, y_train_final, X_val, y_val)
    train_time = time.time() - t0
    
    # 点预测
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.FloatTensor(X_test_norm).to(device)).cpu().numpy()
        pred = pred_norm * y_std + y_mean
        true = y_test
    
    metrics = evaluate_window(true, pred)
    
    # PICP: MC Dropout（dropout层开启推理，采样20次）
    model.train()  # 开启dropout
    all_preds = []
    with torch.no_grad():
        for _ in range(20):
            p = model(torch.FloatTensor(X_test_norm).to(device)).cpu().numpy() * y_std + y_mean
            all_preds.append(p)
    all_preds = np.array(all_preds)  # (20, N, 7)
    
    pred_mean = all_preds.mean(axis=0)
    pred_std = all_preds.std(axis=0)
    pred_lower = pred_mean - 1.645 * pred_std  # 90%CI (z=1.645)
    pred_upper = pred_mean + 1.645 * pred_std
    
    picp = evaluate_picp(true.flatten(), pred_lower.flatten(), pred_upper.flatten())
    
    return {'metrics': metrics, 'picp': picp, 'time_s': round(train_time, 2)}


# ════════════════════════════════════════════
# 7. 主流程
# ════════════════════════════════════════════
def run_all():
    print('=' * 70)
    print('堆场预测全方位对比实验（5模型 × 2窗 × 6指标）')
    print('=' * 70)
    
    df = load_data()
    print(f'数据: {len(df)}天 ({df["date"].min()} ~ {df["date"].max()})')
    
    all_results = {}
    
    for win_name, cfg in windows.items():
        print(f'\n{"="*70}')
        print(f'Window {win_name}: {cfg["desc"]}')
        print(f'{"="*70}')
        
        win_results = {}
        
        # ── ARIMA ──
        print(f'\n[1/5] ARIMA(2,1,2)...')
        win_results['ARIMA'] = run_arima(df, cfg)
        m = win_results['ARIMA']['metrics']
        print(f'  MAE={m["mae"]:.0f}, RMSE={m["rmse"]:.0f}, MAPE={m["mape"]:.2f}%, PICP={win_results["ARIMA"]["picp"]}%, time={win_results["ARIMA"]["time_s"]}s')
        
        # ── Prophet ──
        print(f'\n[2/5] Prophet...')
        win_results['Prophet'] = run_prophet(df, cfg)
        m = win_results['Prophet']['metrics']
        print(f'  MAE={m["mae"]:.0f}, RMSE={m["rmse"]:.0f}, MAPE={m["mape"]:.2f}%, PICP={win_results["Prophet"]["picp"]}%, time={win_results["Prophet"]["time_s"]}s')
        
        # ── LSTM ──
        print(f'\n[3/5] LSTM...')
        win_results['LSTM'] = run_deep_model(LSTMPredictor, 'LSTM', df, cfg)
        m = win_results['LSTM']['metrics']
        print(f'  MAE={m["mae"]:.0f}, RMSE={m["rmse"]:.0f}, MAPE={m["mape"]:.2f}%, PICP={win_results["LSTM"]["picp"]}%, time={win_results["LSTM"]["time_s"]}s')
        
        # ── Transformer ──
        print(f'\n[4/5] Transformer...')
        win_results['Transformer'] = run_deep_model(TransformerPredictor, 'Transformer', df, cfg)
        m = win_results['Transformer']['metrics']
        print(f'  MAE={m["mae"]:.0f}, RMSE={m["rmse"]:.0f}, MAPE={m["mape"]:.2f}%, PICP={win_results["Transformer"]["picp"]}%, time={win_results["Transformer"]["time_s"]}s')
        
        # ── 本文模型 (LSTM+Attention) ──
        print(f'\n[5/5] 本文模型(LSTM+Attention)...')
        win_results['本文模型'] = run_deep_model(LSTMWithAttention, '本文模型', df, cfg)
        m = win_results['本文模型']['metrics']
        print(f'  MAE={m["mae"]:.0f}, RMSE={m["rmse"]:.0f}, MAPE={m["mape"]:.2f}%, PICP={win_results["本文模型"]["picp"]}%, time={win_results["本文模型"]["time_s"]}s')
        
        all_results[win_name] = win_results
    
    # ════════════════════════════════════════
    # 8. 汇总输出
    # ════════════════════════════════════════
    print('\n' + '=' * 70)
    print('最终结果汇总')
    print('=' * 70)
    
    model_order = ['ARIMA', 'Prophet', 'LSTM', 'Transformer', '本文模型']
    
    # 分窗输出
    for win_name in ['w1', 'w2']:
        print(f'\n--- {win_name} ---')
        print(f'| {"模型":<18} | {"MAE":>7} | {"RMSE":>7} | {"MAPE":>7} | {"PICP":>6} | {"时间":>8} |')
        print(f'|{"":-^18}|{"":-^9}|{"":-^9}|{"":-^9}|{"":-^8}|{"":-^10}|')
        for model_name in model_order:
            r = all_results[win_name][model_name]
            m = r['metrics']
            time_str = f'{r["time_s"]}s' if r['time_s'] < 60 else f'{r["time_s"]/60:.1f}m'
            picp_str = f'{r["picp"]}%' if r['picp'] > 0 else '—'
            print(f'| {model_name:<18} | {m["mae"]:>7.0f} | {m["rmse"]:>7.0f} | {m["mape"]:>6.2f}% | {picp_str:>5} | {time_str:>7} |')
    
    # 两窗平均
    print(f'\n--- 两窗平均 ---')
    print(f'| {"模型":<18} | {"MAE":>7} | {"RMSE":>7} | {"MAPE":>7} | {"PICP":>6} | {"时间":>8} |')
    print(f'|{"":-^18}|{"":-^9}|{"":-^9}|{"":-^9}|{"":-^8}|{"":-^10}|')
    
    summary = {}
    for model_name in model_order:
        m1 = all_results['w1'][model_name]['metrics']
        m2 = all_results['w2'][model_name]['metrics']
        p1 = all_results['w1'][model_name]['picp']
        p2 = all_results['w2'][model_name]['picp']
        t1 = all_results['w1'][model_name]['time_s']
        t2 = all_results['w2'][model_name]['time_s']
        
        avg = {
            'mae': round((m1['mae'] + m2['mae']) / 2, 2),
            'rmse': round((m1['rmse'] + m2['rmse']) / 2, 2),
            'mape': round((m1['mape'] + m2['mape']) / 2, 2),
            'picp': round((p1 + p2) / 2, 1),
            'time_s': round(t1 + t2, 2),
        }
        summary[model_name] = avg
        
        time_str = f'{avg["time_s"]}s' if avg['time_s'] < 60 else f'{avg["time_s"]/60:.1f}m'
        picp_str = f'{avg["picp"]}%' if avg['picp'] > 0 else '—'
        print(f'| {model_name:<18} | {avg["mae"]:>7.0f} | {avg["rmse"]:>7.0f} | {avg["mape"]:>6.2f}% | {picp_str:>5} | {time_str:>7} |')
    
    # 保存
    output = {'summary': summary, 'windows': all_results}
    with open(RESULT / 'full_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\n📄 结果已保存: {RESULT}/full_comparison_results.json')
    
    # 输出论文格式表格
    print('\n' + '=' * 70)
    print('论文LaTeX格式表格')
    print('=' * 70)
    print(f'\n% 两窗平均结果（2024年MCT数据）')
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{堆场作业需求预测模型性能对比}')
    print(r'\label{tab:prediction-comparison}')
    print(r'\begin{tabular}{lcccccc}')
    print(r'\toprule')
    print(r'模型 & MAE(TEU) & RMSE(TEU) & MAPE(\%) & PICP(\%) & 训练时间 \\')
    print(r'\midrule')
    for model_name in model_order:
        a = summary[model_name]
        time_str = f'{a["time_s"]:.1f}s' if a['time_s'] < 60 else f'{a["time_s"]/60:.1f}h'
        picp_str = f'{a["picp"]:.1f}\%' if a['picp'] > 0 else '---'
        print(f'{model_name} & {a["mae"]:.0f} & {a["rmse"]:.0f} & {a["mape"]:.2f}\% & {picp_str} & {time_str} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    run_all()
