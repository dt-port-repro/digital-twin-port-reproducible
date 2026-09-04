# 第五章 堆场作业预测与优化模型 — 实验复现包

## 快速复现

```bash
# 在复现包根目录下依次运行（数据源为 output/10_*.parquet 特征产物，由清洗Step 10生成）
python experiments/chapter5/scripts/16_lstm_prediction.py   # 表5.1 预测模型对比
python experiments/chapter5/scripts/18_yard_selection.py    # 表5.2 堆场选位对比
python experiments/chapter5/scripts/17_ppo_coordinator.py   # 表5.3 PPO协调器
```

输出论文表5.1-5.3全部数据。PPO训练（17）耗时较长，可用已发布的 `output/ppo_results/ppo_results.json` 直接核对。

## 数据来源

基于 **深圳妈湾港（MCT）2024年全年日频数据**（366天）。预处理文件：

| 文件 | 内容 |
|:---|:---|
| `data/processed/05_export_manifest.parquet` | 出口集装箱清单（45MB） |
| `data/processed/06_yard_definition.parquet` | 堆场箱位定义（246,856箱位） |
| `data/processed/06_movement_events.parquet` | 集装箱移动事件流（60MB） |

## 规范化结果文件

| 文件 | 内容 | 论文对应 |
|:---|:---|---:|
| `output/lstm_results/full_comparison_results.json` | 5模型预测对比 | 表5.1 |
| `output/yard_selection_results/selection_results_v2.json` | 三阶段选位 vs FCFS | 表5.2 |
| `output/ppo_results/ppo_results.json` | PPO动态 vs 静态权重 | 表5.3 |
| `output/lstm_results/quantile_lstm_model.pt` | 分位数LSTM模型权重 | §5.1 |
| `output/ppo_results/ppo_w1.pt` / `ppo_w2.pt` | PPO模型权重 | §5.3 |

## 核心实验数据

### 表5.1 预测模型对比

| 模型 | MAE(TEU) | RMSE(TEU) | MAPE(%) | PICP(%) |
|:---|---:|---:|---:|---:|
| ARIMA(2,1,2) | 2239.9 | 2695.8 | 5.22 | 64.0 |
| LSTM | 2358.3 | 2843.5 | 5.36 | 12.1 |
| Transformer | 2154.4 | 2738.8 | 4.96 | 57.8 |
| **本文模型(LSTM+Attention)** | **1960.2** | **2415.7** | **4.47** | **41.8** |

### 表5.2 堆场选位对比

| 方法 | 平均惩罚 | 标准差 | 改善 |
|:---|---:|---:|:---|
| FCFS先到先得 | 0.1695 | 0.0713 | — |
| **三阶段惩罚选位** | **0.1121** | **0.0529** | **↓33.9%** |

### 表5.3 PPO协调器

| 窗口 | PPO奖励 | 静态基线 | 提升 |
|:---:|:---:|:---:|:---:|
| W1 | -318.7 | -404.2 | +85.5 |
| W2 | -427.7 | -584.0 | +156.3 |
| **平均** | **-373.2** | **-494.1** | **+120.9 (24.5%)** |

### 图表对照

| 文件 | 论文对应 | 数据来源 |
|:---|:---|---:|
| `table_5_1_prediction.csv` | 表5.1 预测模型对比 | `full_comparison_results.json` |
| `table_5_2_yard.csv` | 表5.2 堆场选位对比 | `selection_results_v2.json` |
| `table_5_2_ppo.csv` | 表5.2 PPO明细（两窗奖励） | `ppo_results.json` |
| `table_5_2_collaborative.csv` | 表5.2 协同效果 | `prediction_yard_integration.json` |
| `fig5_1_model_comparison.png` | 图5.1 预测模型对比 | `scripts/17_full_comparison.py` |
| `fig5_2_training_curve.png` | 图5.2 训练曲线 | `scripts/16_lstm_prediction.py` |
| `fig5_4_ppo_reward.png` | 图5.4 PPO奖励曲线 | `scripts/17_ppo_coordinator.py` |

## 复现完整实验

如需重新训练模型（耗时较长）：

```bash
# 第1步：LSTM基础预测（§5.1）
python scripts/16_lstm_prediction.py

# 第2步：五模型全方位对比
python scripts/17_full_comparison.py

# 第3步：堆场选位（§5.2）
python scripts/18_yard_selection.py

# 第4步：PPO训练（§5.3）
python scripts/17_ppo_coordinator.py
```

## 目录结构

```
experiments/chapter5/
├── run_all.py                  ← 一键复现
├── README.md                   ← 本文件
├── data/README.md              ← 数据来源说明
├── scripts/
│   ├── verify_paper_data.py    ← 论文数据验证
│   └── ... (实验脚本，参考 output/chapter5_reproducibility/scripts/)
└── output/verified/
    ├── table_5_1_prediction.csv
    ├── table_5_2_yard.csv
    └── table_5_2_ppo.csv
```
