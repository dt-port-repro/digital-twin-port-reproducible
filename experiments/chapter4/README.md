# 论文第四章实验材料 — 港口数字孪生配载优化

## 实验总览

基于妈湾港真实数据（~105万条集装箱记录），6艘代表性外贸船+10艘验证船。

### 实验列表

| 实验 | 内容 | 核心结论 |
|------|------|----------|
| 实验1 | 6船GA-RH基准优化 | 小船fitness 0.75-0.85 → 大船0.46，单调递减(r=-0.97) |
| 实验2 | 大规模实例（+10新船） | GA-RH优于纯GA约5-10%，验证可扩展性 |
| 实验3 | 协同效果(γ=0) | 独立优化对比，验证堆场协同模块价值 |
| 实验4 | 鲁棒性分析 | ±20%扰动CV<1.2%，算法高度稳定 |
| 实验5 | 消融分析 | 后处理规则提升<0.3%，GA已充分收敛 |
| 实验6 | 编码策略对比 | A原始 vs B扁平+惩罚 vs C近似最优(200gen)；C比A高~1.4% |

### 目录结构

```
chapter4/
├── README.md              # 本文件
├── data/                  # 中间特征数据
│   ├── 01_vessel_profiles.parquet
│   ├── 02_bay_structure.parquet
│   ├── 03_container_master.parquet
│   ├── 10_stowage_features.parquet
│   └── test_ships.parquet
├── scripts/               # 实验脚本
│   ├── 15_ga_rh_stowage.py       # GA-RH优化核心
│   ├── phase1_new_ships.py       # 实验2: GA-RH
│   ├── phase2_pure_ga.py         # 实验2: 纯GA对比
│   ├── exp3_gamma0.py            # 实验3
│   ├── exp4_robustness.py        # 实验4
│   ├── exp5_ablation.py          # 实验5
│   └── exp6_encoding.py          # 实验6
├── results/               # 实验结果(Parquet)
│   ├── test_*.parquet            # 实验1各船原始结果
│   ├── phase1_new_ships.parquet  # 实验2 GA-RH
│   ├── phase2_pure_ga.parquet    # 实验2 纯GA
│   ├── phase3_gamma0.parquet     # 实验3
│   ├── exp4_robustness.parquet   # 实验4
│   ├── exp5_ablation.parquet     # 实验5
│   ├── exp6_encoding.parquet     # 实验6
│   └── all_experiments.parquet   # 全部合并
    ├── table_exp1_baseline.tex
    └── table_exp6_encoding.tex
```

### 运行环境

- Python 3.13, pandas, numpy, reportlab
- 无外部ML依赖（GA为纯Python实现）
- 数据来源：妈湾港(MCT) 2024年真实运营数据

### 数据安全

本仓库仅包含聚合结果和中间特征。原始数据（Excel/CSV）不在此目录。
