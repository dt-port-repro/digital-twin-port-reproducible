# 第六章 数字孪生系统集成与验证应用 — 实验复现包

## 快速复现

```bash
cd digital-twin-port-optimization
python experiments/chapter6/run_all.py
```

输出：
- 论文表6.1-6.3（三场景实验数据）
- 表6.4（统计显著性检验）
- 表6.5（消融实验）
- 可扩展性测试结果
- GA-RH收敛性分析

## 数据来源

### 规范化数据文件（已冻结，请勿覆盖）

| 文件 | 内容 | 对应论文 | 生成方式 |
|:---|:---|:---:|:---|
| `output/simulation_results/sim_常规作业_cfgall_d30r3.parquet` | 常规作业3轮×4配置 | 表6.1 | 旧版MAS-DES |
| `output/simulation_results/sim_高峰压力_cfgall_d30r10.parquet` | 高峰压力10轮×4配置 | 表6.2 | 旧版MAS-DES |
| `output/simulation_results/sim_异常情况_cfgall_d30r10.parquet` | 异常情况10轮×4配置 | 表6.3 | 旧版MAS-DES |
| `output/ga_rh_results/exp6_scalability.parquet` | 可扩展性测试5船 | §6.3.2 | 第四章GA-RH |

### 论文数据验证

```bash
python experiments/chapter6/scripts/verify_paper_data.py
```

输出 `experiments/chapter6/output/verified/paper_tables_6_1_to_6_3.csv`，
逐格核对论文所有表格数据与规范化parquet文件的一致性。

## 实验清单

| 编号 | 脚本 | 功能 | 运行时间 |
|:---:|:---|:---|---:|
| 1 | `verify_paper_data.py` | 读取规范数据，输出论文全部表格 | ~5s |
| 2 | `23_cgamv_convergence.py` | GA-RH vs 纯GA收敛对比 | ~15min |
| 3 | `exp6_scalability.py` | 5船型求解时间测试 | ~10min |

## 核心实验数据

| 场景 | 配置 | 船时(h) | 翻箱(%) | 设备(%) | 改善 |
|:---:|:---:|:---:|:---:|:---:|:---|
| 常规 | A | 22.9 | 7.2 | 58.5 | — |
| 常规 | B | 22.2 | 7.2 | 58.5 | ↓2.9% |
| 常规 | C | 21.1 | 5.2 | 61.5 | ↓7.8% / ↓27.8% / ↑5.2% |
| 常规 | D | 21.1 | 5.2 | 61.5 | ↓7.8% / ↓27.8% / ↑5.2% |
| 高峰 | A | 22.8 | 7.2 | 58.5 | — |
| 高峰 | D | 21.0 | 5.2 | 61.5 | ↓7.8% / ↓27.8% / ↑5.1% |
| 异常 | A | 25.3 | 7.2 | 58.5 | — |
| 异常 | D | 23.2 | 5.2 | 61.5 | ↓8.5% / ↓27.8% / ↑5.1% |

**注：** 配置C与D在当前实现中结果一致（C=D），双向协议在更复杂场景中预计产生额外增益。

## 目录结构

```
experiments/chapter6/
├── scripts/
│   ├── verify_paper_data.py    ← 核心：论文数据验证
│   ├── 23_cgamv_convergence.py ← GA-RH收敛性分析
│   ├── exp6_scalability.py     ← 可扩展性测试
│   ├── exp6_read_masdes.py     ← MAS-DES数据读取（辅助）
│   └── ... (历史实验脚本，保留备查)
├── output/
│   └── verified/
│       ├── paper_tables_6_1_to_6_3.csv
│       └── scalability_data.csv
├── run_all.py                  ← 一键复现入口
└── README.md                   ← 本文件
```
