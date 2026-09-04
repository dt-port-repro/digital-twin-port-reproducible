# 数字孪生港口协同优化 — 完整复现包

## 论文：基于数字孪生的港口配载与堆场协同优化研究

---

## 一、数据可用性声明

本研究数据来源于深圳妈湾港（MCT）2024年真实运营数据（含105万条集装箱移动记录、船舶数据及堆场246,856箱位结构参数）。因数据使用协议限制，原始数据无法公开共享。经脱敏处理后的关键实验数据集及完整复现代码已包含在本复现包中。有数据访问需求的读者可通过通讯作者申请获取脱敏原始数据，申请需经港口管理方审批并签署数据使用协议。所有代码遵循 Apache License 2.0 开源。相关数据链接：[https://pan.quark.cn/s/682dc4846e9f](https://pan.quark.cn/s/682dc4846e9f)（下载 `data/` 文件夹，解压/放置到仓库根目录，与 `scripts/`、`02_code/` 同级）

---

## 二、软硬件依赖

### 硬件要求
本复现包在以下配置上测试通过：

| 组件 | 规格 |
|------|------|
| CPU | AMD Ryzen AI MAX+ 395（16核） |
| 内存 | 112 GB |
| 磁盘 | 仓库~80MB；完整复现另需夸克数据包（~2.5GB，见数据可用性声明） |
| GPU | Radeon 8060S（集成显卡，CPU即可运行） |

说明：GitHub 仓库仅含代码、脱敏标准化结果与图表；MCT 原始数据经脱敏后通过夸克网盘分发（见数据可用性声明）。所有实验在 CPU 上即可完成。

### 软件依赖
- **操作系统**: Linux, Windows 10/11, macOS 12+
- **Python**: 3.10+（推荐 3.12）
- **包管理**: pip 或 uv

### Python 依赖
```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pyarrow>=12.0.0
```

---

## 三、文件结构

```
replication_package/
│
├── README.md                         # 本文件
├── LICENSE                           # Apache-2.0许可证
├── Data_Availability_Statement.md    # 数据可用性声明
├── Dockerfile                        # 容器化运行环境
├── requirements.txt                  # Python依赖清单
├── verify_replication.py             # 复现验证脚本
│
├── data/                           # MCT 2024 数据（夸克网盘获取，放置到仓库根目录）
│   ├── raw/2024/                    # 原始运营数据（经港口侧脱敏）
│   ├── processed/                   # 清洗后数据集（8个parquet + 质量报告）
│   ├── samples/                     # 样例数据CSV
│   ├── ship_sample_100.csv          # 第四章船舶样例数据
│   ├── ship_sample_100_large.csv    # 第四章船舶样例数据（大）
│   ├── ship_sample_80_large.csv     # 第四章船舶样例数据（80%大）
│   └── 放置说明_README.md           # 数据放置与结构说明
│
├── 01_data/                        # 数据字典（随仓库内置）
│   └── codebook/data_dictionary.md  # 数据字典（字段说明）
│
├── scripts/                          # 数据清洗管线（Step 1-10）
│
├── experiments/                      # 原始实验脚本
│   ├── chapter4/scripts/             # 第四章GA-RH配载
│   ├── chapter5/scripts/             # 第五章预测-优化
│   └── chapter6/scripts/             # 第六章MAS-DES
│
├── 02_code/                          # 结构化源代码
│   ├── data/models.py                # 仿真数据模型
│   ├── modules/agents.py             # MAS智能体
│   ├── stowage_optimization/         # 第四章优化算法
│   ├── yard_prediction/              # 第五章§5.1 堆场预测
│   ├── yard_optimization/            # 第五章§5.2 堆场选位
│   ├── rl_coordinator/               # 第五章§5.3 PPO
│   ├── simulation/                   # 第六章仿真引擎
│   └── visualization/                # 论文图表生成
│
├── 03_results/                       # 实验结果
│   ├── canonical/                    # 标准数据源
│   ├── figures/                      # 第六章图（PNG+SVG）
│   ├── tables/                       # 论文表格（CSV）
│   └── logs/                         # 运行日志
│
├── 04_derivations/                    # 公式推导（第3-5章，论文附录C承诺）
│   ├── 论文核心公式推导.docx          # 核心公式推导正式文档（论文附录C C.3 指向）
│   └── supplementary_derivations.md   # 推导补充（总览/逐章推导/代码核对）
│
├── 05_appendix/                      # 附录A-E
│
└── output/                           # 第四、五章输出
    ├── large_scale/                  # 第四章图
    └── chapter5/                     # 第五章图+结果JSON
```

### 数据与代码对应关系

| 数据 | 位置 | 用于 |
|------|------|------|
| MCT原始数据 | `data/raw/2024/` | 实验原始输入（夸克获取） |
| 清洗后数据 | `data/processed/` | 第四章实验输入（夸克获取） |
| 第四章规范结果 | `03_results/canonical/exp*_*.parquet` | 论文表4.5-4.14 |
| 第六章仿真结果 | `03_results/canonical/sim_all_cfgall_d30r10.parquet` | 论文表6.1-6.4/图6.2-6.4 |
| 第五章结果 | `output/chapter5/results/*.json` | 论文表5.1-5.3/图5.2-5.6 |
| 论文表格 | `03_results/tables/table_*.csv` | 可直接用于论文 |
| 论文图表 | `03_results/figures/` + `output/` | 论文插图 |

---

## 四、各章实验说明

### 第四章：船舶智能配载优化

| 实验 | 内容 | 对应表格 | 运行命令 |
|------|------|---------|---------|
| GA-RH基本性能 | 6船, gen=50, pop=60 | 表4.5 | `run_all_ch4.py` |
| GA-RH vs 纯GA | 6船对比 | 表4.6 | `run_all_ch4.py` |
| FCFS基线 | 6船FCFS | 表4.7 | `run_all_ch4.py` |
| γ=0消融 | gen=80 | 表4.8-4.9 | `run_all_ch4.py` |
| 后处理/编码/鲁棒性 | 多项对比 | 表4.10-4.14 | `run_*.py` |
| 大规模实验 | 100船×3算法 | 图4.2-4.6 | `run_large_scale_experiments.py` |

```bash
# 一键运行全部第四章实验（需MCT数据）
python experiments/chapter4/scripts/run_ch4_unified.py

# 或使用重构后的代码
python 02_code/stowage_optimization/run_all_ch4.py
```

### 第五章：堆场作业预测与优化

| 模块 | 内容 | 对应 | 运行命令 |
|------|------|------|---------|
| §5.1 预测 | LSTM-GNN-注意力 + Deep Ensemble | 表5.1/图5.2 | `experiments/chapter5/scripts/16_lstm_prediction.py` |
| §5.2 选位 | 三阶段惩罚函数堆场选位 | 表5.2/图5.4 | `experiments/chapter5/scripts/18_yard_selection.py` |
| §5.3 PPO | 预测-优化强化学习协调器 | 表5.3/图5.6 | `experiments/chapter5/scripts/17_ppo_coordinator.py` |

```bash
# 论文实验结果对应脚本（自 output/10_* 特征产物生成）
python experiments/chapter5/scripts/16_lstm_prediction.py   # 预测对比 -> 表5.1
python experiments/chapter5/scripts/18_yard_selection.py    # 三阶段选位 -> 表5.2
python experiments/chapter5/scripts/17_ppo_coordinator.py   # PPO协调器 -> 表5.3
```

> 说明：`02_code/yard_prediction/`、`02_code/yard_optimization/`、`02_code/rl_coordinator/`
> 为模块化重构实现（接口与实验脚本一致）；论文表5.1-5.3的正式数据源为
> `output/chapter5/results/*.json` 与上述 experiments 脚本。

### 第六章：数字孪生系统验证

| 场景 | 配置 | 仿真参数 | 对应 |
|------|------|---------|------|
| 常规作业 | A/B/C/D | 30天×10轮, ~3.1艘/天 | 表6.1/图6.2 |
| 高峰压力 | A/B/C/D | 30天×10轮, ~7.8艘/天 | 表6.2/图6.3 |
| 异常情况 | A/B/C/D | 30天×10轮, 故障率1.5% | 表6.3/图6.4 |

```bash
# 一键运行全部仿真
python 02_code/simulation/run_scenarios.py

# 或使用原始仿真脚本
python experiments/chapter6/scripts/exp6_complete.py
```

---

## 五、复现验证

### 5.1 复现范围说明

本复现包的验证分为两个层级：

| 层级 | 验证内容 | 是否需要原始数据 | 适用场景 |
|------|---------|---------------|---------|
| **完整验证** | 原始数据 → 清洗 → 实验 → 图表 | 需要 MCT 数据 | 有原始数据的读者 |
| **代码+结果验证** | 代码完整性 + 实验结果 vs 论文图表一致性 | 不需要 | 审稿人/无原始数据的读者 |

**关于原始数据：** 妈湾港（MCT）2024年原始运营数据（经港口侧脱敏）通过夸克网盘链接获取（见本文档开头数据可用性声明），解压后置于 `01_data/` 目录下即可运行完整验证。本仓库同时包含预处理后的标准化结果（`03_results/canonical/`）与实验数据集，无需原始数据即可验证核心实验结果与论文图表的一致性（`--quick` / 图表对照）。

### 5.2 验证命令

```bash
# 安装依赖
pip install -r requirements.txt

# 完整验证（需要原始数据）
python verify_replication.py

# 代码+结果验证（不需要原始数据）
python verify_replication.py --skip-raw-data

# 仅数值一致性验证（最快）
python verify_replication.py --quick
```

### 5.3 结果预期

正常输出以 `✅ 全部验证通过！` 结尾。验证覆盖 14 个维度共 120+ 项检查，包括：文档完整性、代码完整性、实验数据与 canonical 数据数值一致性、表格与 parquet 交叉验证、跨章节数据血缘、parquet 文件完整性等。失败项会在输出中以 ❌ 标记并列出具体原因。

**复现容差与已知说明（重要）：**

1. **第四章 GA-RH 为随机优化算法**：6 艘测试船中，4 艘小船（≤1,761 箱）重跑 fitness 与论文表4.5 数值一致至 canonical 存储精度（±1e-6）；2 艘大船（CGAMV 2,993 箱 / APESP 4,008 箱）存在多峰特性，不同运行环境（Python/numpy 版本、hash seed）下 fitness 波动约 ±1.5%，属遗传算法正常随机性。论文表4.5 的权威数值以 `03_results/canonical/exp1_garh.parquet` 为准（该文件与论文逐位一致），重跑脚本 `run_ch4_unified.py` 验证算法可复现性而非逐位复制。
2. **第六章仿真**：`03_results/canonical/sim_all_cfgall_d30r10.parquet` 为 3 场景×4 配置×10 轮的行级均值（与论文表6.1-6.4 口径一致）；论文表格显示 1 位小数，个别格存在 ±0.1pp 舍入差异（如异常场景 D 配置 5.35% 显示为 5.4%）。
3. **f₁ 列说明**：论文表4.8 的 GA-RH(f₁) 列（GA-RH 解的翻箱指标）曾因早期脚本字段名错误（读取不存在的 `detail['stability']` 键）在 parquet 中为占位 0。因 fitness 公式已知且可逆，f₁ 已从 `exp1_garh.parquet` 的 fitness/efficiency/balance/penalty **精确反推恢复**（反推值与论文表4.8 GA-RH(f₁) 列 6/6 船逐位吻合，2026-09 验证），相关脚本已修复。论文表4.5 的"翻箱 f₁"列数值以论文为准，其反推基准见 `03_results/tables/table_4_9_aggregate.csv` 注释。

### 5.4 Docker 运行（可选）

```bash
docker build -t dt-port .
docker run dt-port python verify_replication.py --quick   # 容器内运行核心数值一致性自检
```

注：容器仅包含代码与标准化结果，`--quick` 模式执行数值一致性验证（实验数据 vs canonical）；
完整验证（含原始数据检查）请在宿主机获取夸克数据包后运行。

### 5.5 重新生成论文图表（可选）

```bash
# 全流程从实验到图表
python run_all.py --pipeline all

# 或仅图表生成
python run_all.py --pipeline viz
```

---

## 六、关键结果汇总

### 第四章
| 指标 | 值 |
|------|------|
| GA-RH平均fitness | 0.6033（6船） |
| GA-RH vs 纯GA | Δ<1.21%（两者等价） |
| GA-RH vs FCFS | 平均提升18% |
| 鲁棒性CV | <1.23%（算法高度稳定） |
| 大规模时间-箱量r | ≈0.95（GA-RH） |

### 第五章
| 指标 | 值 |
|------|------|
| 预测MAPE | 4.47%（两窗平均） |
| PICP | 87.0%（通过分位数损失） |
| 三阶段选位 | 惩罚值降低33.9% vs FCFS |
| PPO协调器 | W1+21.1% / W2+26.7% vs 静态基线 |

### 第六章
| 指标 | 配置C（组合） | 配置D（全协同） |
|------|:----------:|:----------:|
| 常规船时改善 | ↓5.5% | ↓3.0% |
| 高峰船时改善 | ↓5.8% | ↓2.9% |
| 异常船时改善 | ↓5.8% | ↓5.4% |
| 翻箱率改善 | ↓13.5% | ↓27.0% |
| 设备利用率提升 | ↑2.6% | ↑5.6% |
