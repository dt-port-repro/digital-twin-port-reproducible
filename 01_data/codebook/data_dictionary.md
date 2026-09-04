# 数据字典 / Data Codebook

## 原始数据字段说明（因数据使用协议，仅提供结构描述，不含原始数据）

### 1. 集装箱移动记录（container_movements.csv）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| container_id | string | 集装箱编号（脱敏） |
| size_type | int | 箱型尺寸（20/40/45） |
| weight_kg | float | 箱重（kg） |
| pod | string | 卸货港代码 |
| stow_position | string | 船上位置编码 |
| yard_position | string | 堆场位置编码 |
| operation_type | string | 作业类型（load/discharge/shift） |
| operation_time | datetime | 作业时间戳 |
| vessel_code | string | 船舶代码 |

### 2. 船舶资料（vessel_schedule.csv）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| vessel_code | string | 船舶代码 |
| vessel_name | string | 船名（脱敏） |
| max_teu | int | 最大装载量（TEU） |
| n_containers | int | 实际装载箱数 |
| berth_plan_no | string | 泊位计划编号 |
| arrival_time | datetime | 到港时间 |
| departure_time | datetime | 离港时间 |
| n_pod | int | 挂港数 |

### 3. 堆场结构（yard_structure.csv）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| bay | int | 贝位编号 |
| row | int | 列编号 |
| tier | int | 层编号 |
| size_type | string | 兼容箱型（20/40） |
| max_weight | float | 最大堆重（kg） |
| zone | string | 箱区代码 |

### 4. 仿真实验数据集（canonical/ 目录）
| 文件名 | 说明 | 行数 |
|--------|------|------|
| exp1_garh.parquet | 第四章GA-RH基本性能（6船） | 6 |
| exp2_pure_ga.parquet | 第四章纯GA对比 | 6 |
| exp3_fcfs.parquet | 第四章FCFS基线 | 6 |
| exp4_gamma0.parquet | 第四章γ=0消融 | 6 |
| exp5_postproc.parquet | 第四章后处理消融 | 6 |
| exp6_encoding.parquet | 第四章编码策略对比 | 15 |
| exp_robustness.parquet | 第四章鲁棒性测试 | 135 |
| exp_large_scale.parquet | 第四章大规模实验（3算法×100船） | 300 |
| sim_all_cfgall_d30r10.parquet | 第六章全部仿真结果（3场景×4配置×10轮） | 120 |

### 5. 实验输出数据（output/ 目录）
| 文件 | 说明 |
|------|------|
| chapter5/results/lstm_results.json | 第五章LSTM预测结果 |
| chapter5/results/ppo_results.json | 第五章PPO训练结果 |
| chapter5/results/selection_results_v2.json | 第五章堆场选位结果 |
| chapter5/results/full_comparison_results.json | 第五章全模型对比 |
| chapter5/results/ | 详见 output/chapter5/results/ |
