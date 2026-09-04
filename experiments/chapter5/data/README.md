# 第五章实验数据说明

## 数据来源

本实验所有数据基于 **深圳妈湾港（MCT）2024年全年真实运营数据**，经港口侧脱敏后使用。

原始运营数据通过夸克网盘获取（见仓库根目录 `Data_Availability_Statement.md` 与 `README.md` 数据可用性声明）。第五章用到的主要原始文件：

| 文件 | 内容 |
|:---|:---|
| `1 2024年MCT船舶基本资料.xlsx` | 船舶资料 |
| `2 2024年MCT典型船舶贝位结构.csv` | 典型船舶贝位结构 |
| `4 2024年上半年MCT出口集装箱清单及5 装船位置.xlsx` | 上半年出口清单 |
| `4 2024年下半年MCT出口集装箱清单及5 装船位置.xlsx` | 下半年出口清单 |
| `6 MCT堆场箱位定义.xlsx` | 堆场箱位定义（246,856箱位） |
| `8 2024年上半年MCT集装箱移动事件流.xlsx` | 上半年移动事件 |
| `8 2024年下半年MCT集装箱移动事件流.xlsx` | 下半年移动事件 |

## 预处理数据

原始数据经清洗、转换后存为 `01_data/processed_data/` 下的 parquet 文件（第五章主要使用 `05_export_manifest.parquet`、`06_yard_definition.parquet`、`06_movement_events.parquet`、`07_yard_cells.parquet`）。字段语义见 `01_data/codebook/data_dictionary.md`。
