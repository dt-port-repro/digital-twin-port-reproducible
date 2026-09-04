"""clean_yard_occupancy.py — 堆场数据清洗

清洗流程（引用 scripts/01_to_04_clean_basics.py + scripts/07_parse_xce_dump.py）：
  1. 读取堆场箱位定义 Excel → 解析 34 万行箱位
  2. 读取堆场快照 xce.dmp → 解析 Oracle dump 格式
  3. 合并箱位定义与快照 → 建立堆场实时占用视图
  4. 计算日占用率：yard_occupancy_by_date
  5. 输出清洗后 parquet → data/processed/06_yard_definition.parquet

用法:
  from data_cleaning.clean_yard_occupancy import clean_yard
  df = clean_yard(input_path, output_path)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def clean_yard_definition(
    input_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """清洗堆场箱位定义。

    Args:
        input_path: 原始 Excel 路径（6 MCT堆场箱位定义.xlsx）
        output_path: 可选，输出 parquet 路径

    Returns:
        清洗后的 DataFrame（~34万行）
    """
    print(f"[clean_yard] 读取堆场定义: {input_path}")
    df = pd.read_excel(input_path)
    print(f"  原始行数: {len(df)}")

    # 列名标准化
    df.columns = [
        col.strip().replace(" ", "_").replace("/", "_")
        for col in df.columns
    ]

    # 筛选有效箱位（排除特殊区域）
    # （具体筛选规则视数据情况而定）

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  已保存: {output_path}")

    return df


def compute_daily_occupancy(
    yard_cells_path: str,
    movement_events_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """计算堆场日占用率。

    Args:
        yard_cells_path: 堆场单元格 parquet 路径
        movement_events_path: 移动事件 parquet 路径
        output_path: 可选，输出路径

    Returns:
        日占用率 DataFrame
    """
    print("[clean_yard] 计算日占用率...")
    cells = pd.read_parquet(yard_cells_path)
    events = pd.read_parquet(movement_events_path)

    print(f"  堆场单元格: {len(cells)}")
    print(f"  移动事件: {len(events)}")

    # 按日期聚合占用情况
    # （具体逻辑视实际数据建模）

    result = pd.DataFrame()  # 占位
    if output_path:
        result.to_parquet(output_path, index=False)
    return result


if __name__ == "__main__":
    df = clean_yard_definition(
        "data/raw/2024/6 MCT堆场箱位定义.xlsx",
        "data/processed/06_yard_definition.parquet",
    )
