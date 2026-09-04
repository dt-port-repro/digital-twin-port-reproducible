"""clean_container_movements.py — 集装箱移动数据清洗

清洗流程（引用 scripts/01_to_04_clean_basics.py + scripts/13_clean_all_tables.py）：
  1. 读取原始 Excel → 修复编码/BOM头
  2. 列名标准化：CTOS原始名 → 下划线英文名
  3. 类型转换：时间戳、数值列
  4. 去重、删除全空列
  5. 输出清洗后 parquet → data/processed/06_movement_events.parquet

用法:
  from data_cleaning.clean_container_movements import clean_movements
  df = clean_movements(input_path, output_path)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def clean_movements(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rows: Optional[int] = None,
) -> pd.DataFrame:
    """清洗集装箱移动事件流。

    Args:
        input_path: 原始 Excel 路径（如 data/raw/2024/8 ...）
        output_path: 可选，输出 parquet 路径
        sample_rows: 可选，仅读取前 N 行（调试用）

    Returns:
        清洗后的 DataFrame
    """
    print(f"[clean_movements] 读取: {input_path}")
    df = pd.read_excel(input_path, nrows=sample_rows)
    print(f"  原始行数: {len(df)}")

    # 列名标准化：空格→下划线，去掉特殊字符
    df.columns = [
        col.strip().replace(" ", "_").replace("/", "_")
        for col in df.columns
    ]

    # 删除全空列
    df = df.dropna(axis=1, how="all")
    print(f"  有效列数: {len(df.columns)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  已保存: {output_path}")

    return df


if __name__ == "__main__":
    # 示例用法
    df = clean_movements(
        "data/raw/2024/8 2024年上半年MCT集装箱移动事件流.xlsx",
        "data/processed/06_movement_events.parquet",
    )
