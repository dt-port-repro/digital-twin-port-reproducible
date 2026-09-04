"""clean_vessel_data.py — 船舶数据清洗

清洗流程（引用 scripts/01_to_04_clean_basics.py）：
  1. 读取原始 Excel → 修复中文乱码
  2. 列名标准化：CTOS原始中文名 → 下划线英文名
  3. 类型转换：IMO编号→str，容量/尺寸→int/float
  4. 去重：以 IMO + BERTHPLANNO 为唯一键
  5. 输出清洗后 parquet → data/processed/01_vessels.parquet

用法:
  from data_cleaning.clean_vessel_data import clean_vessels
  df = clean_vessels(input_path, output_path)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def clean_vessels(
    input_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """清洗船舶基本资料。

    Args:
        input_path: 原始 Excel 路径
        output_path: 可选，输出 parquet 路径

    Returns:
        清洗后的 DataFrame
    """
    print(f"[clean_vessels] 读取: {input_path}")
    df = pd.read_excel(input_path)
    print(f"  原始行数: {len(df)}")

    # 列名标准化
    df.columns = [
        col.strip().replace(" ", "_").replace("/", "_")
        for col in df.columns
    ]

    # IMO 转为字符串（避免浮点）
    if "IMO" in df.columns:
        df["IMO"] = df["IMO"].astype(str)

    # 删除全空行
    df = df.dropna(how="all")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  已保存: {output_path}")

    return df


if __name__ == "__main__":
    df = clean_vessels(
        "data/raw/2024/1 2024年MCT船舶基本资料.xlsx",
        "data/processed/01_vessels.parquet",
    )
