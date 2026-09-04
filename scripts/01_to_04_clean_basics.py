"""Step 1-4：基础清洗 — 表一(船舶)、表二(贝位)、表三(靠离泊)、表六(堆场定义)

处理内容：
  - CSV/Excel 编码修复（BOM头、中文乱码）
  - 列名标准化（CTOS原始名 → 下划线英文名）
  - 类型转换（时间、数值）
  - 基础质量统计
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from utils.paths import raw_2024_dir, processed_dir
from utils.schema import VESSELS, BAY_STRUCTURE, BERTH_PLAN, YARD_DEFINITION


def _normalize_columns(df: pd.DataFrame, name_map: Dict[str, str]) -> pd.DataFrame:
    """标准化列名：原始CTOS名 → 统一下划线英文名"""
    rename = {}
    for raw_col in df.columns:
        col_str = str(raw_col).strip().strip('"').strip("'")
        for raw_name, std_name in name_map.items():
            if col_str == raw_name or col_str.strip() == raw_name.strip():
                rename[raw_col] = std_name
                break
    # 未映射的列保留原名
    df = df.rename(columns=rename)
    # 移除标准化后重复的列名（如 AVESSELNAME 和 CVESSELNAME 都映射到 c_vessel_name）
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df

def _apply_schema_dtypes(df: pd.DataFrame, table_def) -> pd.DataFrame:
    """根据表定义强制转换列类型，避免 PyArrow 因混合类型报错"""
    for col_def in table_def.columns:
        if col_def.name not in df.columns:
            continue
        if col_def.dtype == "str":
            df[col_def.name] = df[col_def.name].astype(str).replace("nan", pd.NA).replace("None", pd.NA)
        elif col_def.dtype == "int":
            df[col_def.name] = pd.to_numeric(df[col_def.name], errors="coerce")
        elif col_def.dtype == "float":
            df[col_def.name] = pd.to_numeric(df[col_def.name], errors="coerce")
        elif col_def.dtype == "datetime":
            df[col_def.name] = pd.to_datetime(df[col_def.name], errors="coerce")
        elif col_def.dtype == "bool":
            df[col_def.name] = df[col_def.name].map({"Y": True, "N": False, "1": True, "0": False, 1: True, 0: False})
    return df


def clean_vessels(raw_dir: Path) -> pd.DataFrame:
    """表一：船舶基本资料"""
    path = raw_dir / "1 2024年MCT船舶基本资料.xlsx"
    print(f"[Step 1] 读取船舶资料: {path.name}")
    df = pd.read_excel(path)

    # 列名标准化
    flat_map = {}
    for c in VESSELS.columns:
        for rn in c.raw_names:
            flat_map[rn] = c.name
    df = _normalize_columns(df, flat_map)

    print(f"  行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"  列名: {list(df.columns)}")

    # 类型转换（根据 schema）
    df = _apply_schema_dtypes(df, VESSELS)

    # 保存
    out = processed_dir() / "01_vessels.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✅ 保存: {out}")
    return df


def clean_bay_structure(raw_dir: Path) -> pd.DataFrame:
    """表二：船舶贝位结构（CSV，注意BOM头）"""
    path = raw_dir / "2 2024年MCT典型船舶贝位结构.csv"
    print(f"[Step 2] 读取贝位结构: {path.name}")

    # 尝试多种编码读取CSV
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"  编码: {enc} ✅")
            break
        except (UnicodeDecodeError, Exception) as e:
            print(f"  编码: {enc} ❌ ({str(e)[:50]})")

    if df is None:
        raise ValueError("无法读取贝位结构CSV文件")

    # 清理列名（BOM头可能使第一列名带特殊字符）
    df.columns = [str(c).strip().strip('"').strip("'").replace('\ufeff', '') for c in df.columns]
    # 激进去除BOM/编码残留：保留纯字母数字和下划线的列名
    import re
    cleaned = []
    for c in df.columns:
        # 提取第一个纯英文词（含下划线）
        match = re.search(r'([A-Za-z_][A-Za-z0-9_]*)', c)
        cleaned.append(match.group(1) if match else c)
    df.columns = cleaned

    # 列名标准化
    flat_map = {}
    for c in BAY_STRUCTURE.columns:
        for rn in c.raw_names:
            flat_map[rn] = c.name
    df = _normalize_columns(df, flat_map)

    print(f"  行数: {len(df)}, 列数: {len(df.columns)}")

    # 类型转换（根据 schema）
    df = _apply_schema_dtypes(df, BAY_STRUCTURE)

    # 保存
    out = processed_dir() / "02_bay_structure.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✅ 保存: {out}  ({df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB)")
    return df


def clean_berth_plan(raw_dir: Path) -> pd.DataFrame:
    """表三：靠离泊计划"""
    path = raw_dir / "3、2024年MCT船舶靠离泊计划.xlsx"
    print(f"[Step 3] 读取靠离泊计划: {path.name}")
    df = pd.read_excel(path)

    flat_map = {}
    for c in BERTH_PLAN.columns:
        for rn in c.raw_names:
            flat_map[rn] = c.name
    df = _normalize_columns(df, flat_map)

    # 时间列转换
    for col in ["eta", "etd", "actual_berth", "actual_depart"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"  行数: {len(df)}")
    print(f"  时间范围: {df['actual_berth'].min()} ~ {df['actual_berth'].max()}")

    out = processed_dir() / "03_berth_plan.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✅ 保存: {out}")
    return df


def clean_yard_definition(raw_dir: Path) -> pd.DataFrame:
    """表六：堆场箱位定义"""
    path = raw_dir / "6 MCT堆场箱位定义.xlsx"
    print(f"[Step 6] 读取堆场箱位定义: {path.name}")
    df = pd.read_excel(path)

    flat_map = {}
    for c in YARD_DEFINITION.columns:
        for rn in c.raw_names:
            flat_map[rn] = c.name
    df = _normalize_columns(df, flat_map)

    # 布尔转换
    if "is_useful" in df.columns:
        df["is_useful"] = df["is_useful"].map({"Y": True, "N": False, "1": True, "0": False, 1: True, 0: False})

    print(f"  行数: {len(df)}")
    print(f"  可用箱位: {df['is_useful'].sum() if 'is_useful' in df.columns else 'N/A'}/{len(df)}")

    out = processed_dir() / "06_yard_definition.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✅ 保存: {out}")
    return df


def run_all():
    """运行所有基础清洗步骤"""
    raw = raw_2024_dir()
    print(f"原始数据目录: {raw}")
    print(f"输出目录: {processed_dir()}\n")

    clean_vessels(raw)
    clean_bay_structure(raw)
    clean_berth_plan(raw)
    clean_yard_definition(raw)

    print("\n✅ Step 1-4 基础清洗完成!")


if __name__ == "__main__":
    run_all()
