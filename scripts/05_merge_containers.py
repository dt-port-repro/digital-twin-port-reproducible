"""
Step 5：合并出口清单 — 表四(出口清单) + 表五(装船位置)

上半年：546,051 行 | 下半年：603,071 行 → 合并后 ~1,149,122 行
两表在 Excel 中合并在一个文件内，上下半年各一个文件。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from pathlib import Path

from utils.paths import raw_2024_dir, processed_dir
from utils.schema import EXPORT_MANIFEST


def load_half(path: Path, label: str) -> pd.DataFrame:
    """加载半年的出口清单数据"""
    print(f"[Step 5] 读取 {label}: {path.name}")
    df = pd.read_excel(path)

    # 检查 sheet 结构（可能含多个 sheet）
    print(f"  Sheet: {path.stem}, 列数: {len(df.columns)}, 行数: {len(df)}")

    # 列名标准化
    flat_map = {}
    for c in EXPORT_MANIFEST.columns:
        for rn in c.raw_names:
            flat_map[rn] = c.name

    rename = {}
    for raw_col in df.columns:
        col_str = str(raw_col).strip().strip('"').strip("'")
        for raw_name, std_name in flat_map.items():
            if col_str == raw_name or col_str.strip() == raw_name.strip():
                rename[raw_col] = std_name
                break
    df = df.rename(columns=rename)

    # 标记数据来源
    df["data_half"] = label

    return df


def merge_export_manifest():
    """合并上下半年出口清单"""
    raw = raw_2024_dir()

    h1_path = raw / "4 2024年上半年MCT出口集装箱清单及5 装船位置.xlsx"
    h2_path = raw / "4 2024年下半年MCT出口集装箱清单及5 装船位置.xlsx"

    df_h1 = load_half(h1_path, "上半年")
    df_h2 = load_half(h2_path, "下半年")

    # 合并
    df_all = pd.concat([df_h1, df_h2], ignore_index=True)
    print(f"\n  合并后总行数: {len(df_all)}")
    print(f"  合计: {len(df_h1)} + {len(df_h2)} = {len(df_all)}")

    # 类型转换
    if "container_size" in df_all.columns:
        df_all["container_size"] = pd.to_numeric(
            df_all["container_size"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        )
    if "gross_weight" in df_all.columns:
        df_all["gross_weight"] = pd.to_numeric(df_all["gross_weight"], errors="coerce")

    # 检查关键字段缺失
    required = ["container_id", "container_no", "berth_plan_no", "container_size"]
    for col in required:
        if col in df_all.columns:
            null_pct = df_all[col].isna().mean() * 100
            if null_pct > 0:
                print(f"  ⚠️  {col} 缺失率: {null_pct:.2f}%")

    # 去重检查
    if "container_id" in df_all.columns:
        dupes = df_all["container_id"].duplicated().sum()
        if dupes:
            print(f"  ⚠️  CONTAINER_ID 重复: {dupes} 行")

    # 保存
    out = processed_dir() / "05_export_manifest.parquet"
    df_all.to_parquet(out, index=False)
    print(f"\n  ✅ 保存: {out}")
    print(f"  文件大小: {out.stat().st_size / 1024 / 1024:.1f} MB")

    return df_all


if __name__ == "__main__":
    merge_export_manifest()
