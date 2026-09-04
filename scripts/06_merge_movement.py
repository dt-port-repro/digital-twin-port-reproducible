"""
Step 6：合并移动事件流 — 表八

上半年 + 下半年 → 全年完整移动事件流
按时间排序，标准化操作类型字段
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from utils.paths import raw_2024_dir, processed_dir
from utils.schema import MOVEMENT_EVENTS


def load_movement_half(path, label):
    """加载半年移动事件流"""
    print(f"[Step 6] 读取 {label}: {path.name}")
    df = pd.read_excel(path)

    print(f"  Sheet: {path.stem}, 列数: {len(df.columns)}, 行数: {len(df)}")

    # 列名标准化
    flat_map = {}
    for c in MOVEMENT_EVENTS.columns:
        for rn in c.raw_names:
            flat_map[rn] = c.name

    rename = {}
    for raw_col in df.columns:
        col_str = str(raw_col).strip().strip('"').strip("'")
        best_match = None
        best_len = 0
        for raw_name, std_name in flat_map.items():
            if col_str == raw_name or col_str.strip() == raw_name.strip():
                rename[raw_col] = std_name
                break
            # Fuzzy match for long Chinese names (取最长匹配)
            if raw_name in col_str or col_str in raw_name:
                if len(raw_name) > best_len:
                    best_len = len(raw_name)
                    best_match = (raw_col, std_name)
        else:
            if best_match:
                rename[best_match[0]] = best_match[1]

    df = df.rename(columns=rename)
    # 去重：原始Excel可能有同名列，重命名后产生重复
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df["data_half"] = label
    return df


def merge_movement_events():
    """合并全年移动事件流"""
    raw = raw_2024_dir()

    h1_path = raw / "8 2024年上半年MCT集装箱移动事件流.xlsx"
    h2_path = raw / "8 2024年下半年MCT集装箱移动事件流.xlsx"

    df_h1 = load_movement_half(h1_path, "上半年")
    df_h2 = load_movement_half(h2_path, "下半年")

    # 确保列一致
    common_cols = list(set(df_h1.columns) & set(df_h2.columns))
    df_all = pd.concat([df_h1[common_cols], df_h2[common_cols]], ignore_index=True)
    print(f"\n  合并后总行数: {len(df_all)}")

    # 时间列排序
    time_cols = ["op_time", "in_time", "out_time"]
    for col in time_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_datetime(df_all[col], errors="coerce")

    # 按操作时间排序
    if "op_time" in df_all.columns:
        df_all = df_all.sort_values("op_time").reset_index(drop=True)
        print(f"  时间范围: {df_all['op_time'].min()} ~ {df_all['op_time'].max()}")

    # 操作类型分布统计
    if "op_type" in df_all.columns:
        op_dist = df_all["op_type"].value_counts()
        print(f"\n  操作类型分布:")
        for k, v in op_dist.items():
            print(f"    {k}: {v} ({v/len(df_all)*100:.1f}%)")

    # 尺寸标准化
    if "container_size" in df_all.columns:
        df_all["container_size"] = pd.to_numeric(
            df_all["container_size"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        )

    # 保存
    out = processed_dir() / "06_movement_events.parquet"
    df_all.to_parquet(out, index=False)
    print(f"\n  ✅ 保存: {out}")
    print(f"  文件大小: {out.stat().st_size / 1024 / 1024:.1f} MB")

    return df_all


if __name__ == "__main__":
    merge_movement_events()
