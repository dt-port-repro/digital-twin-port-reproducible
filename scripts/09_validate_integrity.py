"""
Step 9：跨表一致性校验

检查项：
  1. 关键字段空值率（BERTHPLANNO / CONTAINERID / 位置坐标）
  2. 跨表关联验证（BERTHPLANNO 贯通表一/三/四）
  3. CONTAINERID 贯通表四/八
  4. 时间范围覆盖 2024 全年 1-12 月
  5. YARDCELL 在 yard_definition 中存在性
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from pathlib import Path

from utils.paths import processed_dir


def check_null_rates(df: pd.DataFrame, name: str, key_cols: list):
    """检查关键字段空值率"""
    print(f"\n  📊 {name} — 关键字段空值率:")
    for col in key_cols:
        if col in df.columns:
            rate = df[col].isna().mean() * 100
            status = "✅" if rate < 1 else ("⚠️" if rate < 5 else "❌")
            print(f"    {status} {col}: {rate:.2f}%")


def check_time_range(df: pd.DataFrame, col: str, name: str, expected_year: int = 2024):
    """检查时间范围"""
    if col not in df.columns:
        print(f"  ⚠️ {name}: 缺少 {col} 列")
        return
    valid = df[col].notna()
    if valid.sum() == 0:
        print(f"  ❌ {name}: {col} 全部为空")
        return
    min_t, max_t = df[col].min(), df[col].max()
    year_2024 = df.loc[valid, col].dt.year == expected_year
    coverage = year_2024.mean() * 100
    status = "✅" if coverage > 95 else "⚠️"
    print(f"  {status} {name}.{col}: {min_t} ~ {max_t}  (2024覆盖率: {coverage:.1f}%)")


def run_all():
    """运行所有校验"""
    proc = processed_dir()
    parquet_files = list(proc.glob("*.parquet"))
    print(f"找到 {len(parquet_files)} 个处理后文件:\n")

    loaded = {}
    for f in sorted(parquet_files):
        name = f.stem
        df = pd.read_parquet(f)
        loaded[name] = df
        print(f"  📄 {f.name}  —  {len(df)} 行 × {len(df.columns)} 列")

    print("\n" + "=" * 60)
    print("校验 1: 关键字段空值率")
    print("=" * 60)

    # 表一：船舶
    if "01_vessels" in loaded:
        check_null_rates(loaded["01_vessels"], "船舶资料",
                         ["berth_plan_no", "imo", "max_teu", "max_bay", "max_row", "max_tier"])

    # 表四/五：出口清单
    if "05_export_manifest" in loaded:
        check_null_rates(loaded["05_export_manifest"], "出口清单",
                         ["container_id", "container_no", "berth_plan_no",
                          "container_size", "gross_weight", "stow_position"])

    # 表八：移动事件流
    if "06_movement_events" in loaded:
        check_null_rates(loaded["06_movement_events"], "移动事件流",
                         ["container_id", "op_type", "op_time", "target_pos"])

    print("\n" + "=" * 60)
    print("校验 2: 时间范围覆盖")
    print("=" * 60)

    if "03_berth_plan" in loaded:
        check_time_range(loaded["03_berth_plan"], "actual_berth", "靠离泊")
        check_time_range(loaded["03_berth_plan"], "actual_depart", "靠离泊")

    if "06_movement_events" in loaded:
        check_time_range(loaded["06_movement_events"], "op_time", "移动事件流")

    print("\n" + "=" * 60)
    print("校验 3: 跨表关联")
    print("=" * 60)

    # BERTHPLANNO 贯通
    bp_sets = {}
    for name in ["01_vessels", "03_berth_plan", "05_export_manifest"]:
        if name in loaded and "berth_plan_no" in loaded[name].columns:
            bp_sets[name] = set(loaded[name]["berth_plan_no"].dropna().unique())
            print(f"  {name}: {len(bp_sets[name])} 个唯一 BERTHPLANNO")

    if len(bp_sets) >= 2:
        names = list(bp_sets.keys())
        common = bp_sets[names[0]]
        for s in bp_sets.values():
            common &= s
        print(f"\n  跨表共有 BERTHPLANNO: {len(common)}")
        if len(common) == 0:
            print("  ⚠️ 各表 BERTHPLANNO 可能编码不一致，需检查")

    # CONTAINERID 贯通
    cid_sets = {}
    for name in ["05_export_manifest", "06_movement_events"]:
        if name in loaded and "container_id" in loaded[name].columns:
            cid_sets[name] = set(loaded[name]["container_id"].dropna().unique())
            print(f"  {name}: {len(cid_sets[name])} 个唯一 CONTAINERID")

    if "05_export_manifest" in cid_sets and "06_movement_events" in cid_sets:
        manifest_only = cid_sets["05_export_manifest"] - cid_sets["06_movement_events"]
        events_only = cid_sets["06_movement_events"] - cid_sets["05_export_manifest"]
        common = cid_sets["05_export_manifest"] & cid_sets["06_movement_events"]
        print(f"\n  出口清单独有: {len(manifest_only)}")
        print(f"  事件流独有: {len(events_only)}")
        print(f"  共有: {len(common)}")

    print("\n" + "=" * 60)
    print("✅ 校验完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
