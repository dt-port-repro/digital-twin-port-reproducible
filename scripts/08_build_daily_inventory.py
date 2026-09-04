"""
Step 8：逐日堆场占用重建

核心逻辑（来自港口方SQL）：
  SELECT t.day, t.yardcell, t.terminalcode, t.yardlaneno,
         tc.containerid, tc.containerno, tc.intime, tc.outtime, tc.outyardcell
  FROM t_xce3 t
  LEFT JOIN t_comtainers tc
    ON t.yardcell = tc.outyardcell
    AND t.day BETWEEN tc.intime AND tc.outtime

⚠️ 注意：outyardcell 是箱子最后存放位置，不代表历史轨迹。
上述条件确保了只有"在该时间段内该箱位确实存着这个箱子"的记录才会关联上。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from pathlib import Path

from utils.paths import processed_dir


def build_daily_inventory(
    xce3_path: Path,
    comtainers_path: Path,
    output_path: Path,
    chunk_size: int = 100000,
):
    """
    逐日堆场占用重建。

    参数：
        xce3_path: t_xce3 parquet 路径（day, yardcell, ...）
        comtainers_path: t_comtainers parquet 路径（containerid, intime, outtime, outyardcell）
        output_path: 输出路径
        chunk_size: 处理时每批天数
    """
    print("[Step 8] 重建逐日堆场占用")

    # 加载堆场快照
    xce3 = pd.read_parquet(xce3_path)
    print(f"  t_xce3: {len(xce3)} 行")
    print(f"  日期范围: {xce3['day'].min()} ~ {xce3['day'].max()}")
    print(f"  唯一箱位数: {xce3['yard_cell'].nunique()}")

    # 加载集装箱记录
    containers = pd.read_parquet(comtainers_path)
    print(f"  t_comtainers: {len(containers)} 行")

    # 确保日期为 datetime
    xce3["day"] = pd.to_datetime(xce3["day"])
    containers["in_time"] = pd.to_datetime(containers["in_time"])
    containers["out_time"] = pd.to_datetime(containers["out_time"])

    # 按天逐批处理（避免内存爆炸）
    days = sorted(xce3["day"].unique())
    results = []

    for i in range(0, len(days), chunk_size):
        batch_days = days[i : i + chunk_size]
        xce3_batch = xce3[xce3["day"].isin(batch_days)]

        # 关键关联逻辑：
        # t_xce3.yardcell = t_comtainers.outyardcell
        # AND t_xce3.day BETWEEN t_comtainers.intime AND t_comtainers.outtime
        merged = xce3_batch.merge(
            containers,
            left_on="yard_cell",
            right_on="out_yard_cell",
            how="left",
            suffixes=("", "_ctn"),
        )

        # 时间区间过滤
        mask = (merged["day"] >= merged["in_time"]) & (merged["day"] <= merged["out_time"])
        merged = merged[mask | merged["container_id"].isna()]  # 保留空箱位

        results.append(merged)

        if (i // chunk_size + 1) % 5 == 0 or len(results) < 3:
            print(f"  处理进度: {min(i + chunk_size, len(days))}/{len(days)} 天")

    df = pd.concat(results, ignore_index=True)
    print(f"\n  ✅ 重建完成: {len(df)} 行")
    print(f"  其中占用箱位: {df['container_id'].notna().sum()} / {len(df)}")

    # 保存
    df.to_parquet(output_path, index=False)
    print(f"  输出: {output_path}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return df


def run():
    """运行逐日堆场占用重建"""
    proc = processed_dir()

    xce3_path = proc / "07_t_xce3.parquet"
    ctn_path = proc / "07_t_comtainers.parquet"
    output_path = proc / "08_daily_inventory.parquet"

    if not xce3_path.exists():
        print(f"❌ 未找到 t_xce3 数据: {xce3_path}")
        print("请先运行 07_parse_xce_dump.py 解析 xce.dmp")
        return

    if not ctn_path.exists():
        print(f"❌ 未找到 t_comtainers 数据: {ctn_path}")
        print("请先运行 07_parse_xce_dump.py 解析 xce.dmp")
        return

    build_daily_inventory(xce3_path, ctn_path, output_path)


if __name__ == "__main__":
    run()
