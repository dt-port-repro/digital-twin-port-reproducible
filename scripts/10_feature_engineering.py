"""
Step 10：特征工程（v2 — 整合Oracle新数据）
=============================================
目标：为论文三大模型构造特征矩阵，整合8张清洗表

新增/改进：
  1. 07_yard_cells → 每日真实堆场占用率（替代近似推导）
  2. 08_containers → 集装箱停留时间统计
  3. 修复旧脚本引用已删除列的问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

PROC = Path('data/processed')
OUT = Path('output')
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 0. 工具函数
# ═══════════════════════════════════════════════════════════════

def safe_parse_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


# ═══════════════════════════════════════════════════════════════
# 1. 配载优化特征（修复空列引用 + 整合新数据）
# ═══════════════════════════════════════════════════════════════

def build_stowage_features():
    """船舶配载优化特征（每个箱子一条样本）"""
    print("=" * 60)
    print("特征 1：配载优化")
    print("=" * 60)

    vessels = pd.read_parquet(PROC / "01_vessels.parquet")
    bay = pd.read_parquet(PROC / "02_bay_structure.parquet")
    manifest = pd.read_parquet(PROC / "05_export_manifest.parquet")
    containers = pd.read_parquet(PROC / "08_containers.parquet")

    # ── 船舶特征（修复：max_bay/max_row/tier已被删除，保留可用列） ──
    avail_cols = [c for c in [
        "berth_plan_no", "max_teu", "length", "width", "height",
        "gross_weight", "dead_weight", "net_weight", "max_tier"
    ] if c in vessels.columns]

    vessel_features = vessels[avail_cols].copy()
    rename_map = {c: f"v_{c}" for c in avail_cols if c != "berth_plan_no"}
    # 船舶类型特征
    vtype_col = [c for c in vessels.columns if '船舶类型' in c]
    if vtype_col:
        vessel_features['v_type'] = vessels[vtype_col[0]]
        vessel_features['is_barge'] = (vessel_features['v_type'] == 'S').astype(int)

    vessel_features.rename(columns=rename_map, inplace=True)
    print(f"  船舶: {len(vessel_features)} 条 ({len(avail_cols)} 列)")

    # ── 贝位结构汇总（修复列名） ──
    type_col = [c for c in bay.columns if 'VESSELTYPE' in c.upper()]
    type_col = type_col[0] if type_col else bay.columns[0]
    print(f"  贝位类型列: '{type_col}'")

    # 安全处理 size_type 空值
    bay['size_type'] = bay.get('size_type', 'UNK').fillna('UNK')

    bay_agg = bay.groupby(type_col).agg(
        bay_total_cells=("iso_cell", "count"),
        bay_20ft_cells=("size_type", lambda x: (x == "20").sum()),
        bay_40ft_cells=("size_type", lambda x: (x == "40").sum()),
    ).reset_index()
    bay_agg.rename(columns={type_col: 'vessel_type_code'}, inplace=True)

    # ── 集装箱特征 ──
    stow_col = 'stow_position'
    if stow_col not in manifest.columns:
        pos_cols = [c for c in manifest.columns if '位' in c and ('贝' in c or '层' in c)]
        stow_col = pos_cols[0] if pos_cols else None

    box_cols = ["container_id", "container_no", "container_size", "container_type",
                "empty_full", "gross_weight", "pod", "outbound_voy", "berth_plan_no"]
    if stow_col:
        box_cols.append(stow_col)

    box_features = manifest[[c for c in box_cols if c in manifest.columns]].copy()
    if stow_col and stow_col in box_features.columns:
        box_features.rename(columns={stow_col: "stow_position"}, inplace=True)

    box_features["weight_kg"] = pd.to_numeric(box_features.get("gross_weight", 0), errors="coerce").fillna(0)
    box_features["is_heavy"] = (pd.to_numeric(box_features.get("weight_kg", 0), errors="coerce") > 15000).astype(int)
    # is_empty/is_reefer: 处理列不存在时 .get() 返回标量而非 Series 的情况
    empty_col = box_features.get("empty_full")
    box_features["is_empty"] = ((empty_col == "E") if isinstance(empty_col, pd.Series) else pd.Series(0, index=box_features.index)).astype(int)
    reefer_col = box_features.get("container_type")
    box_features["is_reefer"] = ((reefer_col == "RF") if isinstance(reefer_col, pd.Series) else pd.Series(0, index=box_features.index)).astype(int)

    # ── 目的港统计 ──
    if "pod" in box_features.columns:
        pod_stats = box_features.groupby("pod").agg(
            pod_count=("container_id", "count"),
            pod_heavy_ratio=("is_heavy", "mean"),
        ).reset_index()
    else:
        pod_stats = pd.DataFrame()

    # ── 集装箱停留时间特征（整合08_containers） ──
    # 远期预约数据(INTIME在2025+)有合理的预约入住，不参与停留时间统计
    containers['dwell_hours'] = (containers['OUTTIME'] - containers['INTIME']).dt.total_seconds() / 3600
    # 过滤合理的停留时间（1小时~365天），排除未来预约和异常值
    valid_dwell = containers[(containers['dwell_hours'] >= 1) & (containers['dwell_hours'] <= 8760)]
    dwell_stats = valid_dwell.groupby('CONTAINERNO').agg(
        dwell_mean_h=('dwell_hours', 'mean'),
        dwell_median_h=('dwell_hours', 'median'),
        dwell_max_h=('dwell_hours', 'max'),
    ).reset_index()
    dwell_stats.columns = ['container_no', 'dwell_mean_h', 'dwell_median_h', 'dwell_max_h']
    box_features = box_features.merge(dwell_stats, on='container_no', how='left')

    # ── 合并 ──
    for c in ["berth_plan_no", "container_id"]:
        if c in box_features.columns: box_features[c] = box_features[c].astype(str)
        if c in vessel_features.columns: vessel_features[c] = vessel_features[c].astype(str)

    stowage = box_features.merge(vessel_features, on="berth_plan_no", how="left")
    if len(pod_stats) > 0:
        stowage = stowage.merge(pod_stats, on="pod", how="left")

    out_path = OUT / "10_stowage_features.parquet"
    stowage.to_parquet(out_path, index=False)
    print(f"\n  输出: {out_path}")
    print(f"  维度: {stowage.shape[0]:,} × {stowage.shape[1]}")
    print(f"  列: {list(stowage.columns[:12])}...")
    print(f"  含停留时间: {'dwell_mean_h' in stowage.columns}")

    return stowage


# ═══════════════════════════════════════════════════════════════
# 2. 泊位调度特征（基本不变，去掉已删除列引用）
# ═══════════════════════════════════════════════════════════════

def build_berth_features():
    """泊位调度特征（每船一条样本）"""
    print("\n" + "=" * 60)
    print("特征 2：泊位调度")
    print("=" * 60)

    vessels = pd.read_parquet(PROC / "01_vessels.parquet")
    berth = pd.read_parquet(PROC / "03_berth_plan.parquet")
    manifest = pd.read_parquet(PROC / "05_export_manifest.parquet")

    # 类型统一
    for c in ["berth_plan_no"]:
        if c in berth.columns: berth[c] = berth[c].astype(str)
        if c in vessels.columns: vessels[c] = vessels[c].astype(str)

    bf = berth.merge(vessels, on="berth_plan_no", how="left", suffixes=("", "_v"))

    # 时间特征
    bf = safe_parse_dates(bf, ["eta", "etd", "actual_berth", "actual_depart", "靠泊时间"])

    bf["planned_duration_h"] = (bf["etd"] - bf["eta"]).dt.total_seconds() / 3600
    bf["actual_duration_h"] = (bf["actual_depart"] - bf["actual_berth"]).dt.total_seconds() / 3600
    bf["berth_delay_h"] = (bf["actual_berth"] - bf["eta"]).dt.total_seconds() / 3600
    bf["depart_delay_h"] = (bf["actual_depart"] - bf["etd"]).dt.total_seconds() / 3600

    # 船舶分类（max_teu 仍在）
    bf["is_large"] = (bf.get("max_teu", 0) > 8000).astype(int) if "max_teu" in bf.columns else 0
    bf["is_medium"] = ((bf.get("max_teu", 0) > 3000) & (bf.get("max_teu", 0) <= 8000)).astype(int)

    # 集装箱量
    if "container_id" in manifest.columns:
        box_count = manifest.groupby("berth_plan_no").agg(
            total_boxes=("container_id", "count"),
            heavy_ratio=("gross_weight", lambda x: (pd.to_numeric(x, errors="coerce") > 15000).mean()),
            reefer_count=("container_type", lambda x: (x == "RF").sum()),
        ).reset_index()
        box_count["berth_plan_no"] = box_count["berth_plan_no"].astype(str)
        bf = bf.merge(box_count, on="berth_plan_no", how="left")

    # 时间窗口
    bf["month"] = bf["actual_berth"].dt.month
    bf["weekday"] = bf["actual_berth"].dt.dayofweek
    bf["hour"] = bf["actual_berth"].dt.hour

    out_path = OUT / "10_berth_features.parquet"
    bf.to_parquet(out_path, index=False)
    print(f"\n  输出: {out_path}")
    print(f"  维度: {bf.shape[0]:,} × {bf.shape[1]}")
    print(f"  列: {list(bf.columns)[:12]}...")

    return bf


# ═══════════════════════════════════════════════════════════════
# 3. 堆场利用率特征（使用07_yard_cells真实数据替代近似推导）
# ═══════════════════════════════════════════════════════════════

def build_yard_features():
    """堆场利用率时间序列 — 从移动事件累计进出推导 + 堆场容量"""
    print("\n" + "=" * 60)
    print("特征 3：堆场利用率（移动事件流 → 累计占用推导）")
    print("=" * 60)

    # ── 堆场总容量（从07_yard_cells取每日箱位数） ──
    # 注意：07_yard_cells是结构定义表，不是占用率，但可用作容量基准
    print("\n  加载堆场容量...")
    cells = pd.read_parquet(PROC / "07_yard_cells.parquet")
    cells['DAY'] = pd.to_datetime(cells['DAY'], errors='coerce')
    total_cells = cells['YARDCELL'].nunique()

    # 按 lane 统计容量（供区域分析用）
    lane_capacity = cells.groupby('YARDLANENO')['YARDCELL'].nunique().reset_index()
    lane_capacity.columns = ['yard_lane', 'capacity']
    print(f"  总箱位数: {total_cells:,} ({len(lane_capacity)} 个车道)")

    # ── 从移动事件流推导每日堆场占用 ──
    print("\n  加载移动事件 + 计算年初存量...")
    mvt = pd.read_parquet(PROC / "06_movement_events.parquet")
    mvt = safe_parse_dates(mvt, ["op_time", "in_time", "out_time"])

    # 用 containers 表计算年初存量（2024-01-01前入场且仍在场）
    containers = pd.read_parquet(PROC / "08_containers.parquet")
    containers = safe_parse_dates(containers, ["INTIME", "OUTTIME"])
    initial_occ = containers[
        (containers['INTIME'] < '2024-01-01') &
        ((containers['OUTTIME'] >= '2024-01-01') | containers['OUTTIME'].isna())
    ].shape[0]
    print(f"  年初存量: {initial_occ:,} 箱 ({initial_occ/total_cells*100:.1f}% 占用)")

    # ── 每日进出量 ──
    mvt['day'] = mvt['op_time'].dt.floor('D')

    # 判断进场/出场类型
    # 进场：G=Gate-in(闸口进场), B=From Berth(卸船) — 箱子进入堆场
    # 出场：B=Board Vessel(装船), G=Gate-out(闸口出场) — 箱子离开堆场
    # S=Shift(内部移箱)不改变总占用
    in_types = ['G', 'B']
    out_types = ['B', 'G']

    yard_flow = mvt.groupby('day').agg(
        total_moves=('op_time', 'count'),
        inbound=('in_type', lambda x: x.isin(in_types).sum()),
        outbound=('out_type', lambda x: x.isin(out_types).sum()),
    ).reset_index()

    # 累计占用（用年初存量校准）
    yard_flow = yard_flow.sort_values('day')
    yard_flow['net_flow'] = yard_flow['inbound'] - yard_flow['outbound']
    yard_flow['cumulative_occupancy'] = (initial_occ + yard_flow['net_flow'].cumsum()).clip(lower=0)
    yard_flow['utilization_rate'] = (yard_flow['cumulative_occupancy'] / total_cells * 100).clip(upper=100)

    # ── 日内高峰（小时级） ──
    mvt['hour_bin'] = mvt['op_time'].dt.floor('h')
    hourly = mvt.groupby('hour_bin').size().reset_index(name='moves_per_hour')
    hourly['day'] = hourly['hour_bin'].dt.floor('D')
    daily_peak = hourly.groupby('day').agg(
        peak_hour_moves=('moves_per_hour', 'max'),
        avg_hourly_moves=('moves_per_hour', 'mean'),
    ).reset_index()

    yard_flow = yard_flow.merge(daily_peak, on='day', how='left')
    yard_flow[['peak_hour_moves', 'avg_hourly_moves']] = yard_flow[['peak_hour_moves', 'avg_hourly_moves']].fillna(0)

    # 移动强度
    yard_flow['moves_per_cell'] = yard_flow['total_moves'] / total_cells

    # 滚动特征
    for col in ['utilization_rate', 'total_moves', 'moves_per_cell']:
        for w in [3, 7, 14]:
            yard_flow[f'{col}_ma{w}d'] = yard_flow[col].rolling(w, min_periods=1).mean()

    yard_flow['month'] = yard_flow['day'].dt.month
    yard_flow['weekday'] = yard_flow['day'].dt.dayofweek

    out_path = OUT / "10_yard_features.parquet"
    yard_flow.to_parquet(out_path, index=False)
    print(f"\n  输出: {out_path}")
    print(f"  维度: {yard_flow.shape[0]:,} × {yard_flow.shape[1]}")
    print(f"  列: {list(yard_flow.columns)}")
    print(f"  占用率范围: {yard_flow['utilization_rate'].min():.1f}% ~ {yard_flow['utilization_rate'].max():.1f}%")
    print(f"  平均占用率: {yard_flow['utilization_rate'].mean():.1f}%")

    return yard_flow


# ═══════════════════════════════════════════════════════════════
# 4. 集装箱停留时间分析（全新特征）
# ═══════════════════════════════════════════════════════════════

def build_dwell_features():
    """集装箱停留时间分析特征"""
    print("\n" + "=" * 60)
    print("特征 3b：集装箱停留时间")
    print("=" * 60)

    containers = pd.read_parquet(PROC / "08_containers.parquet")

    # 停留时间（过滤合理范围：1h~365d，排除远期预约）
    containers['dwell_hours'] = (containers['OUTTIME'] - containers['INTIME']).dt.total_seconds() / 3600
    valid = (containers['dwell_hours'] >= 1) & (containers['dwell_hours'] <= 8760)

    # 每日汇总（仅用有效停留时间）
    daily_dwell = containers[valid].groupby(containers[valid]['INTIME'].dt.floor('D')).agg(
        containers_arrived=('CONTAINERID', 'count'),
        dwell_mean_h=('dwell_hours', 'mean'),
        dwell_median_h=('dwell_hours', 'median'),
        dwell_p25_h=('dwell_hours', lambda x: x.quantile(0.25)),
        dwell_p75_h=('dwell_hours', lambda x: x.quantile(0.75)),
        dwell_gt48h_ratio=('dwell_hours', lambda x: (x > 48).mean()),
        dwell_gt72h_ratio=('dwell_hours', lambda x: (x > 72).mean()),
    ).reset_index()
    daily_dwell.columns = ['day', 'containers_arrived', 'dwell_mean_h',
                           'dwell_median_h', 'dwell_p25_h', 'dwell_p75_h',
                           'dwell_gt48h_ratio', 'dwell_gt72h_ratio']

    out_path = OUT / "10_dwell_features.parquet"
    daily_dwell.to_parquet(out_path, index=False)
    print(f"\n  输出: {out_path}")
    print(f"  维度: {daily_dwell.shape[0]:,} × {daily_dwell.shape[1]}")
    print(f"  列: {list(daily_dwell.columns)}")
    print(f"  平均停留: {daily_dwell['dwell_mean_h'].mean():.1f}h")

    return daily_dwell


# ═══════════════════════════════════════════════════════════════
# 5. PPO 状态空间（更新：使用真实堆场占用率）
# ═══════════════════════════════════════════════════════════════

def build_ppo_state_space(yard, dwell):
    """PPO 协同环境状态空间 — 每6小时一个时间步"""
    print("\n" + "=" * 60)
    print("特征 4：PPO 状态空间")
    print("=" * 60)

    berth = pd.read_parquet(OUT / "10_berth_features.parquet")
    stowage = pd.read_parquet(OUT / "10_stowage_features.parquet")

    # 时间步（6h）
    berth['time_bin'] = pd.to_datetime(berth['actual_berth']).dt.floor('6h')

    berth_agg = berth.groupby('time_bin').agg(
        berth_occupancy=('berth_plan_no', 'count'),
        avg_duration_h=('actual_duration_h', 'mean'),
        avg_delay_h=('berth_delay_h', 'mean'),
        large_vessels=('is_large', 'sum'),
        total_boxes=('total_boxes', 'sum'),
    ).reset_index().sort_values('time_bin')

    # 堆场占用率（从每日插值到6h）
    yard_6h = yard[['day', 'utilization_rate', 'total_moves', 'moves_per_cell']].copy()
    yard_6h['time_bin'] = yard_6h['day']
    # 用前向填充近似6h粒度
    all_bins = pd.date_range(start=yard_6h['time_bin'].min(), end=yard_6h['time_bin'].max(), freq='6h')
    yard_6h = yard_6h.set_index('time_bin').reindex(all_bins, method='ffill').reset_index()
    yard_6h.rename(columns={'index': 'time_bin'}, inplace=True)

    # 停留时间（同样插值到6h）
    dwell_6h = dwell[['day', 'dwell_mean_h', 'dwell_gt48h_ratio']].copy()
    dwell_6h['time_bin'] = dwell_6h['day']
    dwell_6h = dwell_6h.set_index('time_bin').reindex(all_bins, method='ffill').reset_index()
    dwell_6h.rename(columns={'index': 'time_bin'}, inplace=True)

    # 合并
    state = berth_agg.merge(yard_6h, on='time_bin', how='outer').sort_values('time_bin')
    state = state.merge(dwell_6h[['time_bin', 'dwell_mean_h', 'dwell_gt48h_ratio']],
                        on='time_bin', how='left')
    state = state.fillna(0)

    # 滞后特征
    for col in ['berth_occupancy', 'utilization_rate', 'avg_duration_h',
                'total_moves', 'dwell_mean_h']:
        if col in state.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                state[f'{col}_lag{lag}'] = state[col].shift(lag)

    out_path = OUT / "10_ppo_state_space.parquet"
    state.to_parquet(out_path, index=False)
    print(f"\n  输出: {out_path}")
    print(f"  维度: {state.shape[0]:,} × {state.shape[1]}")
    print(f"  列: {list(state.columns[:14])}...")
    print(f"  时间范围: {state['time_bin'].min()} ~ {state['time_bin'].max()}")
    print(f"  时间步数: {len(state):,}")

    return state


# ═══════════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("Step 10：特征工程 v2 (整合 Oracle 新数据)")
    print("=" * 60)

    t0 = pd.Timestamp.now()

    stowage = build_stowage_features()
    berth = build_berth_features()
    yard = build_yard_features()
    dwell = build_dwell_features()
    ppo = build_ppo_state_space(yard, dwell)

    elapsed = (pd.Timestamp.now() - t0).total_seconds()

    print("\n" + "=" * 60)
    print("  ✅ 特征工程完成!")
    print("=" * 60)
    print(f"\n  耗时: {elapsed:.1f}s")
    print(f"\n  输出文件:")
    for f in sorted(OUT.glob("10_*.parquet")):
        sz = f.stat().st_size / 1024 / 1024
        print(f"    {f.name} ({sz:.1f} MB)")

    return stowage, berth, yard, dwell, ppo


if __name__ == "__main__":
    run()
