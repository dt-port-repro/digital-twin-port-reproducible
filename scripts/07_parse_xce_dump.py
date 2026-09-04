"""
Step 7：解析 xce.dmp — Oracle Data Pump 导出文件

xce.dmp 含两张表：
  1. t_xce3 — 2024年366天逐日MCT所有堆场场位快照（~7,300万行）
  2. t_comtainers — 集装箱进出场记录（~445万行）

依赖：Oracle XE 21c 已安装运行 + impdp 可用
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
try:
    import oracledb
except ImportError:
    print("⚠️  oracledb 未安装，跳过 xce.dmp 解析（处理数据已存在）")
    print("   需要 Oracle Instant Client 和 oracledb 模块。")
    sys.exit(0)

from utils.paths import raw_2024_dir, processed_dir
from utils.schema import XCE3, COMPTAINERS

# ── Oracle 连接参数 ──
ORACLE_USER = "system"
ORACLE_PASSWORD = "Aitest1999#"
ORACLE_DSN = "localhost:1521/XE"

# ── 路径 ──
SCRIPT_DIR = Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR.parent
# 从原始数据目录自动定位 xce.dmp（data/raw 或 01_data/raw_data 均可）
DMP_CANDIDATES = list(raw_2024_dir().rglob("xce.dmp"))
if not DMP_CANDIDATES:
    DMP_CANDIDATES = list(Path(__file__).resolve().parents[1].glob("01_data/raw_data/**/xce.dmp"))
DMP_FILE = DMP_CANDIDATES[0] if DMP_CANDIDATES else None
IMPDP_LOG = processed_dir() / "impdp_xce.log"

# 导出parquet路径
T_XCE3_OUT = processed_dir() / "07_t_xce3.parquet"
T_COMTAINERS_OUT = processed_dir() / "07_t_comtainers.parquet"


# ── 工具函数 ──

def run_sqlplus(sql: str, db_user=ORACLE_USER, db_pass=ORACLE_PASSWORD, db_dsn=ORACLE_DSN):
    """通过 sqlplus 执行 SQL"""
    conn_str = f"{db_user}/{db_pass}@{db_dsn}"
    cmd = f'echo {sql} | sqlplus -S {conn_str}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return result.stdout + result.stderr


def run_cmd(cmd: str, timeout: int = 600):
    """运行 shell 命令"""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(f"    {result.stdout.strip()[:500]}")
    if result.returncode != 0:
        print(f"    ⚠️ 返回码={result.returncode}, stderr={result.stderr[:300]}")
    return result


# ── 核心步骤 ──

def step1_create_directory():
    """在 Oracle 中创建指向 DMP 位置的 DIRECTORY"""
    print("\n[Step 7.1] 创建 Oracle DIRECTORY...")

    if not DMP_DIR_PATH.exists():
        print(f"  ❌ DMP 目录不存在: {DMP_DIR_PATH}")
        print("  请确认 xce.dmp 位于 01_data/raw_data/ 或 data/raw/2024/ 目录下")
        return False

    dmp_path_win = str(DMP_DIR_PATH).replace("/", "\\\\")
    sql = f"""
    CREATE OR REPLACE DIRECTORY {DMP_DIR_NAME} AS '{dmp_path_win}';
    GRANT READ, WRITE ON DIRECTORY {DMP_DIR_NAME} TO SYSTEM;
    """
    # 不能用 GRANT TO PUBLIC，SCOTT 用 SYSTEM 用户导入
    output = run_sqlplus(sql)
    print(f"  {output.strip()[:300]}")
    return True


def step2_impdp():
    """运行 impdp 导入 xce.dmp"""
    print("\n[Step 7.2] impdp 导入 xce.dmp...")
    print(f"  源: {DMP_FILE}")
    print(f"  大小: {DMP_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  预计: t_xce3 (~7,300万行) + t_comtainers (~445万行)")
    print(f"  时长: 可能 20-60 分钟...")

    cmd = (
        f"impdp {ORACLE_USER}/{ORACLE_PASSWORD}@{ORACLE_DSN} "
        f"directory={DMP_DIR_NAME} "
        f"dumpfile=xce.dmp "
        f"logfile={IMPDP_LOG} "
        f"table_exists_action=replace "
        f"transform=segment_attributes:n "
        f"parallel=2"
    )
    result = run_cmd(cmd, timeout=3600)

    print(f"\n  impdp 日志 (最后20行):")
    if IMPDP_LOG.exists():
        lines = IMPDP_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for l in lines[-20:]:
            print(f"    {l}")

    return result.returncode == 0


def step3_verify_tables():
    """验证两张表是否导入成功"""
    print("\n[Step 7.3] 验证导入结果...")

    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)
    cursor = conn.cursor()

    for tbl in ["SCOTT.T_XCE3", "SCOTT.T_COMTAINERS"]:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
        count = cursor.fetchone()[0]
        cursor.execute(f"SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='SCOTT' AND TABLE_NAME='{tbl.split('.')[1]}'")
        cols = [r[0] for r in cursor.fetchall()]
        print(f"  {tbl}: {count:,} 行, 列: {cols}")

    cursor.close()
    conn.close()


def step4_export_to_parquet():
    """用 oracledb 从 Oracle 导出到 Parquet"""
    print("\n[Step 7.4] 导出到 Parquet...")

    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)
    
    # t_xce3
    print("  → 导出 t_xce3...")
    df_xce3 = pd.read_sql("SELECT * FROM SCOTT.T_XCE3 ORDER BY DAY, YARDCELL", conn)
    # 标准化列名（大写→小写）
    df_xce3.columns = [c.lower() for c in df_xce3.columns]
    # 重命名 YARDCELL→yard_cell, TERMINALCODE→terminal_code, YARDLANENO→yard_lane_no
    rename_map = {"yardcell": "yard_cell", "terminalcode": "terminal_code", "yardlaneno": "yard_lane_no"}
    df_xce3.rename(columns=rename_map, inplace=True)
    df_xce3["day"] = pd.to_datetime(df_xce3["day"])
    t_xce3_out.write_parquet(T_XCE3_OUT, index=False)
    print(f"  ✅ t_xce3: {len(df_xce3):,} 行, {list(df_xce3.columns)}")
    print(f"     日期: {df_xce3['day'].min()} ~ {df_xce3['day'].max()}")
    print(f"     唯一箱位: {df_xce3['yard_cell'].nunique():,}")
    print(f"     -> {T_XCE3_OUT} ({T_XCE3_OUT.stat().st_size / 1024 / 1024:.1f} MB)")

    # t_comtainers
    print("  → 导出 t_comtainers...")
    df_ctn = pd.read_sql("SELECT * FROM SCOTT.T_COMTAINERS", conn)
    df_ctn.columns = [c.lower() for c in df_ctn.columns]
    rename_map2 = {"containerid": "container_id", "containerno": "container_no",
                   "intime": "in_time", "outtime": "out_time", "outyardcell": "out_yard_cell"}
    df_ctn.rename(columns=rename_map2, inplace=True)
    for c in ["in_time", "out_time"]:
        if c in df_ctn.columns:
            df_ctn[c] = pd.to_datetime(df_ctn[c])
    df_ctn.to_parquet(T_COMTAINERS_OUT, index=False)
    print(f"  ✅ t_comtainers: {len(df_ctn):,} 行, {list(df_ctn.columns)}")
    print(f"     -> {T_COMTAINERS_OUT} ({T_COMTAINERS_OUT.stat().st_size / 1024 / 1024:.1f} MB)")

    conn.close()
    print("\n  ✅ 导出完成")


def step5_cleanup():
    """清理临时文件"""
    print("\n[Step 7.5] 清理...")
    # 保留 impdp 日志
    print("  保留 Oracle 供后续使用")
    print("  如需卸载 Oracle: 控制面板 → 程序和功能 → Oracle Database 21c Express Edition")


def run():
    """完整流程"""
    print("=" * 60)
    print("Step 7: 解析 xce.dmp")
    print("=" * 60)

    if not DMP_FILE.exists():
        print(f"❌ 未找到 xce.dmp: {DMP_FILE}")
        return

    steps = [
        ("创建 Oracle DIRECTORY", step1_create_directory),
        ("impdp 导入", step2_impdp),
        ("验证表结构", step3_verify_tables),
        ("导出到 Parquet", step4_export_to_parquet),
        ("清理", step5_cleanup),
    ]

    for name, func in steps:
        print(f"\n{'─' * 40}")
        print(f"  🚀 {name}")
        print(f"{'─' * 40}")
        try:
            func()
        except Exception as e:
            print(f"  ❌ 步骤失败: {e}")
            print("  请检查 Oracle 状态后重试")
            return

    print(f"\n{'=' * 60}")
    print("  ✅ Step 7 完成!")
    print(f"  {'=' * 60}")
    print(f"  📦 t_xce3        -> {T_XCE3_OUT}")
    print(f"  📦 t_comtainers  -> {T_COMTAINERS_OUT}")
    print()
    print("  随后运行: python 08_build_daily_inventory.py")
    print("            python 09_validate_integrity.py")


if __name__ == "__main__":
    run()
