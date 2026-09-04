"""
Run All — 一键运行完整数据流水线 → 实验结果 → 图表生成。

三条流水线：
  1. 数据清洗（raw → processed）：python run_all.py --pipeline data
  2. 全部实验（processed → canonical）：python run_all.py --pipeline experiments
  3. 图表生成（canonical → tables/figures）：python run_all.py --pipeline viz

  全部执行：python run_all.py --pipeline all

用法：
  python run_all.py --pipeline data          # 仅数据清洗
  python run_all.py --pipeline experiments   # 仅实验（依赖 data 完成）
  python run_all.py --pipeline viz           # 仅图表生成（依赖 experiments 完成）
  python run_all.py --pipeline all           # 全流程
  python run_all.py --pipeline data --steps 1-3  # 仅清洗步骤 1-3
"""
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def run_script(script_path, label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {script_path}")
    print(f"{'=' * 60}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if result.returncode != 0:
        print(f"  ❌ {label} 失败 (exit={result.returncode})")
        sys.exit(1)
    print(f"  ✅ {label} 完成")
    return True


def pipeline_data(steps=None, skip_step10=False):
    """数据清洗：raw → processed"""
    pipeline = {
        1: ("scripts/01_to_04_clean_basics.py", "Step 1-4: 基础数据清洗"),
        5: ("scripts/05_merge_containers.py", "Step 5: 集装箱合并"),
        6: ("scripts/06_merge_movement.py", "Step 6: 移动事件合并"),
        7: ("scripts/07_parse_xce_dump.py", "Step 7: xce.dmp 解析"),
        8: ("scripts/08_build_daily_inventory.py", "Step 8: 日库存构建"),
        9: ("scripts/09_validate_integrity.py", "Step 9: 完整性验证"),
        10: ("scripts/10_feature_engineering.py", "Step 10: 特征工程"),
    }
    skipped = {10} if skip_step10 else set()
    if steps:
        for s in sorted(steps):
            if s in pipeline and s not in skipped:
                run_script(ROOT / pipeline[s][0], pipeline[s][1])
            elif s in skipped:
                print(f"  ⏭ {pipeline[s][1]} (--skip-step10)")
    else:
        for s in sorted(pipeline):
            if s not in skipped:
                run_script(ROOT / pipeline[s][0], pipeline[s][1])
            else:
                print(f"  ⏭ {pipeline[s][1]} (--skip-step10)")
    print("\n  ✅ 数据清洗流水线完成")


def pipeline_experiments():
    """全部实验：processed → canonical"""
    exp_steps = [
        (ROOT / "experiments" / "chapter4" / "scripts" / "run_ch4_unified.py",
         "第四章 GA-RH 配载实验"),
        (ROOT / "experiments" / "chapter5" / "scripts" / "16_lstm_prediction.py",
         "第五章 LSTM 预测实验 (16_lstm_prediction.py)"),
        (ROOT / "experiments" / "chapter5" / "scripts" / "18_yard_selection.py",
         "第五章 堆场选位实验 (18_yard_selection.py)"),
    ]
    for script_path, label in exp_steps:
        if script_path.exists():
            run_script(script_path, label)
        else:
            print(f"  ⚠️ 跳过（脚本不存在）: {label}")
    print("\n  ✅ 实验流水线完成")


def pipeline_viz():
    """图表生成：canonical → tables + figures"""
    viz_scripts = [
        (ROOT / "02_code" / "visualization" / "generate_tables.py",
         "第四章全部表格 (表4.5-4.14)"),
        (ROOT / "02_code" / "visualization" / "gen_fig47.py",
         "图4.7 收敛曲线"),
        (ROOT / "02_code" / "visualization" / "gen_fig48_49.py",
         "图4.8-4.9 参数敏感性 + 种群规模"),
        (ROOT / "02_code" / "visualization" / "gen_fig5_2.py",
         "图5.2 预测训练曲线"),
        (ROOT / "02_code" / "visualization" / "gen_fig5_4.py",
         "图5.4 选位对比"),
        (ROOT / "02_code" / "visualization" / "gen_fig5_6.py",
         "图5.6 PPO奖励曲线"),
        (ROOT / "02_code" / "visualization" / "gen_fig6_1.py",
         "图6.2 常规场景对比"),
        (ROOT / "02_code" / "visualization" / "gen_fig6_2.py",
         "图6.3 高峰场景对比"),
        (ROOT / "02_code" / "visualization" / "gen_fig6_3.py",
         "图6.4 异常场景对比"),
        (ROOT / "02_code" / "visualization" / "gen_table6_4.py",
         "表6.4 统计检验"),
        (ROOT / "02_code" / "visualization" / "gen_des_scenario_tables.py",
         "表6.1-6.3 + S14-S16 DES表"),
    ]
    for script_path, label in viz_scripts:
        if script_path.exists():
            run_script(script_path, label)
        else:
            print(f"  ⚠️ 跳过（脚本不存在）: {label}")
    print("\n  ✅ 图表生成流水线完成")


def main():
    parser = argparse.ArgumentParser(description="端到端复现流水线")
    parser.add_argument("--pipeline", type=str, default="all",
                        choices=["data", "experiments", "viz", "all"],
                        help="要执行的流水线")
    parser.add_argument("--steps", type=str, default="",
                        help="数据清洗步骤范围，如 '1-3' 或 '5,6,7'")
    parser.add_argument("--skip-step10", action="store_true",
                        help="跳过Step 10特征工程（当output/10_stowage_features.parquet已存在时使用）")
    args = parser.parse_args()

    steps = set()
    if args.steps:
        for part in args.steps.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                steps.update(range(int(a), int(b) + 1))
            else:
                steps.add(int(part))

    if args.pipeline in ("data", "all"):
        pipeline_data(steps if steps else None, skip_step10=args.skip_step10)
    if args.pipeline in ("experiments", "all"):
        pipeline_experiments()
    if args.pipeline in ("viz", "all"):
        pipeline_viz()

    print(f"\n{'=' * 60}")
    print("  🎉 流水线全部执行完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
