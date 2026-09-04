"""
第六章实验一键复现
"""
import subprocess, sys, time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / 'scripts'
ROOT = Path(__file__).resolve().parent.parent.parent

EXPERIMENTS = [
    # === 核心实验数据验证（读取规范化parquet文件，输出论文表格）===
    ('论文数据验证', 'verify_paper_data.py'),

    # === 补充实验（GA-RH收敛性 + 可扩展性）===
    ('可扩展性测试', 'exp6_scalability.py'),
    # 收敛性分析（23_cgamv_convergence.py）耗时~15min，单独运行：
    # python experiments/chapter6/scripts/23_cgamv_convergence.py
]

def run_all():
    print('=' * 60)
    print('第六章实验一键复现')
    print('=' * 60)
    total_ok = 0
    total_fail = 0
    for name, script in EXPERIMENTS:
        print(f'\n{"─"*60}')
        print(f'[{name}] 运行 {script}...')
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / script)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f'  ✅ 完成 ({elapsed:.0f}s)')
            total_ok += 1
        else:
            print(f'  ❌ 失败 (returncode={result.returncode}, {elapsed:.0f}s)')
            print(f'  stderr: {result.stderr[:300]}')
            total_fail += 1

    print(f'\n{"="*60}')
    print(f'结果: {total_ok} 成功, {total_fail} 失败 (共 {len(EXPERIMENTS)} 项)')

if __name__ == '__main__':
    run_all()
