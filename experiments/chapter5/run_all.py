"""
第五章实验一键复现
"""
import subprocess, sys, time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / 'scripts'
ROOT = Path(__file__).resolve().parent.parent.parent

EXPERIMENTS = [
    ('论文数据验证', 'verify_paper_data.py'),
]

def run_all():
    print('=' * 60)
    print('第五章实验一键复现')
    print('=' * 60)
    total_ok = total_fail = 0
    for name, script in EXPERIMENTS:
        print(f'\n{"-" * 60}')
        print(f'[{name}] 运行 {script}...')
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / script)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f'  OK ({elapsed:.0f}s)')
            print(result.stdout[-500:])
            total_ok += 1
        else:
            print(f'  FAIL (code={result.returncode})')
            print(f'  stderr: {result.stderr[:300]}')
            total_fail += 1
    print(f'\n{"=" * 60}')
    print(f'结果: {total_ok} 成功, {total_fail} 失败')

if __name__ == '__main__':
    run_all()
