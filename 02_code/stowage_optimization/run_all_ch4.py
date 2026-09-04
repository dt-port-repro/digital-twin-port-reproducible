"""第四章全套实验复现脚本（gen=50, pop=60统一参数）
基于ga_rh_algorithm.py，跑所有6条船的各项实验
输出保存到 03_results/canonical/

因MCT原始数据受使用协议限制，此脚本需在有数据访问权限的环境运行。
预计算结果已存储在 03_results/canonical/exp*.parquet，可直接验证。
"""
import pandas as pd, numpy as np, time, sys, warnings, json
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / '02_code' / 'stowage_optimization'
sys.path.insert(0, str(CODE))

try:
    import ga_rh_algorithm as grh
except (ImportError, FileNotFoundError) as e:
    print(f'警告: 无法加载GA-RH算法: {e}')
    print('因MCT原始数据不可用，此脚本无法运行。')
    print('预计算结果位于 03_results/canonical/exp*.parquet')
    print('运行 python verify_replication.py 验证结果一致性')
    sys.exit(0)

UNIFORM_GEN = 50
UNIFORM_POP = 60

DATA = ROOT / 'data' / 'processed'
OUTPUT = ROOT / 'output'
EXPORT = ROOT / '03_results' / 'canonical'
EXPORT.mkdir(parents=True, exist_ok=True)

try:
    sf = pd.read_parquet(OUTPUT / '10_stowage_features.parquet')
    bay = pd.read_parquet(DATA / '02_bay_structure.parquet')
except FileNotFoundError as e:
    print(f'数据文件缺失: {e}')
    print('请确保原始MCT数据已放置在 data/processed/ 和 output/ 目录下')
    sys.exit(1)

TEST_SHIPS = [
    ('5830653246812', 'CNTIG', 2782),
    ('5830091570670', 'CNCT', 3125),
    ('5830724359571', 'MXNT', 4402),
    ('5843115885494', 'OFUT', 8229),
    ('5831623918662', 'CGAMV', 10358),
    ('5831687798789', 'APESP', 11926),
]
