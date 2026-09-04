"""统一标准重跑：编码实验(补齐B到5轮) + APESP鲁棒性(补齐到15轮)
基于ga_rh_algorithm.py，统一gen=50, pop=60
因MCT原始数据受使用协议限制，预计算结果在 03_results/canonical/*.parquet
"""
import pandas as pd, numpy as np, time, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
CODE = ROOT / '02_code' / 'stowage_optimization'
sys.path.insert(0, str(CODE))

try:
    import ga_rh_algorithm as grh
except (ImportError, FileNotFoundError):
    print('警告: 因MCT原始数据不可用，此脚本无法运行。')
    print('预计算结果位于 03_results/canonical/*.parquet')
    sys.exit(0)

UNIFORM_GEN = 50
UNIFORM_POP = 60
EXPORT = ROOT / '03_results' / 'canonical'
EXPORT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / 'data' / 'processed'
OUTPUT = ROOT / 'output'

try:
    sf = pd.read_parquet(OUTPUT / '10_stowage_features.parquet')
    bay = pd.read_parquet(DATA / '02_bay_structure.parquet')
except FileNotFoundError as e:
    print(f'数据文件缺失: {e}')
    sys.exit(1)

np.random.seed(42)
