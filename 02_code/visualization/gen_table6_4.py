"""
Generate/verify Table 6.4 Statistical significance
Data source: pre-computed table_6_4.csv
"""
import pandas as pd
from pathlib import Path

_VIZ = Path(__file__).resolve().parent.parent.parent
CANON = _VIZ / '03_results' / 'canonical'
TABLES = _VIZ / '03_results' / 'tables'

# Read pre-computed table
src = TABLES / 'table_6_4.csv'
if src.exists():
    df = pd.read_csv(src)
    header = f"{'scenario':<6} {'indicator':<14} {'config':<6} {'mean':<8} {'improve':<10} {'d':<8}"
    print('Table 6.4  Statistical significance (pre-computed)')
    print()
    print('-' * len(header))
    for _, r in df.iterrows():
        print(f"{r['scenario']:<6} {r['indicator']:<14} {r['config']:<6} "
              f"{r['mean']:<8} {r['improvement']:<10} {r['cohens_d']:<8}")
    print()
    print(f'Done. {len(df)} rows.')
else:
    print('Warning: table_6_4.csv not found, cannot generate.')
    print('The table was pre-computed from original simulation runs.')
