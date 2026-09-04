"""
第四章统一重跑脚本（gen=50, pop=60）
基于15_ga_rh_stowage.py的run()函数，修改GA参数为统一值
"""
import pandas as pd, numpy as np, time, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path.cwd()
SCRIPTS = ROOT / 'experiments' / 'chapter4' / 'scripts'
sys.stdout.reconfigure(line_buffering=True)

import importlib.util
spec = importlib.util.spec_from_file_location("grh_mod", SCRIPTS / "15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grh)

# 统一GA参数
UNIFORM_GEN = 50
UNIFORM_POP = 60

OUT = ROOT / 'output'
PROC = ROOT / 'data' / 'processed'

sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
bay = pd.read_parquet(PROC / '02_bay_structure.parquet')

# 测试船舶
TEST_SHIPS = [
    ('5830653246812', 'CNTIG', 2782),
    ('5830567479361', 'CNCT', 7300),
    ('5831334867883', 'MXNT', 13894),
    ('5832068726663', 'OFUT', 15258),
    ('5831575061746', 'CGAMV', 13830),
    ('5830653078367', 'APESP', 13892),
]

type_teu = {'CMPES':23112,'AFULN':17292,'HORA':16010,'CGPAN':15072,
            'EAOT':14026,'CGASK':12917,'ULX':13167,'CMCAS':11388,
            'CGRIG':10034,'H7E':9087,'HHCB':6542,'AKBRCL':5086,
            'KSL':3105,'HMB':2817,'FPCE':1773}

np.random.seed(42)
all_results = []
t0_all = time.time()

for idx, (bpn, ship_name, teu) in enumerate(TEST_SHIPS):
    print(f'\n{"="*60}', flush=True)
    print(f'[{idx+1}/{len(TEST_SHIPS)}] {ship_name}', flush=True)
    t0 = time.time()

    # 取船舶数据
    ship_df = sf[sf['berth_plan_no'] == bpn].copy()
    if len(ship_df) == 0:
        ship_df = sf[sf['vessel_code'] == ship_name].copy()
    n_boxes = len(ship_df)
    
    # 匹配bay（与原脚本逻辑一致）
    ename = str(ship_df['e_vessel_name'].iloc[0]).strip() if 'e_vessel_name' in ship_df.columns else ship_name
    matched_bay = bay[bay['VESSELTYPECODE'] == ename].copy()
    if len(matched_bay) < n_boxes:
        best_code = min(type_teu.keys(), key=lambda c: abs(type_teu[c] - teu))
        matched_bay = bay[bay['VESSELTYPECODE'] == best_code].copy()
        print(f'  bay匹配: {ename} → {best_code}', flush=True)
    
    if len(matched_bay) < n_boxes:
        print(f'  ❌ 箱位不足: {n_boxes}箱 > {len(matched_bay)}slot', flush=True)
        continue
    
    n_slots = len(matched_bay)
    
    vessel_info = pd.Series({
        'max_teu': teu, 'e_vessel_name': ename,
        'length': ship_df['length'].iloc[0] if 'length' in ship_df else 0,
        'width': ship_df['width'].iloc[0] if 'width' in ship_df else 0,
    })
    
    prob = grh.VesselProblem(ename, bpn, ship_df, matched_bay, vessel_info)
    print(f'  {n_boxes}箱, {n_slots}slot, GA: pop={UNIFORM_POP}, gen={UNIFORM_GEN}', flush=True)
    
    optimizer = grh.GARHOptimizer(prob, pop_size=UNIFORM_POP, generations=UNIFORM_GEN)
    try:
        result = optimizer.optimize(verbose=True)
    except Exception as e:
        print(f'  ❌ 失败: {e}', flush=True)
        import traceback; traceback.print_exc()
        continue
    
    elapsed = time.time() - t0
    detail = result['best_detail']
    fitness = result['best_fitness']
    
    all_results.append({
        'berth_plan_no': bpn, 'vessel_code': ship_name,
        'max_teu': teu, 'n_containers': n_boxes, 'n_slots': n_slots,
        'fitness': fitness, 'rehandle': detail.get('rehandle', 0),
        'efficiency': detail.get('efficiency', 0),
        'balance': detail.get('balance', 0),
        'yard_collab': detail.get('yard_collab', 0),
        'penalty': detail.get('penalty', 0),
        'time_s': round(elapsed, 1),
        'n_gen': UNIFORM_GEN, 'n_pop': UNIFORM_POP,
        'experiment': 'exp1_unified',
    })
    
    print(f'  ✅ {ship_name}: fitness={fitness:.4f}, time={elapsed:.0f}s ({elapsed/60:.1f}min)', flush=True)
    pd.DataFrame(all_results).to_parquet(OUT / 'ch4_unified_results.parquet', index=False)

total = time.time() - t0_all
print(f'\n{"="*60}')
print(f'全部完成: {len(all_results)}/{len(TEST_SHIPS)} 艘船')
print(f'总时间: {total:.0f}s ({total/60:.1f}min)')
print(f'{"="*60}')
if all_results:
    print('\n结果汇总:')
    print(f'{"船名":<8} {"箱量":>6} {"fitness":>10} {"时间(s)":>8}')
    print('-'*35)
    for r in all_results:
        print(f'{r["vessel_code"]:<8} {r["n_containers"]:>6.0f} {r["fitness"]:>10.4f} {r["time_s"]:>8.0f}')
