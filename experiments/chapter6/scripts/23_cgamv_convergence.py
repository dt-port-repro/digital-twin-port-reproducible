"""
CGAMV收敛曲线 — GA-RH vs 纯GA × 各5轮 (pop=60, gen=30)
"""
import pandas as pd, numpy as np, time, sys
from pathlib import Path
import importlib.util
sys.stdout.reconfigure(line_buffering=True)

PROC = Path('data/processed'); OUT = Path('output')
R = Path('experiments/chapter4/results'); R.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("grh","scripts/15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec); spec.loader.exec_module(grh)

sf = pd.read_parquet(OUT/'10_stowage_features.parquet')
bay = pd.read_parquet(PROC/'02_bay_structure.parquet')
tmap = {'CMPES':23112,'AFULN':17292,'HORA':16010,'CGPAN':15072,'EAOT':14026,
        'CGASK':12917,'ULX':13167,'CMCAS':11388,'CGRIG':10034,'H7E':9087,
        'HHCB':6542,'AKBRCL':5086,'KSL':3105,'HMB':2817,'FPCE':1773}

ename, bpn = 'CGAMV', '5831575061746'
print(f"Loading {ename}...", flush=True)
ship = sf[sf['berth_plan_no']==bpn].copy()
teu = ship['max_teu'].iloc[0]
matched = bay[bay['VESSELTYPECODE']==ename]
if len(matched) < len(ship):
    best = min(tmap.keys(), key=lambda c: abs(tmap[c]-teu))
    matched = bay[bay['VESSELTYPECODE']==best]
    print(f"  fallback: {best} ({tmap[best]} TEU)", flush=True)
vi = pd.Series({'max_teu':teu,'e_vessel_name':ename})
prob = grh.VesselProblem(ename, bpn, ship, matched, vi)
print(f"  {prob.n_container}箱, {prob.n_slot} slots", flush=True)

N_RUNS = 3  # 3×2=6 runs, ~70min
seeds = [42, 123, 256]
print(f"\n=== CGAMV收敛: GA-RH × {N_RUNS}, PureGA × {N_RUNS} (不同随机序列) ===\n", flush=True)
rows = []
for run_i in range(N_RUNS):
    np.random.seed(seeds[run_i])
    for label, skip in [('GA-RH', False), ('PureGA', True)]:
        opt = grh.GARHOptimizer(prob, pop_size=60, generations=30)
        t0 = time.time()
        res = opt.optimize(verbose=False, skip_heuristics=skip)
        dt = time.time()-t0
        d = res['best_detail']
        h = res['history']
        rows.append({'method':label,'run':run_i,'fitness':res['best_fitness'],
                     'f1':float(d.get('rehandle',0)),'f2':float(d.get('efficiency',0)),
                     'f3':float(d.get('balance',0)),'penalty':float(d.get('penalty',0)),
                     'time_s':round(dt,1),'history':str(h)})
        print(f"  {label}#{run_i}: fit={res['best_fitness']:.4f} f1={float(d.get('rehandle',0)):.4f} ({dt:.0f}s)", flush=True)
        # Don't reset seed - let PureGA run with natural continuation

pd.DataFrame(rows).to_parquet(R/'convergence_cgamv.parquet', index=False)
print(f"\nSaved convergence_cgamv.parquet", flush=True)

# Series for plotting
hist_rows = []
for r in rows:
    h = eval(r['history'])
    for gen, val in enumerate(h):
        hist_rows.append({'method':r['method'],'run':r['run'],'gen':gen,'fitness':val})
pd.DataFrame(hist_rows).to_parquet(R/'convergence_cgamv_series.parquet', index=False)
print(f"Saved convergence_cgamv_series.parquet ({len(hist_rows)} rows)", flush=True)
print("✅ CGAMV收敛完成", flush=True)
