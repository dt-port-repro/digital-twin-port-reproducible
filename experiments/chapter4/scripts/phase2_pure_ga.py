"""Phase 2: 10条新船 纯GA（跳过规则启发式）"""
import pandas as pd, numpy as np, time, warnings, importlib.util
from pathlib import Path
warnings.filterwarnings('ignore')

PROC, OUT = Path('data/processed'), Path('output')
spec = importlib.util.spec_from_file_location("grh", Path.cwd() / "scripts" / "15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec); spec.loader.exec_module(grh)

sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
bay = pd.read_parquet(PROC / '02_bay_structure.parquet')
type_teu = {'CMPES':23112,'AFULN':17292,'HORA':16010,'CGPAN':15072,'EAOT':14026,'CGASK':12917,
            'ULX':13167,'CMCAS':11388,'CGRIG':10034,'H7E':9087,'HHCB':6542,'AKBRCL':5086,'KSL':3105,'HMB':2817,'FPCE':1773}

new_ships = [
    ('9NI','5831154035520'),('ABCN','5832329860472'),('AKBRCL','5831832161821'),
    ('CAMNL','5831781433667'),('CNTT','5830785036267'),('CGARG','5831138788929'),
    ('XOCE','5832370859748'),('CGCRS','5831260650104'),('CGADN','5831444223264'),('CGINY','5831060515404'),
]
results = []

for i,(ename,bpn) in enumerate(new_ships):
    ship = sf[sf['berth_plan_no']==bpn]; n=len(ship); teu=ship['max_teu'].iloc[0]; npod=ship['pod'].nunique()
    mb = bay[bay['VESSELTYPECODE']==ename]
    if len(mb)<n:
        best=min(type_teu.keys(),key=lambda c:abs(type_teu[c]-teu))
        mb=bay[bay['VESSELTYPECODE']==best]
    vi = pd.Series({'max_teu':teu,'e_vessel_name':ename,'length':ship['length'].iloc[0] if 'length' in ship else 0,'width':ship['width'].iloc[0] if 'width' in ship else 0})
    prob = grh.VesselProblem(ename,bpn,ship,mb,vi)
    t0=time.time()
    opt = grh.GARHOptimizer(prob,pop_size=60,generations=80)
    res=opt.optimize(verbose=False,skip_heuristics=True)  # ← 关键：跳过规则启发式
    el=time.time()-t0
    det=res['best_detail']
    hist=res['history']
    row={'vessel_code':ename,'n_containers':n,'n_pod':npod,'n_slots':prob.n_slot,'max_teu':teu,
         'fitness':res['best_fitness'],'f1_rehandle':det.get('rehandle',0),'f2_efficiency':det.get('efficiency',0),
         'f3_balance':det.get('balance',0),'f4_yard_collab':det.get('yard_collab',0),'penalty':det.get('penalty',0),
         'time_s':round(el,1),'n_gen':80,'n_pop':60,'method':'pure_GA','convergence_start':hist[0],'convergence_end':hist[-1]}
    results.append(row)
    print(f'[{i+1}/10] {ename} {n}箱 纯GA fitness={row["fitness"]:.4f} {el:.0f}s')

pd.DataFrame(results).to_parquet(OUT/'ga_rh_results'/'phase2_pure_ga.parquet',index=False)
print(f'\n✅ Phase2 done: {len(results)} ships saved')
