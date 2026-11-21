#!/usr/bin/env python3
"""Refit all 240 main-figure QRC/SVR cells from the supplied frozen features."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration'))
import numpy as np
import pandas as pd
from readouts import fit_svr


def case(job):
    dataset,t,seed=job
    base=ROOT/'code/BFS-QRC/FINAL/16_dissipative_protocol_migration/cache'
    with np.load(base/'tokens'/f'{dataset}_canonical_general_Q24_T30.npz',allow_pickle=False) as z:labels=z['labels']
    split=json.loads((base/'splits'/dataset/f'seed{seed:03d}.json').read_text())
    result=[]
    with np.load(ROOT/'data/frozen_graph_features'/dataset/f'T{t:02d}_seed{seed:03d}.npz',allow_pickle=False) as z:
        values=z['features'];ps=z['p_values']
    for p in (0.,.1):
        x=values[np.flatnonzero(np.isclose(ps,p))[0]]
        metric,_=fit_svr(x,labels,split,seed=seed)
        result.append(dict(dataset=dataset,t=t,seed=seed,p=p,system='closed_dtc_qrc' if p==0 else 'open_dtc_qrc',**metric))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    raw=pd.read_csv(ROOT/'source_data/main/graph_stream_q12_metrics_all_seeds.csv')
    ref=raw[raw.system.isin(['closed_dtc_qrc','open_dtc_qrc'])]
    jobs=list(ref[['dataset','t','seed']].drop_duplicates().itertuples(index=False,name=None))
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,batch in enumerate(pool.map(case,jobs),1):
            rows.extend(batch)
            if i%20==0:print(f'Checked {i}/{len(jobs)} feature cases',flush=True)
    new=pd.DataFrame(rows);keys=['dataset','t','seed','p','system']
    joined=new.merge(ref,on=keys,validate='one_to_one',suffixes=('_refit','_archived'))
    assert len(joined)==240
    for field in ('test_auc','val_auc','C','gamma','gamma_scale'):
        joined[field+'_absolute_error']=np.abs(joined[field+'_refit']-joined[field+'_archived'])
    joined.to_csv(ROOT/'audit/frozen_graph_refit_comparison.csv',index=False)
    errors={field:float(joined[field+'_absolute_error'].max()) for field in ('test_auc','val_auc','C','gamma','gamma_scale')}
    report=dict(passed=all(e<1e-10 for e in errors.values()),refit_cells=240,feature_cases=120,
                maximum_errors=errors,scope='All main graph QRC RBF-SVR cells; original frozen features, full validation grids, no quantum re-extraction')
    (ROOT/'audit/frozen_graph_refit_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
