#!/usr/bin/env python3
"""Recompute one complete Q15 hopping g/seed case without historical caches.

This CPU entry uses the delivered and independently checked original C++ kernel
and the unchanged validation/refit functions. Full cases are expensive. Results
are written to runs/, never to manuscript source data or research directories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g',type=float,default=.84)
    p.add_argument('--seed',type=int,choices=[0,10,20,30,40,50],default=0)
    p.add_argument('--grid',choices=['six','main'],default='six')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--chunk-size',type=int,default=8)
    p.add_argument('--features-only',action='store_true')
    args=p.parse_args()
    if args.threads<1 or args.chunk_size<1:p.error('threads/chunk-size must be positive')
    if not 0<=args.g<=.98 or abs(args.g/.02-round(args.g/.02))>1e-8:p.error('g must be on the frozen 0.02 grid')
    if args.grid=='main' and abs(args.g-.84)>1e-10:p.error('main application uses fixed g=0.84')
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    root=Path(__file__).resolve().parents[1];code=root/'code'
    sys.path.insert(0,str(code/'BFS-QRC/hopping_gscan_fixed_p_20260908'))
    import backend
    import numpy as np
    import pandas as pd
    from a01_core import load_config,reset_suffix_weights,_best_svr,_best_ridge,_refit_svr,_refit_ridge,regression_metrics,TARGET_NAMES
    cfg=load_config()
    data_path=code/'Transition/shnitsel_a01_surface_hopping/data/processed/a01_windows.npz'
    with np.load(data_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    ps=tuple(cfg.p_grid) if args.grid=='main' else (0.,.2,.4,.6,.8,1.)
    out=root/'runs/hopping'/args.grid/f'g{args.g:.2f}'/f'seed{args.seed:03d}'
    out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),code/'BFS-QRC/hopping_gscan_fixed_p_20260908/backend.py',
             code/'BFS-QRC/hopping_gscan_fixed_p_20260908/cpu_kernel.cpp',code/'Transition/shnitsel_a01_surface_hopping/a01_core.py',
             code/'spatrace_dtc_qrc/01_algorithm_validation/hamiltonian_dtc_qrc.py',data_path]
    contract=dict(g=args.g,seed=args.seed,p=list(ps),q=15,t=20,W=.5,L=3,
      reset='between events only; no reset during TM3',
      sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    lock=out/'run_contract.json'
    if lock.exists():assert json.loads(lock.read_text())==contract,'Source/input changed; use a new output directory'
    else:lock.write_text(json.dumps(contract,indent=2)+'\n')
    kernel=root/'runs/cpu_kernel';kernel.mkdir(parents=True,exist_ok=True)
    shutil.copy2(code/'BFS-QRC/hopping_gscan_fixed_p_20260908/cpu_kernel.cpp',kernel/'cpu_kernel.cpp');backend.HERE=kernel
    weights=np.stack([reset_suffix_weights(20,p) for p in ps])
    arrays=np.empty((len(ps),len(data['tokens']),360),np.float32)
    chunks=out/'chunks';chunks.mkdir(exist_ok=True)
    for start in range(0,len(data['tokens']),args.chunk_size):
        end=min(start+args.chunk_size,len(data['tokens']));file=chunks/f'{start:04d}_{end:04d}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as z:values=z['features']
        else:
            bank,info=backend.suffixes(data['tokens'][start:end],g=args.g,seed=args.seed,threads=args.threads)
            values=np.tensordot(weights,bank,axes=(1,0)).astype(np.float32)
            tmp=file.with_suffix('.partial.npz');np.savez_compressed(tmp,features=values);tmp.replace(file)
            file.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
        assert values.shape==(len(ps),end-start,360) and np.isfinite(values).all()
        arrays[:,start:end]=values;print(f'Features {end}/{len(data["tokens"])}',flush=True)
    if args.features_only:return
    tr,va,te=[data['split']==i for i in range(3)]
    truth=data['targets'].astype(float)
    dec=cfg.payload['decoders'];alphas=dec['ridge_alpha_grid']
    candidates=[(c,g,e) for c in dec['svr_C_grid'] for g in dec['svr_gamma_scale_grid'] for e in dec['svr_epsilon_grid']]
    rows=[];predictions=[]
    for pi,pvalue in enumerate(ps):
        for readout,width in [('final',120),('tm3',360)]:
            x=arrays[pi,:,:width]
            for decoder in ('linear','rbf_svr'):
                for ti,target in enumerate(TARGET_NAMES):
                    y=truth[:,ti]
                    if decoder=='linear':
                        selected=_best_ridge(x[tr],y[tr],x[va],y[va],alphas)
                        pred=_refit_ridge(x[tr],y[tr],x[va],y[va],x[te],selected['alpha'])
                    else:
                        selected=_best_svr(x[tr],y[tr],x[va],y[va],candidates,256.)
                        pred=_refit_svr(x[tr],y[tr],x[va],y[va],x[te],selected,256.)
                    rows.append(dict(g=args.g,seed=args.seed,p=pvalue,target=target,readout=readout,decoder=decoder,
                                     feature_dim=width,**selected,**regression_metrics(y[te],pred)))
                    predictions.append(np.asarray(pred,np.float32))
                    print(f'Decoded {len(rows)}/{len(ps)*16}',flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    np.savez_compressed(out/'predictions.npz',predictions=np.stack(predictions),
        target=np.array([r['target'] for r in rows]),readout=np.array([r['readout'] for r in rows]),
        decoder=np.array([r['decoder'] for r in rows]),p=np.array([r['p'] for r in rows]),
        target_values=truth[te].astype(np.float32),trajectory_id=data['trajectory_id'][te],endpoint=data['endpoint'][te])
    print(out)

if __name__=='__main__':main()
