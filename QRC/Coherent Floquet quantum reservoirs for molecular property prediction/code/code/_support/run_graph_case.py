#!/usr/bin/env python3
"""Recompute a graph classification case using the original frozen functions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m



def case_directory_name(args):
    """Name outputs by the complete request, including the reset grid."""
    config = json.dumps(vars(args), sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(config.encode('utf-8')).hexdigest()[:16]
    return f'{args.system}_width{args.width}_{args.decoder}_{args.features}_config{digest}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system',choices=['qrc','esn'],required=True)
    parser.add_argument('--dataset',choices=['BACE','BBBP'],default='BACE')
    parser.add_argument('--seed',type=int,choices=[0,10,20,30,40,50],default=0)
    parser.add_argument('--t',type=int,choices=[2,4,6,8,10,12,16,20,24,30],default=30)
    parser.add_argument('--width',type=int,choices=[78,500,1000,2000],default=78)
    parser.add_argument('--p',type=float,nargs='+',default=[0.,.1])
    parser.add_argument('--decoder',choices=['rbf_svr','linear'],default='rbf_svr')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--features',choices=['frozen','fresh'],default='frozen',help='frozen reproduces archived fits; fresh reruns float32 quantum extraction')
    args=parser.parse_args()
    if args.system=='qrc' and args.width!=78:parser.error('the graph QRC readout width is 78')
    if any(not 0<=p<1 for p in args.p):parser.error('graph reset scan uses 0 <= p < 1')
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    root=Path(__file__).resolve().parents[1];code=root/'code'
    mig=code/'BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration'
    legacy=code/'BFS-QRC/FINAL/16_dissipative_protocol_migration'
    sys.path.insert(0,str(mig))
    import numpy as np
    import pandas as pd
    import torch
    import protocol
    from hamiltonian_qrc import compute_suffix_banks,mix_suffix_bank
    from readouts import fit_svr,fit_linear_bank
    torch.set_num_threads(1)
    path=protocol.source_token_path(args.dataset,'canonical_general')
    with np.load(path,allow_pickle=False) as z:tokens=z['tokens'][:,:,:36];labels=z['labels']
    split_path=protocol.source_split_path(args.dataset,args.seed)
    split=json.loads(split_path.read_text())
    out=root/'runs/graph'/args.dataset/f'T{args.t:02d}'/f'seed{args.seed:03d}'/case_directory_name(args)
    out.mkdir(parents=True,exist_ok=True)
    code_files=[Path(__file__).resolve(),mig/'hamiltonian_qrc.py',mig/'readouts.py',mig/'protocol.py',legacy/'esn_features.py',legacy/'protocol.py',code/'spatrace_dtc_qrc/01_algorithm_validation/hamiltonian_dtc_qrc.py']
    contract=dict(arguments=vars(args),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in code_files})
    lock=out/'run_contract.json'
    if lock.exists():assert json.loads(lock.read_text())==contract,'Source/input/config changed; use a clean release extraction'
    else:lock.write_text(json.dumps(contract,indent=2)+'\n')
    if args.system=='qrc' and args.features=='frozen':
        frozen=root/f'data/frozen_graph_features/{args.dataset}/T{args.t:02d}_seed{args.seed:03d}.npz'
        with np.load(frozen,allow_pickle=False) as z:
            features=[]
            for pvalue in args.p:
                indices=np.flatnonzero(np.isclose(z['p_values'],pvalue,rtol=0,atol=1e-12))
                if len(indices)!=1:parser.error('p is not in the supplied frozen grid; use --features fresh')
                features.append((pvalue,z['features'][indices[0]]))
    elif args.system=='qrc':
        banks,_=compute_suffix_banks(tokens,q=12,t_values=(args.t,),seed=args.seed,device=args.device)
        features=[(p,mix_suffix_bank(banks[args.t],p)) for p in args.p]
    else:
        lp=load('release_esn_protocol',legacy/'protocol.py')
        lp.ESN_FEATURE_ROOT=root/'runs/graph_esn_cache'
        sys.modules['protocol']=lp
        try:esn=load('release_esn_features',legacy/'esn_features.py')
        finally:sys.modules['protocol']=protocol
        esn.T_VALUES=(args.t,)
        esn.compute_esn_all_t(args.dataset,'canonical_general',12,args.width,args.seed,device=args.device)
        with np.load(lp.esn_feature_path(args.dataset,'canonical_general',12,args.t,args.width,args.seed),allow_pickle=False) as z:x=z['features']
        features=[(None,x)]
    rows=[]
    for pvalue,x in features:
        name='esn' if pvalue is None else f'p{pvalue:g}'
        if args.decoder=='rbf_svr':
            metrics,art=fit_svr(x,labels,split,seed=args.seed)
            art['screen'].to_csv(out/(name+'_validation.csv'),index=False)
            scores=art['test_scores'];indices=art['test_indices']
        else:
            frame,art=fit_linear_bank(x[None,:,:],labels,split,seed=args.seed,device=args.device)
            metrics=frame.iloc[0].to_dict();scores=art['test_probabilities'][0];indices=art['test_indices']
        pd.DataFrame(dict(sample_index=indices,label=labels[indices],score=scores)).to_csv(out/(name+'_test_scores.csv'),index=False)
        rows.append(dict(system=args.system,dataset=args.dataset,seed=args.seed,t=args.t,width=args.width,p=pvalue,decoder=args.decoder,**metrics))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    print(pd.DataFrame(rows)[['system','dataset','t','seed','p','test_auc']].to_string(index=False))
    print(out)

if __name__=='__main__':main()
