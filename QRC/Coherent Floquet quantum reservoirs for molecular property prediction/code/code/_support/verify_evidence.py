#!/usr/bin/env python3
"""Read-only checks from packaged inputs/predictions/counts to manuscript tables."""
from pathlib import Path
import hashlib
import json
import os
import sys
import zipfile
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code';DATA=ROOT/'source_data'
SEEDS=[0,10,20,30,40,50]


def equal(a,b,tol=1e-11):np.testing.assert_allclose(np.asarray(a,float),np.asarray(b,float),rtol=tol,atol=tol)


def graph_inputs():
    reports=[]
    base=CODE/'BFS-QRC/FINAL/16_dissipative_protocol_migration/cache'
    for name,n,unique in [('BACE',1513,1513),('BBBP',2039,1975)]:
        with np.load(base/'tokens'/f'{name}_canonical_general_Q24_T30.npz',allow_pickle=False) as z:
            tokens=z['tokens'];labels=z['labels'];smiles=z['smiles']
            assert len(labels)==n and len(set(smiles))==unique
            digest=hashlib.sha256();digest.update(str(tokens.dtype).encode());digest.update(np.asarray(tokens.shape,dtype=np.int64).tobytes());digest.update(tokens.tobytes())
            assert digest.hexdigest()==str(z['tokens_sha256'].item())
            assert np.isfinite(tokens).all()
        for seed in SEEDS:
            split=json.loads((base/'splits'/name/f'seed{seed:03d}.json').read_text())
            parts=[np.asarray(split[k],int) for k in ('train_indices','val_indices','test_indices')]
            assert sorted(np.concatenate(parts).tolist())==list(range(n))
            sets=[set(smiles[p]) for p in parts]
            assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
        reports.append(dict(dataset=name,records=n,canonical_structures=unique,splits=6,input_hash_verified=True,structure_overlap=False))
    return reports


def summary_tables():
    raw=pd.read_csv(DATA/'main/graph_stream_q12_metrics_all_seeds.csv')
    assert len(raw)==360
    keys=['dataset','t','system']
    calc=raw.groupby(keys).test_auc.agg(['mean','std','size']).reset_index()
    ref=pd.read_csv(DATA/'main/graph_stream_q12_curves.csv')
    merged=calc.merge(ref,on=keys,validate='one_to_one');assert len(merged)==len(ref)
    equal(merged['mean'],merged.test_auc_mean);equal(merged['std'],merged.test_auc_sd);assert merged['size'].eq(6).all()
    wide=pd.read_csv(DATA/'main/wide_esn_all_seed_metrics.csv');assert len(wide)==96
    calc=wide.groupby(['dataset','t','width']).test_auc.agg(['mean','std','size']).reset_index()
    ref=pd.read_csv(DATA/'main/wide_esn_summary.csv');merged=calc.merge(ref,on=['dataset','t','width'],validate='one_to_one')
    assert len(merged)==16;equal(merged['mean'],merged.test_auc_mean);equal(merged['std'],merged.test_auc_sd)
    hopping=pd.read_csv(DATA/'main/hopping_six_seed_metrics_all_fixed_axes.csv');assert len(hopping)==2688
    q=hopping.query("system=='qrc' and decoder=='rbf_svr' and readout=='final'")
    selected=q.groupby(['target','p']).validation_mae_ev.mean().reset_index().sort_values(['target','validation_mae_ev','p']).groupby('target').first()
    ref=pd.read_csv(DATA/'main/hopping_validation_selected_rbf_final.csv').query("system=='dissipative_dtc_qrc'")
    for row in ref.itertuples():
        equal(selected.loc[row.target,'p'],row.validation_selected_p)
        vals=q[q.target.eq(row.target)&np.isclose(q.p,row.validation_selected_p)].mae_ev
        assert len(vals)==6;equal(vals.mean(),row.test_mae_mean_ev);equal(vals.std(),row.test_mae_sd_ev)
    deph=pd.read_csv(DATA/'main/fig5_dephasing_k8_k32/plotted_data.csv');assert len(deph)==24
    assert deph[deph.k.eq(8)].band_type.eq('95% bootstrap CI').all()
    assert deph[deph.k.eq(32)].band_type.eq('sample SD').all()
    kdir=CODE/'BFS-QRC/reviewer_supplement_20260905/figures'
    for task,file,metric in [('BACE','09a_k32_bace_dephasing_source.csv','test_auc'),('SHNITSEL','09b_k32_hopping_dephasing_source.csv','mae_ratio_to_coherent')]:
        raw=pd.read_csv(kdir/file)
        for (level,p),group in raw.groupby(['dephasing','p']):
            values=group.groupby('seed')[metric].mean();assert list(values.index)==SEEDS
            row=deph[deph.task.eq(task)&deph.k.eq(32)&np.isclose(deph.dephasing,level)&np.isclose(deph.p,p)].iloc[0]
            equal([values.mean(),values.std()],[row['mean'],row.sd])
    return dict(graph_rows=360,esn_width_rows=96,hopping_rows=2688,dephasing_points=24,
                selected_p={k:float(v) for k,v in selected.p.items()},k8_CI_k32_SD_preserved=True)


def hopping_predictions():
    folder=CODE/'Transition/shnitsel_a01_surface_hopping'
    with np.load(folder/'data/processed/a01_windows.npz',allow_pickle=False) as z:
        truth=z['targets'][z['split']==2].astype(float);ids=z['trajectory_id'][z['split']==2];end=z['endpoint'][z['split']==2];names=list(z['target_names'].astype(str))
    ref=pd.read_csv(DATA/'main/hopping_six_seed_metrics_all_fixed_axes.csv')
    count=0;errors=[]
    for file in (folder/'outputs/strict_decoder_only/shards').glob('*.npz'):
        with np.load(file,allow_pickle=False) as z:
            assert np.array_equal(z['trajectory_id'],ids) and np.array_equal(z['endpoint'],end)
            for i,(decoder,target) in enumerate(zip(z['decoder'].astype(str),z['target_name'].astype(str))):
                match=ref[ref.system.eq(str(z['system']))&ref.seed.eq(int(z['seed']))&np.isclose(ref.p,float(z['p']))&ref.readout.eq(str(z['readout']))&ref.decoder.eq(decoder)&ref.target.eq(target)]
                if len(match)==0: continue  # Retained historical shards outside the frozen 27-p grid.
                assert len(match)==1
                mae=float(np.abs(z['predictions'][i].astype(float)-truth[:,names.index(target)]).mean())
                errors.append(abs(mae-match.iloc[0].mae_ev));count+=1
    assert count==2688 and max(errors)<1e-5
    return dict(prediction_rows=count,maximum_float32_prediction_MAE_difference=max(errors))


def gscan():
    raw=pd.read_csv(DATA/'gscan/metrics_all_seeds.csv');summary=pd.read_csv(DATA/'gscan/summary.csv')
    keys=['g','p','target','readout','decoder']
    assert len(raw)==28800 and len(summary)==4800 and not raw.duplicated(keys+['seed']).any()
    calc=raw.groupby(keys).mae_ev.agg(['mean','std','size']).reset_index()
    merged=calc.merge(summary,on=keys,validate='one_to_one');assert len(merged)==4800 and merged['size'].eq(6).all()
    equal(merged['mean'],merged.mae_mean);equal(merged['std'],merged.mae_sd)
    s=summary.copy();groups=['p','target','readout','decoder']
    s['excess']=100*(s.mae_mean/s.groupby(groups).mae_mean.transform('min')-1)
    s['rank']=s.groupby(groups).mae_mean.rank(method='average')
    main=s[s.p.gt(0)&s.decoder.eq('rbf_svr')&s.readout.eq('final')]
    excess=main.groupby('g').excess.mean();rank=main.groupby('g')['rank'].mean()
    equal(excess.idxmin(),.88);equal(rank.idxmin(),.86)
    near=int((main[np.isclose(main.g,.84)].excess<=5+1e-12).sum());assert near==17
    count=0;errors=[];seen=set()
    base=CODE/'BFS-QRC/hopping_gscan_six_fixed_p_20260908/cases'
    for predfile in sorted(base.glob('g*/seed*/predictions.npz')):
        rows=pd.read_csv(predfile.with_name('metrics.csv'))
        with np.load(predfile,allow_pickle=False) as cached:
            z={k:cached[k] for k in cached.files}
            assert len(rows)==96 and z['predictions'].shape==(96,480)
            names=['gap_1fs','gap_5fs','gap_10fs','min_gap_next_5fs']
            for i,row in rows.iterrows():
                assert str(z['target'][i])==row.target and str(z['readout'][i])==row.readout and str(z['decoder'][i])==row.decoder
                equal(z['p'][i],row.p)
                mae=np.abs(z['predictions'][i].astype(float)-z['target_values'][:,names.index(row.target)].astype(float)).mean()
                errors.append(abs(float(mae)-row.mae_ev));count+=1
                key=(row.g,row.seed,row.p,row.target,row.readout,row.decoder);assert key not in seen;seen.add(key)
        match=raw[raw.g.eq(rows.g.iloc[0])&raw.seed.eq(rows.seed.iloc[0])]
        joint=rows.merge(match,on=keys+['seed'],suffixes=('_case','_all'),validate='one_to_one')
        assert len(joint)==96;equal(joint.mae_ev_case,joint.mae_ev_all)
    assert count==28800 and max(errors)<1e-5
    return dict(rows=count,six_seed_cells=4800,main_best_excess_g=float(excess.idxmin()),
                main_best_rank_g=float(rank.idxmin()),g084_within5=near,main_configurations=20,
                maximum_float32_prediction_MAE_difference=max(errors))


def hardware():
    folder=CODE/'BFS-QRC/Quafu_mainline_Q6T6L3_experiment'
    sys.path.insert(0,str(folder/'local'))
    from run_local_analysis import counts_to_features,extract_counts
    gdir=CODE/'BFS-QRC/Shenglian_Q6T6L3_g_scan_cloud'
    jobs=[]
    for z in sorted((folder/'downloaded_five_tasks').glob('*_cloud_results.zip')):
        key=z.name.removesuffix('_cloud_results.zip')
        jobs.append((z,folder/'local_results_five_tasks'/('hardware_features_'+key+'.npz')))
    for z in sorted(gdir.glob('sl_g*_results.zip')):
        key=z.name.removesuffix('_results.zip')
        jobs.append((z,gdir/'repeat_observable_training_results/features'/('hardware_'+key+'_z_zz.npz')))
    count=0;errors=[]
    for archive,ref in jobs:
        with np.load(ref,allow_pickle=False) as data:
            lookup={int(n):i for i,n in enumerate(data['indices'])};expected=data['features'];hashes=data['token_hashes'] if 'token_hashes' in data else None
        with zipfile.ZipFile(archive) as z:
            manifests=[n for n in z.namelist() if n.endswith(('_job_manifest.jsonl','_jobs.jsonl'))];assert len(manifests)==1
            records=[json.loads(line) for line in z.read(manifests[0]).decode().splitlines() if line.strip()]
            assert len(records)==400
            seen=set()
            for record in records:
                rawname=record['raw_file_name'];names=[n for n in z.namelist() if n.endswith('/'+rawname)];assert len(names)==1
                obj=json.loads(z.read(names[0]));counts=extract_counts(obj);assert counts
                assert sum(counts.values())==1024
                i=lookup[int(record['original_index'])];assert i not in seen;seen.add(i)
                if hashes is not None:assert str(record['token_hash'])==str(hashes[i])
                error=float(np.abs(counts_to_features(counts,record)-expected[i]).max());errors.append(error);count+=1
    assert len(jobs)==19 and count==7600 and max(errors)<1e-7
    return dict(archives=19,molecular_count_records=7600,shots_per_record=1024,maximum_reconstructed_feature_error=max(errors),new_cloud_calls=0)


def main():
    results={}
    for name,fn in [('graph_inputs',graph_inputs),('summary_tables',summary_tables),('hopping_predictions',hopping_predictions),('gscan',gscan),('hardware',hardware)]:
        results[name]=fn();print(name,'passed',flush=True)
    results['passed']=True
    (ROOT/'audit/evidence_verification.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
