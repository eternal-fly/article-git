from pathlib import Path
import pandas as pd
import numpy as np
W=Path(__file__).resolve().parent
SRC=W.parent/'source_data/gscan'
if not (SRC/'summary.csv').exists():SRC=W.parent.parents[1]/'hopping_gscan_six_fixed_p_20260908'
s=pd.read_csv(SRC/'summary.csv')
keys=['decoder','readout','target','p']
s['min_mae']=s.groupby(keys).mae_mean.transform('min')
s['excess_pct']=100*(s.mae_mean/s.min_mae-1)
s['rank']=s.groupby(keys).mae_mean.rank(method='average')
subsets=[('All',s),('All, nonzero reset',s[s.p.gt(0)])]
for d in ['rbf_svr','linear']:
 for r in ['final','tm3']:
  a=s[s.decoder.eq(d)&s.readout.eq(r)]
  subsets.extend([(d+'/'+r+', all reset',a),(d+'/'+r+', nonzero reset',a[a.p.gt(0)])])
rows=[];curves=[]
for name,a in subsets:
 g=a.groupby('g').agg(n=('mae_mean','size'),mean_excess_pct=('excess_pct','mean'),mean_rank=('rank','mean'),within5=('excess_pct',lambda z:int((z<=5+1e-12).sum())))
 e=g.mean_excess_pct.idxmin();r=g.mean_rank.idxmin()
 rows.append(dict(subset=name,n=int(g.loc[e,'n']),g_excess=e,min_excess_pct=g.loc[e,'mean_excess_pct'],g_rank=r,min_rank=g.loc[r,'mean_rank'],at084_within5=int(g.loc[.84,'within5']),at084_excess=g.loc[.84,'mean_excess_pct']))
 curves.append(g.reset_index().assign(subset=name))
pd.DataFrame(rows).to_csv(W/'aggregate_summary.csv',index=False)
pd.concat(curves).to_csv(W/'aggregate_curves.csv',index=False)
s.to_csv(W/'normalized_configurations.csv',index=False)
main=s[s.decoder.eq('rbf_svr')&s.readout.eq('final')&s.p.gt(0)]
assert (main[main.g.eq(.84)].excess_pct<=5).sum()==17
assert (main[main.g.eq(.88)].excess_pct<=5).sum()==19
print(pd.DataFrame(rows).to_string(index=False))
