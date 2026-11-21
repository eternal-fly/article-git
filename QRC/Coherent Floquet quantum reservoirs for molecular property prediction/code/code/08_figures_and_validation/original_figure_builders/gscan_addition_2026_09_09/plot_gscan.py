from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import fitz

W=Path(__file__).resolve().parent
R=W.parent
SRC=R.parents[1]/'hopping_gscan_six_fixed_p_20260908'
OUT=W/'figures';OUT.mkdir(exist_ok=True)
S=pd.read_csv(SRC/'summary.csv')
PS=[0,.2,.4,.6,.8,1.]
GS=np.round(np.arange(50)*.02,2)
TARGETS=['gap_1fs','gap_5fs','gap_10fs','min_gap_next_5fs']
NAMES=['Gap, 1 fs','Gap, 5 fs','Gap, 10 fs','Minimum gap, next 5 fs']
COLORS=['#3E4B65','#497EA5','#67A29B','#989D64','#C18B46','#A16D8B']
STYLES=['-','-','--','-.',':','-']
MARKERS=[None,'o',None,None,None,'s']
REGIONS=[('NTC',0,.22,'#F0F4FB','#4F6690'),('Thermal',.22,.84,'#FCF2EE','#916657'),('DTC',.84,.98,'#F0F7EB','#607D4D')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
 'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':True})
records=[]

def draw(kind):
    linear=kind=='linear'
    decoder='linear' if linear else 'rbf_svr'
    ncols=4 if linear else 2
    height=114 if linear else 130
    fig,axes=plt.subplots(2,ncols,figsize=(178/25.4,height/25.4))
    if linear:fig.subplots_adjust(left=.076,right=.987,bottom=.10,top=.82,wspace=.42,hspace=.76)
    else:fig.subplots_adjust(left=.095,right=.984,bottom=.09,top=.82,wspace=.24,hspace=.55)
    sources=[];assertions=[]
    for j,ax in enumerate(axes.flat):
        ti=j%4 if linear else j
        target=TARGETS[ti]
        readout=('final' if j<4 else 'tm3') if linear else kind
        sub=S[S.decoder.eq(decoder)&S.readout.eq(readout)&S.target.eq(target)].copy()
        sources.append(sub.assign(panel='abcdefgh'[j]))
        for pi,p in enumerate(PS):
            c=sub[np.isclose(sub.p,p)].sort_values('g');assert len(c)==50
            x=c.g.to_numpy();y=c.mae_mean.to_numpy();sd=c.mae_sd.to_numpy()
            np.testing.assert_array_equal(x,GS)
            ax.fill_between(x,y-sd,y+sd,color=COLORS[pi],alpha=.075,lw=0,zorder=1)
            line,=ax.plot(x,y,color=COLORS[pi],ls=STYLES[pi],marker=MARKERS[pi],markevery=5,
                ms=1.8 if linear else 2.3,lw=1.15 if pi==0 else .95,
                label=rf'$p={p:.1f}$'+(' (closed)' if pi==0 else ''),zorder=3)
            np.testing.assert_array_equal(line.get_xdata(),x)
            np.testing.assert_array_equal(line.get_ydata(),y)
            assertions.append({'panel':j,'p':p,'readout':readout,'target':target,'n':50,'exact_line_values':True})
        ylim=ax.get_ylim()
        for label,lo,hi,fill,ink in REGIONS:
            ax.axvspan(lo,hi,facecolor=fill,edgecolor='none',zorder=-2)
            ax.add_patch(Rectangle((lo,1.012),hi-lo,.075,transform=ax.get_xaxis_transform(),facecolor=fill,
                                   edgecolor='#9AA0A6',lw=.4,clip_on=False))
            ax.text((lo+hi)/2,1.049,label,transform=ax.get_xaxis_transform(),ha='center',va='center',
                    color=ink,fontsize=6.0 if linear else 7.0,clip_on=False)
        for boundary in [.22,.84]:ax.axvline(boundary,color='#83858A',lw=.65,ls=(0,(3,3)),zorder=2)
        ax.set_ylim(ylim);ax.set_xlim(0,.98)
        ax.set_xlabel(r'Floquet parameter, $g$' if not linear else '$g$',labelpad=2)
        if not linear or ti==0:ax.set_ylabel('Test MAE (eV)',labelpad=2)
        ax.set_xticks([0,.2,.4,.6,.8,.98] if not linear else [0,.5,.98],
                      ['0','0.2','0.4','0.6','0.8','0.98'] if not linear else ['0','0.5','0.98'])
        ax.get_xticklabels()[-1].set_ha('right')
        ax.get_xticklabels()[0].set_ha('left')
        ax.tick_params(labelsize=7 if linear else 7.5,pad=2,length=3,width=.7)
        ax.grid(axis='y',color='#DEE1E5',lw=.5);ax.set_axisbelow(True)
        title=NAMES[ti]
        if linear:title=('Final: ' if readout=='final' else 'TM3: ')+['1 fs','5 fs','10 fs','Next-5-fs min.'][ti]
        ax.set_title(title,y=1.11,pad=2,fontsize=7.1 if linear else 8.5)
        ax.text(-.25 if linear else -.105,1.13,f'({"abcdefgh"[j]})',transform=ax.transAxes,
                fontsize=8 if linear else 9,fontweight='bold',va='bottom',clip_on=False)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3 if not linear else 6,
               bbox_to_anchor=(.54 if not linear else .52,.995),fontsize=7.5 if not linear else 6.7,
               frameon=True,fancybox=False,handlelength=2.4,columnspacing=1.5 if not linear else 1.05,
               borderpad=.4,labelspacing=.4)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    artists=list(fig.legends)
    for ax in axes.flat:
        artists.extend([ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts])
    for artist in artists:
        b=artist.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1,(kind,artist.get_text() if hasattr(artist,'get_text') else 'legend',b)
    name={'final':'fig_gscan_rbf_final','tm3':'fig_gscan_rbf_tm3','linear':'fig_gscan_linear_readouts'}[kind]
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=400)
    pd.concat(sources).to_csv(OUT/f'{name}_source.csv',index=False)
    plt.close(fig)
    pdf=fitz.open(OUT/f'{name}.pdf')
    assert len(pdf)==1
    assert all(f[1] in ['ttf','otf'] for f in pdf[0].get_fonts())
    records.append({'name':name,'panels':len(axes.flat),'curves':len(assertions),'all_50_raw_points':True,
                    'all_artists_inside_canvas':True,'embedded_fonts':True,'curve_checks':assertions})

for kind in ['final','tm3','linear']:draw(kind)
(W/'figure_qa.json').write_text(json.dumps(records,indent=2)+'\n')
print('Three figure exports ready: four RBF/final panels, four RBF/TM3 panels, eight Ridge panels.')
