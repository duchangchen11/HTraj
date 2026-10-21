"""Python-only real generation-chain plots; case ranks frozen before inference."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
from .math_audit import probability
import numpy as np,pandas as pd,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import matplotlib.image as mpimg
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
COLORS=['#426E9F','#CB814A','#4D8D78','#9F739D','#747474']
RECORD=[]
def save(fig,name,sources,claim):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    p=FIG/(name+'.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n');plt.close(fig)
    RECORD.append(dict(figure=name,source_data=sources,claim=claim,PNG_SHA=sha(FIG/(name+'.png')),SVG_SHA=sha(p)))
def main():
    assert not torch.cuda.is_initialized();assert json.loads((OUT/'cache_integrity.json').read_text())['passed']
    m=pd.read_csv(OUT/'tensor_manifest.csv').set_index('sample_id');prob=pd.read_csv(OUT/'probability_mass.csv');e=pd.read_csv(OUT/'coverage_transition.csv');ss=pd.read_csv(OUT/'scene_mechanism_summary.csv');soft=pd.read_csv(OUT/'softargmax_kmeans_comparison.csv');groups=pd.read_csv(OUT/'motion_mechanism_summary.csv');cases=pd.read_csv(OUT/'mechanism_case_manifest.csv')
    histories=torch.load(CACHE/'validation_history_only.pt',map_location='cpu',weights_only=True);hs={s:histories['history'][i].numpy() for i,s in enumerate(histories['sample_ids'])};FIG.mkdir(parents=True,exist_ok=True)
    def load_case(row):
        p=ROOT/m.loc[row['sample_id'],'path'];assert sha(p)==m.loc[row['sample_id'],'sha256'];return torch.load(p,map_location='cpu',weights_only=True)
    def draw_map(ax,row,pack,mode):
        x=pack['tensors'];H,W=x['sigmoid'].shape[-2:];extent=(-2,(W-.5)/.25,(H-.5)/.25,-2)
        if mode=='sigmoid':a=x['sigmoid'][0,0].numpy();im=ax.imshow(a,origin='upper',extent=extent,interpolation='nearest',cmap='magma');label='Raw Sigmoid weight (unnormalized)'
        elif mode=='probability':
            a=probability(x['normalized_weights'].reshape(H,W).numpy());mask=np.ma.masked_less_equal(a,0);im=ax.imshow(mask,origin='upper',extent=extent,interpolation='nearest',cmap='magma',norm=LogNorm(vmin=a[a>0].min(),vmax=a.max()));label='Discrete sampling probability (log scale)'
        else:
            a=np.bincount(x['indices'].numpy().ravel(),minlength=H*W).reshape(H,W);assert a.sum()==10000
            im=ax.imshow(np.ma.masked_equal(a,0),origin='upper',extent=extent,interpolation='nearest',cmap='viridis',norm=LogNorm(vmin=1,vmax=max(2,a.max())));label='Observed draw count (10000 total)'
        h=hs[row['sample_id']];ax.plot(h[:,0],h[:,1],'-o',c='#76B8E8',ms=2,lw=1,label='History8');ax.scatter([row['GT_x']],[row['GT_y']],marker='x',s=45,c='#EC504D',label='GT endpoint (offline)');ax.add_patch(Circle((row['GT_x'],row['GT_y']),25,fill=False,color='#EC504D',lw=1))
        ax.set_title(label,fontsize=8);ax.set_xlabel('Reference x (px)');ax.set_ylabel('Reference y (px)');plt.colorbar(im,ax=ax,fraction=.035,pad=.02)
    def draw_goals(ax,row,pack,paths=False):
        im=mpimg.imread(RES/'data/SDD/train'/row['scene']/'reference.jpg');ax.imshow(im);x=pack['tensors'];g=x['goals'][:,0].numpy();h=hs[row['sample_id']]
        if paths:
            p=x['paths'][:,0].numpy()
            for k in range(20):ax.plot(np.r_[h[-1,0],p[k,:,0]],np.r_[h[-1,1],p[k,:,1]],c='#8C79A8',alpha=.35,lw=.7)
        ax.scatter(g[1:,0],g[1:,1],s=35,marker='o',facecolors='none',edgecolors=COLORS[0],label='19 KMeans centers');ax.scatter(g[0,0],g[0,1],s=100,marker='*',c=COLORS[1],label='1 Softargmax');ax.scatter([row['GT_x']],[row['GT_y']],marker='x',s=45,c='#EC504D',label='GT endpoint (offline)');ax.add_patch(Circle((row['GT_x'],row['GT_y']),25,fill=False,color='#EC504D',lw=1));ax.plot(h[:,0],h[:,1],'-o',c='#59CBAA',ms=2,lw=1,label='History8')
        ax.set_title('Frozen goals'+(' + all20 decoded paths' if paths else ''),fontsize=8);ax.set_xlabel('Reference x (px)');ax.set_ylabel('Reference y (px)');ax.legend(fontsize=6,loc='best')
    if len(cases):
        first=cases.iloc[0].to_dict();pack=load_case(first)
        fig,ax=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
        for a,mode in zip(ax,['sigmoid','probability']):draw_map(a,first,pack,mode)
        fig.suptitle(first['sample_id']+' | first case under frozen rank rule',fontsize=9);save(fig,'01_goal_weights_and_probability',['mechanism_case_manifest.csv','tensor_manifest.csv'],'Unnormalized weights and normalized sampling distribution are different quantities')
        fig,ax=plt.subplots(figsize=(6.5,5));draw_map(ax,first,pack,'density');fig.suptitle(first['sample_id'],fontsize=9);save(fig,'02_actual_10000_draw_density',['mechanism_case_manifest.csv','tensor_manifest.csv'],'Actual draw histogram, not10000 distinct positions')
        fig,ax=plt.subplots(figsize=(6.5,5));draw_goals(ax,first,pack);fig.suptitle(first['sample_id'],fontsize=9);save(fig,'03_centers_softargmax_GT',['mechanism_case_manifest.csv','tensor_manifest.csv'],'19 centers +1 separate softargmax; GT only an offline annotation')
    fig,ax=plt.subplots(figsize=(6.5,3.8))
    for r,c in zip([10,25,50],COLORS):
        a=prob[f'P_mass{r}'].to_numpy();positive=np.sort(a[a>0]);zeros=int((a==0).sum());ax.plot(positive,(np.arange(1,len(positive)+1)+zeros)/len(a),label=f'r={r}px; zero mass {zeros}',c=c)
    ax.axvline(-np.expm1(np.log(.5)/10000),ls='--',c='gray',label='Theoretical P(hit)=0.5');ax.set_xscale('log');ax.set_xlabel('GT-neighborhood probability mass (positive values)');ax.set_ylabel('Empirical CDF, including zero-mass offset');ax.legend(fontsize=7);save(fig,'04_GT_neighborhood_mass_distribution',['probability_mass.csv'],'Offline probability mass, not calibrated uncertainty')
    allr=ss[ss.scope=='ALL_998'].sort_values('radius');xx=np.arange(3);fig,ax=plt.subplots(figsize=(6.5,3.8));ax.bar(xx-.18,allr.proposal_cover_fraction,.36,label='10000 raw draws',color=COLORS[0]);ax.bar(xx+.18,allr.final_cover_fraction,.36,label='Final K20 goals',color=COLORS[1]);ax.set_xticks(xx,[str(r)+'px' for r in allr.radius]);ax.set_ylim(0,1);ax.set_ylabel('Covered-target fraction (same998)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2);save(fig,'05_proposal_vs_final_coverage',['scene_mechanism_summary.csv'],'Within-pipeline coverage comparison; not equal-budget model performance')
    q=e[(e.radius==25)&(~e.C)];counts=q.category.value_counts().reindex(['zero_support_final_miss','positive_support_sampling_miss_final_miss','proposal_hit_final_miss'],fill_value=0);fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Zero support','Positive support,\nno draw hit','Draw hit,\nfinal K20 miss'],counts,color=COLORS[:3]);ax.set_ylabel('Final A25 failed target windows');ax.set_title('Observed stage events among original164 misses');save(fig,'06_final164_stage_events',['coverage_transition.csv','mechanism_type_summary.csv'],'Observed coverage loss co-occurrence, not causal attribution of trajectory error')
    fig,ax=plt.subplots(figsize=(7,4));w=.24
    for j,(source,col,c) in enumerate([('Softargmax1','SoftCover',COLORS[0]),('KMeans19','KMeansCover',COLORS[1]),('Combined20','CombinedCover',COLORS[2])]):ax.bar(xx+(j-1)*w,[soft[col+str(r)].mean() for r in [10,25,50]],w,label=source,color=c)
    ax.set_xticks(xx,['10px','25px','50px']);ax.set_ylim(0,1);ax.set_ylabel('Covered-target fraction');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=3);save(fig,'07_softargmax_KMeans_contributions',['softargmax_kmeans_comparison.csv'],'Descriptive subsets of same frozen collection, not retrained fair K comparisons')
    scenes=ss[(ss.radius==25)&ss.scope.isin(['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'])].sort_values('scope');fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);ix=np.arange(5)
    for off,k,label,c in [(-.18,'proposal_cover_fraction','10000 draws',COLORS[0]),(.18,'final_cover_fraction','Final K20',COLORS[1])]:axes[0].bar(ix+off,scenes[k],.36,label=label,color=c)
    axes[0].set_ylabel('25px coverage fraction');axes[0].set_ylim(0,1);axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2);axes[1].bar(ix,scenes.compression_loss_fraction,color=COLORS[2]);axes[1].set_ylabel('Draw-hit / final-miss fraction')
    for ax in axes:ax.set_xticks(ix,scenes.scope,rotation=25,ha='right')
    save(fig,'08_five_recording_coverage',['scene_mechanism_summary.csv'],'Every recording retained; counts and equal-weight macro separately available')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(factor,g) in zip(axes.flat,groups[groups.radius==25].groupby('factor',sort=True)):
        g=g.sort_values('group');ix=np.arange(len(g));ax.bar(ix-.18,g.proposal_cover_fraction,.36,label='10000 draws',color=COLORS[0]);ax.bar(ix+.18,g.final_cover_fraction,.36,label='Final K20',color=COLORS[1]);ax.set_xticks(ix,[f'{r.group}\n(n={r.targets})' for r in g.itertuples()],rotation=18,ha='right',fontsize=7);ax.set_ylim(0,1);ax.set_title(factor);ax.legend(fontsize=7)
    save(fig,'09_frozen_history_groups',['motion_mechanism_summary.csv'],'Historical groups frozen before this audit; descriptive differences only')
    for row in cases.to_dict('records'):
        pack=load_case(row);fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True);draw_map(axes[0],row,pack,'probability');draw_map(axes[1],row,pack,'density');draw_goals(axes[2],row,pack,True)
        fig.suptitle(f"{row['category']} | {row['sample_id']}\nEpoch30 {CK[:12]} | proposal min {row['ProposalMinDist']:.2f}px / final min {row['FinalMinDist20']:.2f}px | minADE20 {row['minADE20']:.2f}px",fontsize=9)
        meta=dict(case=row,tensor_path=m.loc[row['sample_id'],'path'],tensor_SHA=m.loc[row['sample_id'],'sha256'],all_SHA_fields_in_case=True,no_GT_used_for_generation=True)
        metap=OUT/('case_'+row['case']+'.json')
        if metap.exists():assert json.loads(metap.read_text())==meta
        else:jnew(metap,meta)
        save(fig,'10_case_'+row['case'],['mechanism_case_manifest.csv','case_'+row['case']+'.json','tensor_manifest.csv'],'Frozen rank-selected real mechanism case; all20 paths retained')
    replace_json(OUT/'figure_manifest.json',RECORD);print('Figures',len(RECORD),'all from actual tensors; CPU only',flush=True)
if __name__=='__main__':main()
