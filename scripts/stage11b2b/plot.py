"""Actual epoch nodes and fixed prior-stage case identities; Python-only figures."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import pandas as pd,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scripts.stage11b1c.data import input_records,truths

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'axes.grid':True,'grid.alpha':.15})
COLORS=['#396B9B','#C87542','#57856A','#9972A4','#7E7E7E']
FIG.mkdir(parents=True,exist_ok=True)
records=[]

def save(fig,name,sources,conclusion):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    p=FIG/(name+'.svg');p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig);records.append(dict(figure=name,source_data=sources,conclusion=conclusion,PNG_SHA=sha(FIG/(name+'.png')),SVG_SHA=sha(p)))

def lines(ax,df,y,label=None,color=None):
    ax.plot(df.epoch,df[y],marker='o',lw=1.5,ms=4,label=label or y,color=color)
    ax.set_xlabel('Completed epoch')

def main():
    assert not torch.cuda.is_initialized()
    m=pd.read_csv(OUT/'validation_metrics.csv');v=m[m.epoch>=30];d=json.loads((OUT/'final_decision.json').read_text());last=d['final_complete_validation_epoch']
    old=pd.read_csv(OLD/'epochs.csv');new=pd.read_csv(OUT/'epochs.csv');tr=pd.concat([old,new],ignore_index=True);tr=tr[tr.epoch>=20]
    fig,ax=plt.subplots(figsize=(7.3,3.6));lines(ax,tr,'train_loss','Training BCE sum',COLORS[0]);ax.set_ylabel('Goal BCE + Path BCE (scaled)');a=ax.twinx();lines(a,m[m.epoch>=20],'minADE20','Validation minADE20',COLORS[1]);a.set_ylabel('Validation minADE20 (px)');ax.set_title('Observed training extension; validation only at recorded nodes');ax.legend(loc='upper left');a.legend(loc='upper right');save(fig,'01_training_validation',['epochs.csv','validation_metrics.csv','../stage11b1c/epochs.csv'],'Recorded loss and actual validation nodes; no interpolated missing evaluations')
    fig,ax=plt.subplots(figsize=(6.3,3.6))
    for t,c in zip([10,25,50],COLORS):lines(ax,v,f'A{t}_miss_fraction',f'Goal miss >{t}px',c)
    ax.set_ylabel('Missed-target fraction');ax.legend();ax.set_title('Dynamic goal coverage failures (all 998 targets)');save(fig,'02_goal_miss_fractions',['validation_metrics.csv'],'Strict > threshold failures, dynamic identities')
    fig,ax=plt.subplots(figsize=(6.3,3.6));lines(ax,v,'A25_error_share','A25 error share',COLORS[0]);ax.axhline(.5,color='gray',ls='--',label='Frozen H1 share threshold');ax.set_ylabel('Share of summed minADE20 error');ax.set_ylim(0,1);ax.legend();save(fig,'03_A25_error_contribution',['validation_metrics.csv'],'Error contribution differs from cohort size')
    fixed=pd.read_csv(OUT/'a25_fixed_cohort.csv');fs=fixed.groupby('epoch')[['ADE1','minADE20','minFDE20','minGoalFDE20']].mean().reset_index();fs.to_csv(OUT/'figure04_fixed_cohort_means.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.3,3.6))
    for k,c in zip(['ADE1','minADE20','minGoalFDE20'],COLORS):lines(ax,fs,k,k,c)
    ax.set_ylabel('Mean error (px)');ax.legend();ax.set_title('Fixed epoch30 A25 cohort (same 164 identities)');save(fig,'04_fixed164_errors',['a25_fixed_cohort.csv','figure04_fixed_cohort_means.csv'],'Fixed original failed identities, not the current dynamic cohort')
    t=pd.read_csv(OUT/'a25_transition.csv');t=t[t.comparison=='vs_epoch30']
    fig,ax=plt.subplots(figsize=(6.3,3.6))
    for k,c in zip(['recovered','new_failures','persistent_failure'],COLORS):lines(ax,t,k,k.replace('_',' '),c)
    ax.set_ylabel('Target windows');ax.legend();ax.set_title('A25 identity transitions relative to epoch30');save(fig,'05_cohort_transitions',['a25_transition.csv'],'Recovery and new failures both retained')
    fig,ax=plt.subplots(figsize=(6.3,3.6));lines(ax,v,'ADE1','Default head ADE1',COLORS[0]);lines(ax,v,'minADE20','GT oracle minADE20',COLORS[1]);ax.set_ylabel('Position error (px)');ax.legend();save(fig,'06_default_oracle_gap',['validation_metrics.csv'],'K1 and oracle K20 are distinct evaluation budgets')
    s=pd.read_csv(OUT/'scene_metrics.csv');s=s[s.epoch>=30];fig,ax=plt.subplots(figsize=(7.2,4))
    for (name,x),c in zip(s.groupby('scene'),COLORS):lines(ax,x,'A25_miss_fraction',name,c)
    ax.set_ylabel('A25 missed-target fraction');ax.legend(ncol=2);ax.set_title('All five development recordings');save(fig,'07_recording_goal_coverage',['scene_metrics.csv'],'Unequal scene counts; each curve is its own recording')
    g=pd.read_csv(OUT/'motion_group_metrics.csv');fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
    for ax,(factor,x) in zip(axes.flat,g.groupby('factor',sort=True)):
        for (group,y),c in zip(x.groupby('group'),COLORS):lines(ax,y,'A25_miss_fraction',group,c)
        ax.set_title(factor.replace('hist_','').replace('_',' '));ax.set_ylabel('A25 fraction');ax.legend(fontsize=7)
    save(fig,'08_frozen_motion_strata',['motion_group_metrics.csv'],'Prior-stage historical groups; descriptive association, overlapping windows')
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    for k,c in zip(['GoalFDE1','FDE1','minGoalFDE20','minFDE20'],COLORS):lines(ax,v,k,k,c)
    ax.set_ylabel('Endpoint error (px)');ax.legend(ncol=2);save(fig,'09_goal_path_endpoint',['validation_metrics.csv'],'Each minimum is independent; same-head pairs retained separately')
    if last:
        cases=pd.read_csv(ROOT/'results/tables/stage11b2a/failure_case_manifest.csv');assert len(cases)==21
        packs={30:torch.load(ROOT/'experiments/stage11b1c/cache/predictions/epoch030.pt',map_location='cpu',weights_only=True),last:torch.load(CACHE/f'predictions/epoch{last:03d}.pt',map_location='cpu',weights_only=True)}
        ids=packs[30]['sample_ids'];hist={r['sample_id']:r for r in input_records('validation')};truth=truths('validation');coords=[]
        for ep in [30,last]:
            assert packs[ep]['sample_ids']==ids
            for row in cases.to_dict('records'):
                sid=row['sample_id'];i=ids.index(sid);p=packs[ep]['paths'][:,i].numpy();goal=packs[ep]['goals'][:,i].numpy();y=truth[sid];h=hist[sid]['history']
                for k in range(20):
                    for t in range(12):coords.append(dict(case=row['case'],sample_id=sid,epoch=ep,head=k,future_step=t+1,predicted_x=float(p[k,t,0]),predicted_y=float(p[k,t,1]),goal_x=float(goal[k,0]),goal_y=float(goal[k,1]),GT_x=float(y[t,0]),GT_y=float(y[t,1])))
        pd.DataFrame(coords).to_csv(OUT/'fixed_case_prediction_coordinates.csv.gz',index=False)
        pd.DataFrame([dict(sample_id=sid,history_step=t+1,x=float(x),y=float(y)) for sid in cases.sample_id.unique() for t,(x,y) in enumerate(hist[sid]['history'])]).to_csv(OUT/'fixed_case_history_coordinates.csv',index=False)
        imagecache={}
        def draw(ax,row,ep,compact=False):
            sid=row['sample_id'];i=ids.index(sid);scene=row['scene']
            if scene not in imagecache:imagecache[scene]=mpimg.imread(RES/'data/SDD/train'/scene/'reference.jpg')
            im=imagecache[scene];ax.imshow(im);p=packs[ep]['paths'][:,i].numpy();goals=packs[ep]['goals'][:,i].numpy();h=hist[sid]['history'];y=truth[sid];err=np.linalg.norm(p-y[None],axis=-1).mean(-1);best=int(err.argmin())
            for k in range(20):ax.plot(np.r_[h[-1,0],p[k,:,0]],np.r_[h[-1,1],p[k,:,1]],color='#7777AD',alpha=.45,lw=.8)
            ax.scatter(goals[:,0],goals[:,1],s=11,c='#7777AD',marker='x',alpha=.8)
            ax.plot(h[:,0],h[:,1],'-o',color='white',lw=1.8,ms=2,label='History')
            ax.plot(np.r_[h[-1,0],y[:,0]],np.r_[h[-1,1],y[:,1]],'-',color='#00C88C',lw=2,label='GT (offline)')
            for k,c,label in [(0,'#FFB347','Default head'),(best,'#FF4B63','ADE oracle head')]:ax.plot(np.r_[h[-1,0],p[k,:,0]],np.r_[h[-1,1],p[k,:,1]],color=c,lw=1.6,label=label)
            # Common extent derived from both actual epoch predictions, full history/GT, and all goals. No candidate is cropped away.
            a=np.concatenate([h,y]+[packs[e]['paths'][:,i].numpy().reshape(-1,2) for e in [30,last]]+[packs[e]['goals'][:,i].numpy().reshape(-1,2) for e in [30,last]])
            lo=a.min(0)-25;hi=a.max(0)+25;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.grid(False)
            ax.set_title(f"{row['case']} | epoch {ep}\n{sid}\nminADE20 {err.min():.2f}px",fontsize=7 if compact else 9)
            if not compact:ax.legend(fontsize=7,loc='best')
        for row in cases.to_dict('records'):
            fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
            for ax,ep in zip(axes,[30,last]):draw(ax,row,ep)
            save(fig,'10_case_'+row['case'],['fixed_case_prediction_coordinates.csv.gz','fixed_case_history_coordinates.csv','../stage11b2a/failure_case_manifest.csv'],f'Frozen case epoch30 vs actual epoch{last}; all20 paths/goals retained, oracle offline only')
        fig,axes=plt.subplots(4,2,figsize=(10,13),constrained_layout=True)
        for rr,row in enumerate(cases.iloc[:4].to_dict('records')):
            for cc,ep in enumerate([30,last]):draw(axes[rr,cc],row,ep,True)
        save(fig,'10_fixed_case_overview',['fixed_case_prediction_coordinates.csv.gz','fixed_case_history_coordinates.csv','../stage11b2a/failure_case_manifest.csv'],'First four frozen manifest rows, selected before extension; all21 pairs also supplied')
    jnew(OUT/'figure_manifest.json',records)
    print('Figures',len(records),'CPU only',not torch.cuda.is_initialized(),flush=True)
if __name__=='__main__':main()
