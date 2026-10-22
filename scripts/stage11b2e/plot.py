"""Six real paired control figures; no new candidates, inference or training."""
from .common import *
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    cpu_only();assert json.loads((OUT/'score_integrity.json').read_text())['passed']
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    scene=pd.read_csv(OUT/'scene_comparison.csv').set_index('scope');rf=pd.read_csv(OUT/'failure_recovery.csv')
    decision=json.loads((OUT/'final_decision.json').read_text());colors=['#B46B57','#517F98'];figs=[]
    def save(fig,name,data,claim):
        fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight');p=FIG/(name+'.svg')
        fig.savefig(p,bbox_inches='tight');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
        figs.append(dict(name=name,PNG_SHA=sha(FIG/(name+'.png')),SVG_SHA=sha(p),source_data=data,claim=claim,
            unit='original SDD reference pixels; coverage fraction or counts labeled',statistics='paired descriptive means; overlapping windows; no independent-sample CI'))
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    for ax,scope in zip(axs,['ALL998','FIVE_RECORDING_MACRO']):
        x=np.arange(3);row=scene.loc[scope]
        for j,label in enumerate(['Original','Alternative']):
            y=np.array([row[label+'GoalCover'+str(r)] for r in [10,25,50]])*100
            bars=ax.bar(x+(j-.5)*.34,y,width=.34,color=colors[j],label=label)
            ax.bar_label(bars,fmt='%.2f',fontsize=6,padding=2)
        ax.set_xticks(x,['10 px','25 px','50 px']);ax.set_ylim(0,108);ax.set_ylabel('Goal coverage (%)');ax.set_title(scope);ax.legend(fontsize=7)
    save(fig,'01_goal_coverage','scene_comparison.csv','Fixed same K20 and target set; show all three frozen radii and both aggregations')
    fig,axs=plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    for ax,key in zip(axs,['minADE20','minFDE20']):
        x=np.arange(2)
        for j,label in enumerate(['Original','Alternative']):
            y=[scene.loc[s,label+key] for s in ['ALL998','FIVE_RECORDING_MACRO']]
            bars=ax.bar(x+(j-.5)*.34,y,width=.34,color=colors[j],label=label)
            ax.bar_label(bars,fmt='%.4f',fontsize=6,padding=2)
        ax.set_xticks(x,['Window weighted','Five-recording macro']);ax.set_ylabel(key+' (px)');ax.set_ylim(bottom=0);ax.legend(fontsize=7)
    save(fig,'02_path_metrics','scene_comparison.csv','Actual redecoded full paths; ADE and FDE minima can come from different heads')
    q=rf[rf.original_category25=='proposal_hit_final_miss'];fig,ax=plt.subplots(figsize=(4.7,3.2))
    counts=[int(q.AlternativeGoalCover25.sum()),int((~q.AlternativeGoalCover25).sum())]
    bars=ax.bar(['Recovered at25','Still missed'],counts,color=[colors[1],colors[0]]);ax.bar_label(bars);ax.set_ylim(0,95);ax.set_ylabel('Targets among original fixed86')
    save(fig,'03_original86_recovery','failure_recovery.csv','Recovery is evaluated on the original86 identities, never redefined')
    q=rf[rf.OriginalGoalCover25];fig,ax=plt.subplots(figsize=(4.7,3.2))
    counts=[int(q.AlternativeGoalCover25.sum()),int((~q.AlternativeGoalCover25).sum())]
    bars=ax.bar(['Coverage retained','Newly lost at25'],counts,color=[colors[1],colors[0]]);ax.bar_label(bars);ax.set_ylim(0,910);ax.set_ylabel('Targets among original834 covered')
    save(fig,'04_original_success_regression','failure_recovery.csv','New failures are retained in the comparison, not hidden by recoveries')
    names=['bookstore_2','deathCircle_3','gates_7','hyang_7','nexus_4'];fig,ax=plt.subplots(figsize=(6.0,3.6))
    y=scene.loc[names,'DeltaGoalCover25'].to_numpy()*100
    ax.barh(names,y,color=[colors[1] if v>=0 else colors[0] for v in y]);ax.axvline(0,c='.4',lw=.8)
    bound=max(.2,float(np.max(np.abs(y)))*1.5);ax.set_xlim(-bound,bound)
    for i,(name,v) in enumerate(zip(names,y)):
        ax.text(max(0,v)+bound*.03,i,f'{v:+.3f} pp; n={int(scene.loc[name,"targets"])}',ha='left',va='center',fontsize=7)
    ax.set_xlabel('Alternative - original GoalCover25 (percentage points)')
    save(fig,'05_five_recording_goal_change','scene_comparison.csv','All five reused recordings, including small groups and zero/negative gains')
    # Actual case paths/GT only read after independent CPU scoring completed.
    from scripts.stage11b1c.data import truths
    truth=truths('validation');old=torch.load(PP,map_location='cpu',weights_only=True)
    new=torch.load(CACHE/'alternative_paths.pt',map_location='cpu',weights_only=True)
    hist=json.loads((CONF/'generation_config.json').read_text())['history'];h=torch.load(ROOT/hist['path'],map_location='cpu',weights_only=True)
    oi={s:i for i,s in enumerate(old['sample_ids'])};ni={s:i for i,s in enumerate(new['sample_ids'])};hi={s:i for i,s in enumerate(h['sample_ids'])}
    cases=pd.read_csv(OUT/'case_manifest.csv');fig,axs=plt.subplots(2,2,figsize=(9.5,8),layout='constrained');raw=[]
    for ax,outcome in zip(axs.flat,['recovered','lost','stayed_fail','stayed_cover']):
        case=cases[cases.outcome==outcome]
        if not len(case):ax.text(.5,.5,'No eligible '+outcome+' case\n(no fabricated example)',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
        case=case.iloc[0];sid=case.sample_id;i=oi[sid];j=ni[sid];og=old['goals'][:,i].numpy();ag=new['goals'][:,j].numpy();op=old['paths'][:,i].numpy();ap=new['paths'][:,j].numpy();y=truth[sid];history=h['history'][hi[sid]].numpy()
        for path in op:ax.plot(path[:,0],path[:,1],color=colors[0],alpha=.28,lw=.7)
        for path in ap:ax.plot(path[:,0],path[:,1],color=colors[1],alpha=.28,lw=.7,ls='--')
        ax.plot(history[:,0],history[:,1],color='.4',lw=1.3,label='History')
        ax.plot(y[:,0],y[:,1],color='black',lw=1.3,label='GT (offline)')
        ax.scatter(og[1:,0],og[1:,1],marker='x',c=colors[0],s=25,label='Original centers/paths')
        ax.scatter(ag[1:,0],ag[1:,1],marker='+',c=colors[1],s=25,label='Medoids/new paths')
        ax.scatter(*og[0],c='#7C6293',marker='D',s=16,label='Unchanged Softargmax')
        ax.scatter(*y[-1],marker='*',c='black',s=45);ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis()
        ax.set_title(outcome+' | '+sid.replace('SDD:',''),fontsize=7);ax.set_xlabel('x (px)');ax.set_ylabel('y (px)');ax.legend(fontsize=5)
        raw.append(dict(sample_id=sid,scene=case.scene,outcome=outcome,rank=int(case['rank']),eligible_targets=int(case.eligible_targets),
            original_goals=og.tolist(),alternative_goals=ag.tolist(),original_paths=op.tolist(),alternative_paths=ap.tolist(),
            history=history.tolist(),GT=y.tolist(),checkpoint_SHA=CK,original_cache_SHA=PSHA,
            alternative_goal_SHA=digest(new['goals'][:,j:j+1]),alternative_path_SHA=digest(new['paths'][:,j:j+1]),
            original_goal_SHA=digest(old['goals'][:,i:i+1]),original_path_SHA=digest(old['paths'][:,i:i+1])))
    (OUT/'case_plot_data.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save(fig,'06_real_paired_cases','case_manifest.csv;case_plot_data.json','Fixed median-ID case for each paired outcome; all20 original and newly decoded paths shown; exact overlaps possible')
    (OUT/'figure_manifest.json').write_text(json.dumps(figs,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('FIGURES',len(figs),'PNG/SVG pairs',flush=True)

if __name__=='__main__':main()
