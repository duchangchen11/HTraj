"""Python-only real-data plots and quantile-selected cases; no model calls."""
from .common import *
from .data import load
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    decision=json.loads((OUT/'final_decision.json').read_text());bestep=decision['best_epoch']
    ep=list(csv.DictReader((OUT/'epochs.csv').open()))
    val=[json.loads(p.read_text()) for p in sorted(OUT.glob('validation_epoch*.json'))]
    val=[r for r in val if r['status']=='COMPLETE']
    caption='Y-Net a345ada | Development only | seed17 | raw-pixel inputs | 8 history / 12 future'
    def curve(name,source,keys,title,ylabel):
        if not source:return
        fig,ax=plt.subplots(figsize=(7,3.5))
        for key,label in keys:ax.plot([int(r['epoch']) for r in source],[float(r[key]) for r in source],'-o',ms=4,label=label)
        if bestep is not None:ax.axvline(bestep,color='.5',ls=':',label=f'ADE-selected checkpoint epoch{bestep}')
        ax.set(xlabel='Completed epoch (markers = actual observations)',ylabel=ylabel,title=title);ax.legend(fontsize=7)
        fig.text(.02,-.02,caption,fontsize=7)
        if bestep is not None:fig.text(.02,-.07,'Best checkpoint SHA: '+decision['best_checkpoint']['sha256'],fontsize=6)
        fig.tight_layout();save(fig,name)
    curve('01_training_total_loss',ep,[('train_loss','Original total BCE x1000')],'Training loss; all targets once per epoch','Target-weighted mean training loss')
    curve('02_training_goal_path',ep,[('goal_loss','Goal BCE x1000'),('path_loss','Path BCE x1000')],'Original loss components','Target-weighted mean training loss')
    values=[dict(epoch=r['epoch'],**r['window']) for r in val]
    curve('03_validation_minADE20',values,[('minADE20','minADE@20')],'Validation K20 oracle; all998 targets','Displacement error (raw pixels)')
    curve('04_validation_minFDE20',values,[('minFDE20','minFDE@20')],'Validation K20 oracle; independently selected head','Displacement error (raw pixels)')
    curve('05_validation_K1',values,[('ADE1','ADE1'),('FDE1','FDE1')],'Validation first complete candidate (K1)','Displacement error (raw pixels)')
    if bestep is None:
        (FIG/'NOT_RUN.md').write_text('No complete validation checkpoint; no invented cases or validation plots.\n');return
    best=next(r for r in val if r['epoch']==bestep)
    fig,ax=plt.subplots(figsize=(8,3.6));x=np.arange(5)
    ax.bar(x-.18,[r['minADE20'] for r in best['scenes']],.36,label='minADE@20')
    ax.bar(x+.18,[r['minFDE20'] for r in best['scenes']],.36,label='minFDE@20')
    ax.set_xticks(x,[r['scene']+f"\n(n={r['targets']})" for r in best['scenes']],rotation=20)
    ax.set(ylabel='Error (raw pixels)',title=f'Best ADE-selected epoch{bestep}: K20 oracle by recording');ax.legend()
    fig.text(.02,-.05,caption+' | K20',fontsize=7);fig.tight_layout();save(fig,'06_recording_comparison')
    pinfo=best['prediction_manifest'];assert sha(ROOT/pinfo['path'])==pinfo['sha256']
    pack=torch.load(ROOT/pinfo['path'],map_location='cpu',weights_only=True)
    assert pack['checkpoint_SHA']==decision['best_checkpoint']['sha256']
    index={s:i for i,s in enumerate(pack['sample_ids'])}
    rows=list(csv.DictReader((OUT/f'paired_epoch{bestep:03d}.csv').open()))
    df=load('validation');groups={int(k):g for k,g in df.groupby('metaId',sort=False)}
    mids={r['sample_id']:int(r['meta_id']) for r in csv.DictReader((OUT/'validation_target_manifest.csv').open())}
    cases=[];coords=[]
    for group,key,quantiles in [('low','minADE20',[.05,.10,.15]),('high','minADE20',[.85,.90,.95]),('large_goal','minGoalFDE20',[.85,.90,.95])]:
        ordered=sorted(rows,key=lambda r:(float(r[key]),r['sample_id']))
        for number,q in enumerate(quantiles,1):
            r=ordered[int(np.floor(q*(len(ordered)-1)))];sid=r['sample_id'];i=index[sid]
            xy=groups[mids[sid]][['x','y']].to_numpy(dtype=float);h,y=xy[:8],xy[8:]
            paths=pack['paths'][:,i].numpy();goals=pack['goals'][:,i].numpy();winner=int(r['best_ADE_head'])
            path=RES/'data/SDD/train'/r['scene']/'reference.jpg';bgr=cv2.imread(str(path));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            fig,ax=plt.subplots(figsize=(8,7));ax.imshow(rgb)
            for k,p in enumerate(paths):ax.plot(p[:,0],p[:,1],color='#8B9EB7',alpha=.5,lw=1,label='Predicted K20' if k==0 else None)
            ax.plot(h[:,0],h[:,1],'-o',color='#272727',ms=3,lw=2,label='History8')
            ax.plot(y[:,0],y[:,1],'-o',color='#D17B61',ms=3,lw=2,label='GT future12')
            ax.plot(paths[winner,:,0],paths[winner,:,1],color='#267E7C',lw=2.5,label='ADE oracle path')
            ax.scatter(goals[:,0],goals[:,1],marker='x',color='#6B4C9A',s=25,label='Sampled goals20')
            ax.scatter(y[-1,0],y[-1,1],marker='*',color='#F2BB54',edgecolor='black',s=130,label='GT endpoint',zorder=10)
            allxy=np.concatenate([h,y,paths.reshape(-1,2),goals])
            ax.set_xlim(min(-.5,allxy[:,0].min()-5),max(rgb.shape[1]-.5,allxy[:,0].max()+5))
            ax.set_ylim(max(rgb.shape[0]-.5,allxy[:,1].max()+5),min(-.5,allxy[:,1].min()-5))
            ax.set(xlabel='Reference x (pixels)',ylabel='Reference y (pixels)',title=f"{group} q={q:.2f} | {r['scene']} | epoch{bestep}\n{sid}")
            ax.legend(fontsize=7,loc='best')
            fig.text(.02,.01,f"ADE1/FDE1={float(r['ADE1']):.3f}/{float(r['FDE1']):.3f} px | minADE20/minFDE20={float(r['minADE20']):.3f}/{float(r['minFDE20']):.3f} px\nGoalFDE1/minGoalFDE20={float(r['GoalFDE1']):.3f}/{float(r['minGoalFDE20']):.3f} px | K20; GT only offline\ncheckpoint SHA: {pack['checkpoint_SHA']}",fontsize=7)
            fig.tight_layout(rect=(0,.09,1,1));name=f'case_{group}_{number}';save(fig,name)
            cases.append(dict(group=group,quantile=q,case=name,sample_id=sid,scene=r['scene'],epoch=bestep,checkpoint_SHA=pack['checkpoint_SHA'],
                minADE20=float(r['minADE20']),minFDE20=float(r['minFDE20']),GoalFDE1=float(r['GoalFDE1']),minGoalFDE20=float(r['minGoalFDE20']),
                image_SHA=sha(path),figure=name+'.png'))
            def add(kind,head,array):
                for t,(xx,yy) in enumerate(array):coords.append(dict(case=name,sample_id=sid,kind=kind,head=head,step=12 if kind=='goal' else t+1,x=xx,y=yy))
            add('history',-1,h);add('GT_future',-1,y)
            for k,p in enumerate(paths):add('prediction',k,p);add('goal',k,goals[k:k+1])
    cnew(OUT/'qualitative_case_coordinates.csv',coords)
    duplicated={s:[r['case'] for r in cases if r['sample_id']==s] for s in {r['sample_id'] for r in cases} if sum(r['sample_id']==s for r in cases)>1}
    jnew(OUT/'qualitative_cases.json',dict(cases=cases,selection='Frozen rank quantiles; no aesthetic reselection',overlaps=duplicated,
        oracle_GT_used_only_offline=True,all20_paths_preserved=True,coordinate_modification=False))
    print('Actual curves and9 quantile-selected cases saved; duplicate IDs',len(duplicated),flush=True)

if __name__=='__main__':main()
