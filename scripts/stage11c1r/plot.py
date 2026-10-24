"""Python figures: actual measurements and conditional projections are separated."""
from .common import *
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ORDER=['NoSemantic-initial','NoSemantic-epoch001','Semantic-epoch001','Semantic-epoch005','Semantic-epoch010','Semantic-epoch030']
LABEL=['NS-init','NS-e1','S-e1','S-e5','S-e10','S-e30']
COLORS=['#9C6257','#C09580','#4D748B','#7296A5','#8EABB0','#B2C4C1']

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    sources=FIG/'source_data';sources.mkdir(parents=True,exist_ok=True);manifest=[]
    def save(fig,name,claim,archetype,data):
        src=sources/f'{name}.json';jnew(src,dict(claim=claim,archetype=archetype,
            limitations='15fixed histories, not998targets; no GT metrics; development recordings already exposed',data=data))
        for ext in ['png','svg']:
            path=FIG/f'{name}.{ext}';assert not path.exists();fig.savefig(path,dpi=220,bbox_inches='tight')
            if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        manifest.append(dict(name=name,claim=claim,source=str(src.relative_to(ROOT)),source_SHA=sha(src),
            PNG_SHA=sha(FIG/f'{name}.png'),SVG_SHA=sha(FIG/f'{name}.svg')));plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.4));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='#66757D',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    box(.2,3,2.6,1.25,'CUDA proposals\n10000 x 2 float32\nCUDA centers: 19 x 2','#DDE7EC')
    box(4.1,3,3.1,1.25,'pairwise_distance default: CPU\nDistance matrix + argmin: CPU\n80,152 explicit D2H bytes / iter','#EADDD5')
    box(8.7,3,2.8,1.25,'CUDA selection + mean\nCUDA center update\nCPU mask/index handling','#DDE7EC')
    box(4.2,.6,3.1,1.15,'CUDA center-shift scalar\nPython stopping condition\nTolerance or 1000 iterations','#E5E9E3')
    for start,end in [((2.9,3.6),(4,3.6)),((7.3,3.6),(8.6,3.6)),((9.5,2.9),(7.4,1.2)),((4.1,1.2),(1.6,2.9))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1,'color':'#56646B'})
    ax.text(7.85,4.55,'Implicit indexing payload: unmeasured',ha='center',fontsize=7)
    ax.text(.2,.05,'Frozen author source: no device argument at pairwise call. Transfer payload is a static estimate; no isolated transfer-time percentage.',fontsize=7)
    save(fig,'01_original_device_path','Original clustering mixes CPU distance/assignment and CUDA center updates',
        'schematic-led composite',json.loads((OUT/'source_audit.json').read_text()))
    if not (OUT/'kmeans_iteration_counts.csv').exists():
        jnew(OUT/'figure_manifest.json',dict(available=manifest,remaining='NOT_RUN: no profiling data'));return
    it=pd.read_csv(OUT/'kmeans_iteration_counts.csv');it=it[it['mode']=='observed']
    raw=pd.read_csv(OUT/'kmeans_runtime.csv');pair=pd.read_csv(OUT/'paired_runtime_by_target.csv')
    fig,ax=plt.subplots(figsize=(8,3.2),layout='constrained')
    for i,state in enumerate(ORDER):
        g=it[it.state==state]
        if not len(g):continue
        ax.scatter(i+np.linspace(-.13,.13,len(g)),g.iterations,color=COLORS[i],s=18,alpha=.8)
        ax.plot([i-.25,i+.25],[g.iterations.median()]*2,color='#263C49',lw=1.5)
    ax.set_ylim(0,float(it.iterations.max())*1.2)
    ax.text(.98,.98,'Configured cap: 1000 (not reached)',transform=ax.transAxes,ha='right',va='top',fontsize=7,color='#66757D')
    ax.set_xticks(range(6),LABEL);ax.set_ylabel('Observed KMeans iterations')
    ax.set_title('Same 15 histories per state; line = median; no GT scoring')
    save(fig,'02_checkpoint_iterations','Iterations are measured by pure Python call counting',
        'quantitative grid',it.to_dict('records'))
    fig,ax=plt.subplots(figsize=(6.5,3.6),layout='constrained')
    for i,state in enumerate(ORDER):
        g=raw[raw.state==state];ax.scatter(g.iterations,g.wall_seconds,s=22,color=COLORS[i],label=LABEL[i],alpha=.8)
    ax.set_xlabel('Iterations (separate observed identical run)');ax.set_ylabel('Original unmonitored KMeans wall (s)')
    ax.legend(ncol=2);ax.set_title('Descriptive relationship; device overhead not isolated')
    save(fig,'03_iterations_and_wall','Unmonitored runtime is paired with independently measured iteration counts',
        'quantitative grid',raw.to_dict('records'))
    fig,axs=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    for ax,key,title in [(axs[0],'wall_seconds','Original KMeans only'),(axs[1],'outer_prediction_seconds','Full K20, passive export included')]:
        initial=pair[pair.state==ORDER[0]].set_index('sample_id');after=pair[pair.state==ORDER[1]].set_index('sample_id')
        common=sorted(set(initial.index)&set(after.index))
        for sid in common:ax.plot([0,1],[initial.loc[sid,key],after.loc[sid,key]],'-o',ms=3,lw=.6,alpha=.65,color='#6B8E9D')
        ax.set_xticks([0,1],['Random init','Diagnostic epoch1']);ax.set_ylabel('Measured wall (s)');ax.set_title(title)
    save(fig,'04_nosemantic_before_after','Initial and post-epoch1 runtime use identical fixed histories',
        'quantitative grid',pair[pair.state.isin(ORDER[:2])].to_dict('records'))
    budgets=json.loads((OUT/'training_budget_scenarios.json').read_text());sc=budgets['scenarios'];ledger=budgets['ledger']
    fig,ax=plt.subplots(figsize=(7.5,3.2),layout='constrained')
    bars=ax.bar(range(len(sc)),[s['total_seconds']/3600 for s in sc],color=['#9CAEB6','#648B9F','#B88672'][:len(sc)],
        hatch=['','///','xxx'][:len(sc)])
    for b,s in zip(bars,sc):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{s['total_seconds']/3600:.2f} h",ha='center',va='bottom')
    ax.axhline(ledger['CombinedRemaining']/3600,color='#4C4C4C',ls='--',label='Remaining original hard budget')
    ax.set_xticks(range(len(sc)),['A: historical reference','B: post-epoch1 projection','C: slow stress'][:len(sc)])
    ax.set_ylabel('Projected future full-run wall (h)');ax.legend(fontsize=7)
    ax.set_title('30 epochs + seven998-target validations: none executed here')
    save(fig,'05_three_budget_scenarios','Measured components inform conditional full-run estimates, not completed experiments',
        'quantitative grid',budgets)
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    a=ledger['OriginalStage11C1Used'];b=ledger['Stage11C1RUsed'];c=ledger['CombinedRemaining']
    for x,left,width,color,label in [(axs[0],0,a,'#9FAAAF','C1 used'),(axs[0],a,b,'#638A9E','C1R used'),(axs[0],a+b,c,'#DDE3E5','Remaining')]:
        x.barh([0],[width],left=[left],height=.35,color=color,label=label)
    axs[0].set_xlim(0,18000);axs[0].set_yticks([]);axs[0].set_xlabel('Cumulative allocation (s)');axs[0].legend(loc='upper center',fontsize=7,ncol=3)
    axs[0].set_title(f'Combined used: {a+b:.2f} s')
    axs[0].set_ylim(-.6,.6)
    axs[1].barh([0],[b],height=.35,color='#638A9E');axs[1].barh([0],[2400-b],left=[b],height=.35,color='#DDE3E5')
    axs[1].set_ylim(-.6,.6)
    axs[1].axvline(2250,ls='--',color='#9B5E4F',label='Safety2250 s');axs[1].set_xlim(0,2400);axs[1].set_yticks([])
    axs[1].set_xlabel('This-stage allocation (s)');axs[1].legend(fontsize=7);axs[1].set_title(f'C1R measured: {b:.2f} s')
    save(fig,'06_actual_GPU_budget','The budget ledger includes complete GPU-related processes and failures',
        'quantitative grid',ledger)
    jnew(OUT/'figure_manifest.json',dict(available=manifest,backend='Python/matplotlib Agg',export='PNG220dpi+editableSVG',
        predictions_scored=False,source_data_required=True))
    print('FIGURES_WRITTEN',len(manifest),flush=True)
if __name__=='__main__':main()
