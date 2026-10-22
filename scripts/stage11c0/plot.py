"""Ten evidence-grounded figures, actual map source arrays, no prediction changes."""
from .common import *
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

COLORS=['#777777','#B16A58','#4F8198','#A18A4A'];CONDS=['S0','S1','S2','S3']
def main():
    cpu_only();assert json.loads((OUT/'score_integrity.json').read_text())['passed']
    FIG.mkdir(exist_ok=True,parents=True);(FIG/'source_data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    data=json.loads((CONF/'generation_config.json').read_text());s=pd.read_csv(OUT/'scene_comparison.csv');full=s[s.experiment=='SEMANTIC_TO_GOAL_AND_PATH'];fixed=s[s.experiment=='SEMANTIC_TO_PATH_GIVEN_GOAL']
    bf=pd.read_csv(OUT/'fixed_goal_path_metrics.csv');cases=pd.read_csv(OUT/'qualitative_case_manifest.csv');manifest=[]
    def export(name,obj):
        path=FIG/'source_data'/name
        if name.endswith('.gz'):
            with gzip.open(path,'wt') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False)
        else:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
    def save(fig,name,source,claim):
        fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight');path=FIG/(name+'.svg');fig.savefig(path,bbox_inches='tight')
        path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        manifest.append(dict(name=name,PNG_SHA=sha(FIG/(name+'.png')),SVG_SHA=sha(path),source=source,claim=claim,
            statistics='descriptive same-target comparisons; repeated development data, overlapping windows, no independent-scene p-values'))
        plt.close(fig)
    # Figure1 is a code-native schematic of the actual frozen computation, not fabricated numerical evidence.
    fig,ax=plt.subplots(figsize=(10,4.0));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
    def box(x,y,w,h,text,color='#E3E7EA'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#888888',lw=.7));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1,color='.4'))
    box(.2,3,2.1,1.4,'Frozen history8\n+ six semantic channels\nS0 / zero / mean / flip')
    box(3,3,2.1,1.4,'Frozen Y-Net encoder\nSame history heatmaps\nSame parameters')
    box(5.8,3,2.2,1.4,'A: Goal logits\nCPU TTST10000\n19 centers + Softargmax')
    box(8.7,3,2.7,1.4,'Frozen path decoder\nK20 x12\nA: new goals, new features')
    box(5.8,.5,2.2,1.4,'B: original S0 goals\nNo sampling\nNo reclustering')
    box(8.7,.5,2.7,1.4,'Frozen path decoder\nK20 x12\nB: fixed goals, new features')
    arrow((2.4,3.7),(2.9,3.7));arrow((5.2,3.7),(5.7,3.7));arrow((8.1,3.7),(8.6,3.7));arrow((5.1,3.2),(8.6,1.7));arrow((8.1,1.2),(8.6,1.2))
    ax.text(.3,.65,'All predictions saved before GT scoring.\nNo training / no class relabeling / no test access.\nSame sample seeds; changed categorical distributions.',fontsize=8)
    save(fig,'01_experiment_framework','configs/stage11c0/experiment_protocol.json','Separate full-chain sensitivity from path sensitivity given identical goals')
    # Figure2: true six channels, raw values0..1, no unverified class labels.
    scene='deathCircle_3';sem={c:checked(data['semantics'][c][scene]).numpy()[0] for c in CONDS};rec=data['semantics']['S0'][scene]
    fig,axs=plt.subplots(4,6,figsize=(12,8),layout='constrained');sources=[]
    for ch in range(6):
        sources.append(export(f'02_channel{ch}.json.gz',dict(scene=scene,channel=ch,valid_height=rec['valid_height'],valid_width=rec['valid_width'],
            tensor_sources={c:data['semantics'][c][scene] for c in CONDS},arrays={c:sem[c][ch].tolist() for c in CONDS})))
        for row,c in enumerate(CONDS):
            ax=axs[row,ch];image=ax.imshow(sem[c][ch],vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
            ax.axhline(rec['valid_height']-.5,color='white',lw=.5,ls=':');ax.axvline(rec['valid_width']-.5,color='white',lw=.5,ls=':')
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{c}: channel{ch}',fontsize=7)
    fig.colorbar(image,ax=axs.ravel().tolist(),shrink=.5,label='Actual semantic probability (unknown class map)')
    save(fig,'02_real_semantic_inputs',sources,'Original/zero/valid spatial mean/valid horizontal flip; dotted valid boundaries; actual padding retained in S2/S3')
    from scripts.stage11b1c.data import truths
    truth=truths('validation');hp=checked(data['history']);old=checked(data['original_predictions']);idx={sid:i for i,sid in enumerate(hp['sample_ids'])}
    packs={'S0':old}
    for r in json.loads((OUT/'full_generation.json').read_text())['groups']:packs[r['condition']]=checked(r)
    mm={'S0':{r['sample_id']:r for r in data['original_tensor_records']}}
    for c in CONDS[1:]:mm[c]=pd.read_csv(OUT/f'{c}_tensor_manifest.csv').set_index('sample_id').to_dict('index')
    selected=cases[cases.case_type=='scene_median_ID'];maps={};sources=[]
    for _,row in selected.iterrows():
        sid=row.sample_id;scene=row.scene;arrays={}
        for c in CONDS:
            x=checked(mm[c][sid])['tensors'];p=x['normalized_weights'].numpy().reshape(x['sigmoid'].shape[-2:]).astype(float);p=p/p.sum();arrays[c]=p
        maps[scene]=(sid,arrays);sources.append(export('03_'+scene+'.json.gz',dict(sample_id=sid,scene=scene,grid_weights={c:v.tolist() for c,v in arrays.items()},
            sources={c:mm[c][sid] for c in CONDS},history_raw=hp['history'][idx[sid]].tolist(),GT_raw=truth[sid].tolist())))
    vmax=max(float(v.max()) for sid,arrays in maps.values() for v in arrays.values());fig,axs=plt.subplots(5,4,figsize=(10,11),layout='constrained')
    for row,(scene,(sid,arrays)) in enumerate(maps.items()):
        for col,c in enumerate(CONDS):
            ax=axs[row,col];v=arrays[c];im=ax.imshow(np.ma.masked_less_equal(v,0),norm=LogNorm(vmin=1e-8,vmax=vmax),cmap='magma',interpolation='nearest')
            history=hp['history'][idx[sid]].numpy()*.25;ax.plot(history[:,0],history[:,1],c='#77CBCD',lw=.7)
            ax.scatter(*(truth[sid][-1]*.25),marker='*',c='white',edgecolors='.4',s=20,lw=.4)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{scene} | {c}',fontsize=7)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.5,label='Actual postthreshold normalized categorical probability; log scale')
    save(fig,'03_real_goal_probability_maps',sources,'Five fixed median-ID cases, same log map scale with visual floor1e-8; GT star is offline only; zero weights masked; raw weights unmodified')
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.3),layout='constrained')
    for ax,scope in zip(axs,['ALL998','FIVE_RECORDING_MACRO']):
        q=full[full.scope==scope].set_index('condition').loc[CONDS];bars=ax.bar(CONDS,q.GoalCover25*100,color=COLORS);ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
        ax.set_ylim(0,108);ax.set_ylabel('GoalCover25 (%)');ax.set_title(scope)
    save(fig,'04_goal_coverage25','results/tables/stage11c0/scene_comparison.csv','Full prediction only; fixed sample and radius; window and equal-recording macro kept separate')
    fig,axs=plt.subplots(2,2,figsize=(8.5,6),layout='constrained')
    for row,scope in enumerate(['ALL998','FIVE_RECORDING_MACRO']):
        q=full[full.scope==scope].set_index('condition').loc[CONDS]
        for ax,key in zip(axs[row],['minADE20','minFDE20']):
            bars=ax.bar(CONDS,q[key],color=COLORS);ax.bar_label(bars,fmt='%.3f',fontsize=7,padding=2);ax.set_ylim(0,q[key].max()*1.18)
            ax.set_ylabel(key+' (px)');ax.set_title(scope)
    save(fig,'05_full_path_errors','results/tables/stage11c0/scene_comparison.csv','Actual K20 full-chain errors; minADE and minFDE may choose different oracle heads')
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.6),layout='constrained')
    for c,color in zip(CONDS[1:],COLORS[1:]):
        q=bf[bf.condition==c]
        for ax,key in zip(axs,['Path_point_mean_displacement','Head0_point_mean_displacement']):
            x=np.sort(q[key]);ax.plot(x,np.arange(1,len(x)+1)/len(x),color=color,label=c)
            ax.set_xscale('symlog',linthresh=1);ax.set_xlabel(key+' (px)');ax.set_ylabel('Fraction of target windows');ax.legend()
    save(fig,'06_fixed_goal_path_displacement','results/tables/stage11c0/fixed_goal_path_metrics.csv','Same20 goals and head identities; movement is sensitivity, not an improvement metric')
    names=sorted(hp['scenes']);names=sorted(set(names));fig,axs=plt.subplots(1,3,figsize=(10,4.0),layout='constrained')
    for ax,key,frame,title in zip(axs,['JS_nats','Hungarian_mean_distance','Path_point_mean_displacement'],[full,full,fixed],['Goal JS (nats)','Goal set match (px)','Fixed-goal path move (px)']):
        matrix=np.array([[frame[(frame.scope==scene)&(frame.condition==c)].iloc[0][key] for c in CONDS[1:]] for scene in names])
        im=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0);ax.set_xticks(range(3),CONDS[1:]);ax.set_yticks(range(5),names);ax.set_title(title)
        for row in range(5):
            for col in range(3):ax.text(col,row,f'{matrix[row,col]:.3g}',ha='center',va='center',fontsize=7,color='white' if matrix[row,col]>.65*matrix.max() else 'black')
        fig.colorbar(im,ax=ax,shrink=.7)
    save(fig,'07_recording_sensitivity','results/tables/stage11c0/scene_comparison.csv','All five reused recordings, no independent-scene generalization claim')
    groups=pd.read_csv(OUT/'motion_group_comparison.csv');fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    factors=['hist_speed_group','hist_turn_group','hist_speed_change_group','hist_stationary_group']
    for ax,factor in zip(axs.flat,factors):
        frame=groups[(groups.scope=='ALL998')&(groups.factor==factor)&(groups.experiment=='SEMANTIC_TO_PATH_GIVEN_GOAL')];labels=sorted(frame.group.unique());x=np.arange(len(labels))
        for c,color in zip(CONDS[1:],COLORS[1:]):
            q=frame[frame.condition==c].set_index('group').reindex(labels);ax.plot(x,q.Path_point_mean_displacement,marker='o',ms=3,color=color,label=c)
        counts=frame[frame.condition=='S1'].set_index('group');ax.set_xticks(x,[f'{z}\nn={int(counts.loc[z,"targets"])}' for z in labels],fontsize=6)
        ax.set_title(factor);ax.set_ylabel('Fixed-goal mean path move (px)');ax.set_ylim(bottom=0);ax.legend(fontsize=7)
    save(fig,'08_history_group_sensitivity','results/tables/stage11c0/motion_group_comparison.csv','All frozen history groups; means reflect recording composition, not causal motion dependence')
    transitions=pd.read_csv(OUT/'a25_transition_summary.csv');q=transitions[transitions.scope=='ALL998'].set_index('condition').loc[CONDS[1:]]
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.5),layout='constrained');x=np.arange(3)
    for j,(key,color) in enumerate([('recovered','#4F8198'),('newly_lost','#B16A58')]):
        bars=axs[0].bar(x+(j-.5)*.34,q[key],width=.34,color=color,label=key);axs[0].bar_label(bars,fontsize=7)
    axs[0].set_xticks(x,CONDS[1:]);axs[0].set_ylabel('Target count');axs[0].legend();axs[0].set_ylim(0,max(q.recovered.max(),q.newly_lost.max())*1.2+1)
    bars=axs[1].bar(CONDS[1:],q.total_new_failures,color=COLORS[1:]);axs[1].bar_label(bars);axs[1].axhline(164,c='.4',ls='--',label='S0:164');axs[1].set_ylim(0,max(164,q.total_new_failures.max())*1.2);axs[1].set_ylabel('Total A25 failures');axs[1].legend()
    save(fig,'09_failure_recovery_and_new_failures','results/tables/stage11c0/a25_transition_summary.csv','Fixed original164 failures and834 successes; recoveries always shown with new losses')
    fig,axs=plt.subplots(3,4,figsize=(14,10),layout='constrained');raw=[]
    for row,c in enumerate(CONDS[1:]):
        for col,outcome in enumerate(['recovered','lost','stayed_fail','stayed_cover']):
            ax=axs[row,col];pick=cases[(cases.condition==c)&(cases.case_type==outcome)]
            if not len(pick):ax.text(.5,.5,'No eligible '+outcome,ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
            case=pick.iloc[0];sid=case.sample_id;i=idx[sid];og=old['goals'][:,i].numpy();ag=packs[c]['goals'][:,i].numpy();op=old['paths'][:,i].numpy();ap=packs[c]['paths'][:,i].numpy();y=truth[sid];h=hp['history'][i].numpy()
            for p in op:ax.plot(p[:,0],p[:,1],c=COLORS[0],alpha=.25,lw=.55)
            for p in ap:ax.plot(p[:,0],p[:,1],c=COLORS[CONDS.index(c)],alpha=.32,lw=.65,ls='--')
            ax.scatter(og[:,0],og[:,1],c=COLORS[0],marker='x',s=12);ax.scatter(ag[:,0],ag[:,1],c=COLORS[CONDS.index(c)],marker='+',s=12)
            ax.plot(h[:,0],h[:,1],c='#4F8198',lw=1);ax.plot(y[:,0],y[:,1],c='black',lw=1.2);ax.scatter(*y[-1],c='black',marker='*',s=25)
            ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis();ax.set_title(f'{c} {outcome}\n{sid.replace("SDD:","")}',fontsize=6);ax.set_xlabel('x (px)');ax.set_ylabel('y (px)')
            raw.append(dict(sample_id=sid,condition=c,outcome=outcome,eligible=int(case.eligible),rank=int(case['rank']),
                original_goals=og.tolist(),alternative_goals=ag.tolist(),original_paths=op.tolist(),alternative_paths=ap.tolist(),history=h.tolist(),GT=y.tolist()))
    fig.legend(handles=[Line2D([],[],color=COLORS[0],marker='x',label='S0 paths / goals')]+
        [Line2D([],[],color=COLORS[i],ls='--',marker='+',label=c+' paths / goals') for i,c in enumerate(CONDS) if i>0]+
        [Line2D([],[],color='black',label='GT (offline)'),Line2D([],[],color='#4F8198',label='History')],
        loc='outside upper center',ncol=6,fontsize=7)
    source=export('10_real_paired_cases.json',raw);save(fig,'10_real_goal_path_cases',source,'Fixed median-ID for each paired outcome; all20 paths shown, no GT-based head correspondence')
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('FIGURES',len(manifest),'PNG/SVG pairs; real source arrays exported',flush=True)
if __name__=='__main__':main()
