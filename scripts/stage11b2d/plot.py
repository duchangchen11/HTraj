"""Python-only figures from frozen coordinates; no resampling or model calls."""
from .common import *
from .geometry import assign_final
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def main():
    cpu_guard();FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    f=pd.read_csv(OUT/'target_diagnostics.csv');r=pd.read_csv(OUT/'recording_comparison.csv')
    z=pd.read_csv(OUT/'threshold_support_review.csv');cfg=json.loads((CONF/'analysis_protocol.json').read_text())
    fail=f[f.category25=='proposal_hit_final_miss'];success=f[f.category25=='both_hit']
    colors=['#C16649','#517F98'];figures=[]
    def save(fig,name,data,claim):
        fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight')
        fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
        svg=FIG/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        figures.append(dict(name=name,PNG_SHA=sha(FIG/(name+'.png')),SVG_SHA=sha(FIG/(name+'.svg')),source_data=data,
            claim=claim,scope='descriptive posthoc groups; overlapping targets; no significance test'))
        plt.close(fig)
    def ecdf(ax,key,label,scale=None):
        for q,color,name in [(fail,colors[0],'Lost coverage (n=86)'),(success,colors[1],'Retained (n=834)')]:
            a=np.sort(q[key].dropna().to_numpy());ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',color=color,label=name,lw=1.5)
        if scale:ax.set_xscale(scale)
        ax.set_xlabel(label);ax.set_ylabel('Target-window ECDF');ax.set_ylim(0,1.02)
    fig,ax=plt.subplots(figsize=(5.3,3.5));ecdf(ax,'LocalProposalCount25','Draws within GT 25 px (duplicates included)','log')
    for v in [10,100]:ax.axvline(v,color='.7',ls=':',lw=.8)
    ax.legend(loc='upper left');save(fig,'01_local_draw_counts','target_diagnostics.csv','Lost-coverage targets have fewer local draws; not a causal group comparison')
    fig,axs=plt.subplots(2,2,figsize=(7.4,5.8),layout='constrained')
    for ax,key,label,scale in zip(axs.flat,['LocalWeightedClusterPopulation','AssignedClusterRadiusP90','LocalDrawsInLargeClustersFraction','LocalAssignedClusterCount25'],
        ['Cluster population (local-draw weighted)','Assigned-cluster P90 radius (px; locally weighted)','Fraction of local draws in larger clusters','Number of clusters touched by local draws'],[None,None,None,None]):
        ecdf(ax,key,label,scale)
    axs[0,0].legend(fontsize=7);save(fig,'02_final_cluster_structure','target_diagnostics.csv;cluster_diagnostics.csv','The loss group is not generally absorbed by the largest clusters')
    fig,axs=plt.subplots(1,3,figsize=(11,4.0),layout='constrained')
    manifest=pd.read_csv(OLD/'tensor_manifest.csv').set_index('sample_id');cases=[]
    for ax,sid in zip(axs,cfg['case_ids']):
        row=manifest.loc[sid];pack=torch.load(ROOT/row.path,map_location='cpu',weights_only=True);t=pack['tensors']
        q=f.set_index('sample_id').loc[sid];P=t['proposals_raw'][0,0].numpy();C=t['centers_raw'].numpy();G=np.array([q.GT_x,q.GT_y]);A,_=assign_final(P,C)
        loc=np.linalg.norm(P-G,axis=1)<=25
        ax.scatter(P[:,0],P[:,1],c=A,cmap='tab20',s=2,alpha=.35,rasterized=True)
        ax.scatter(P[loc,0],P[loc,1],s=11,facecolors='none',edgecolors='black',lw=.4,label='Local draws')
        ax.scatter(C[:,0],C[:,1],c=colors[0],marker='x',s=35,lw=1.2,label='19 frozen centers')
        ax.scatter(*G,c='black',marker='*',s=60,label='GT (offline)')
        soft=t['softargmax_raw'].numpy();ax.scatter(*soft,c='#7E6197',marker='D',s=22,label='Softargmax')
        inset=ax.inset_axes([.025,.69,.28,.27])
        inset.scatter(P[:,0],P[:,1],s=.35,c='.65',alpha=.4,rasterized=True)
        inset.scatter(C[:,0],C[:,1],s=6,c=colors[0],marker='x',lw=.5)
        inset.scatter(*G,s=13,c='black',marker='*')
        inset.scatter(*soft,s=5,c='#7E6197',marker='D')
        inset.invert_yaxis();inset.set_aspect('equal',adjustable='datalim')
        inset.set_xticks([]);inset.set_yticks([]);inset.set_title('Full cloud / all 19',fontsize=5,pad=2)
        ax.add_patch(Circle(G,25,fill=False,color='black',lw=1,ls='--'))
        ax.set_aspect('equal');ax.set_xlim(G[0]-115,G[0]+115);ax.set_ylim(G[1]+115,G[1]-115)
        ax.set_xlabel('SDD reference x (px)');ax.set_ylabel('SDD reference y (px)')
        ax.set_title(sid.replace('SDD:deathCircle_3:', '')+'\nlocal='+str(q.LocalProposalCount25)+'; center-GT='+f'{q.NearestCenterToGT25:.2f} px',fontsize=8)
        case=dict(sample_id=sid,scene=q.scene,checkpoint_SHA=CK,cache_file_SHA=row.sha256,
            probability_SHA=row.probability_SHA,proposal_network_SHA=row.proposal_SHA,centers_network_SHA=row.centers_SHA,
            centers_raw_SHA=tensor_sha(t['centers_raw']),goal_SHA=row.goal_SHA,indices_SHA=row.indices_SHA,
            GT=[float(v) for v in G],local_radius25=25,relative_threshold=.01,
            center_coordinates=C.tolist(),softargmax_coordinates=soft.tolist(),
            local_proposal_indices=np.flatnonzero(loc).tolist(),local_proposal_coordinates=P[loc].tolist(),
            local_assignment=A[loc].tolist(),ProposalMinDist=float(q.ProposalMinDist),FinalMinDist20=float(q.FinalMinDist20),
            minADE20=float(q.minADE20),minFDE20=float(q.minFDE20),assignment=cfg['assignment'],
            source_cache=str(row.path),case_rule='reuse Stage2C frozen ranks; viewport only, statistics use all10000')
        cases.append(case)
    axs[-1].legend(loc='upper right',fontsize=6)
    save(fig,'03_frozen_spatial_cases','frozen_case_data.json; original cache files via SHA','Three preregistered cases, including a large-cluster exception; no coordinate movement')
    if (OUT/'frozen_case_data.json').exists():
        assert json.loads((OUT/'frozen_case_data.json').read_text())==cases
    else:jnew(OUT/'frozen_case_data.json',cases)
    fig,axs=plt.subplots(1,2,figsize=(8.0,3.7),layout='constrained')
    scenes=cfg['recordings'];s=r.set_index('scope').loc[scenes]
    axs[0].barh(s.index,s.compression_loss_rate_all*100,color=[colors[0] if k=='deathCircle_3' else colors[1] for k in scenes])
    axs[0].set_xlabel('Compression loss / all targets (%)')
    for i,row in enumerate(s.itertuples()):axs[0].text(row.compression_loss_rate_all*100+.2,i,f'{row.compression_fail}/{row.targets}',va='center',fontsize=7)
    axs[0].set_xlim(0,19)
    for label,q,color in [('deathCircle_3 (65)',fail[fail.scene=='deathCircle_3'],colors[0]),('Other four (21)',fail[fail.scene!='deathCircle_3'],colors[1])]:
        a=np.sort(q.LocalProposalCount25);axs[1].step(a,np.arange(1,len(a)+1)/len(a),where='post',label=label,color=color)
    axs[1].set_xscale('log');axs[1].set_xlabel('Local draws within 25 px');axs[1].set_ylabel('Loss-target ECDF');axs[1].legend(fontsize=7)
    save(fig,'04_recording_contrasts','recording_comparison.csv;target_diagnostics.csv','All five recordings retained; small counts outside deathCircle limit generalization')
    fig,axs=plt.subplots(1,2,figsize=(7.5,3.4),layout='constrained')
    a=np.sort(z.unthresholded_normalized_mass);axs[0].step(a,np.arange(1,len(a)+1)/len(a),where='post',color=colors[0]);axs[0].set_xscale('log')
    axs[0].set_xlabel('Unthresholded normalized mass in GT 25 px');axs[0].set_ylabel('ECDF (77 zero-support targets)')
    axs[1].scatter(z.unthresholded_normalized_mass,z.neighborhood_max_over_global_max,s=14,color=colors[1],alpha=.7)
    axs[1].set_xscale('log');axs[1].axhline(.01,color='.4',ls='--',lw=.8,label='Frozen relative threshold')
    axs[1].set_xlabel('Unthresholded neighborhood mass');axs[1].set_ylabel('Neighborhood / global max weight');axs[1].legend(fontsize=6)
    save(fig,'05_zero_support_raw_mass','threshold_support_review.csv','All77 raw neighborhoods positive but below fixed relative threshold; no counterfactual sampling')
    (OUT/'figure_manifest.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2)+'\n')
    print('FIGURES',len(figures),'PNG/SVG pairs; CPU only',flush=True)

if __name__=='__main__':main()
