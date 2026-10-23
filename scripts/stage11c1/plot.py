"""Evidence-only PNG/SVG exports for the budget-rejected matched control.

The protocol schematic describes a frozen plan, not an executed training run.
The auxiliary cost panel uses measured smoke timings and labelled projections.
No absent performance curve, comparison bar, or real-case plate is fabricated.
"""
from pathlib import Path
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .common import ROOT, OUT, CONF, FIG, sha, jnew

REQUESTED = (
    ('01_matched_experiment_protocol', 'Semantic and NoSemantic matched protocol'),
    ('02_training_losses', 'Actual30-epoch training losses'),
    ('03_validation_minADE20', 'Seven complete validation minADE20 nodes'),
    ('04_validation_GoalCover25', 'Seven complete validation GoalCover25 nodes'),
    ('05_full_metric_comparison', 'Semantic versus NoSemantic full metrics'),
    ('06_recording_minADE20_difference', 'All five recording minADE20 differences'),
    ('07_recording_GoalCover25_difference', 'All five recording GoalCover25 differences'),
    ('08_motion_group_semantic_benefit', 'All frozen history motion groups'),
    ('09_original_A25_failures', 'Original164 failures and834 successes'),
    ('10_real_success_failure_cases', 'Actual RGB and paired K20 predictions'),
    ('11_K1_K20_semantic_benefit', 'K1 versus K20 semantic benefit'),
    ('12_fixed_epoch30_vs_best', 'Fixed Epoch30 versus best-checkpoint comparison'),
)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise RuntimeError('CPU-only plotting requires CUDA_VISIBLE_DEVICES empty')
    execution = json.loads((OUT/'training_execution.json').read_text())
    if execution.get('completed_epochs', 0) != 0:
        raise RuntimeError('This figure entry point documents pretraining budget rejection only')
    pre = json.loads((OUT/'preflight_results.json').read_text())
    budget = json.loads((OUT/'budget_preestimate.json').read_text())
    actual = json.loads((OUT/'gpu_budget.json').read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    source_dir = FIG/'source_data'; source_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False})
    records = []

    def source(name, obj):
        path = source_dir/name; jnew(path,obj)
        return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))

    def save(fig, name, src, claim, kind='quantitative grid'):
        for suffix in ('png', 'svg'):
            p = FIG/(name+'.'+suffix)
            if p.exists():
                raise FileExistsError(str(p))
            fig.savefig(p, dpi=220, bbox_inches='tight')
            if suffix=='svg':
                p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
        records.append(dict(name=name,status='AVAILABLE',PNG_SHA=sha(FIG/(name+'.png')),
                            SVG_SHA=sha(FIG/(name+'.svg')),source=src,claim=claim,archetype=kind,
                            statistics='No new validation metrics; engineering timings are not forecast accuracy evidence'))
        plt.close(fig)

    protocol_source = source('01_matched_experiment_protocol.json',dict(
        purpose='Freeze the planned comparison and clearly mark it unexecuted',
        protocol='HTraj-YNet-SDD-Development-v1', train_recordings=25, train_targets=7496,
        validation_recordings=5, validation_targets=998, history=8,future=12, resize=.25,
        seed=17,predictor_parameters=1641381, total_parameters=53155339,
        input_channels=dict(semantic=6,history_heatmaps=8), new_semantics='Six float32 zero channels, padding included',
        planned_epochs=30, actual_formal_epochs=0, actual_full_validations=0,
        validation_epochs=[1,5,10,15,20,25,30], K=20, proposals=10000, kmeans_centers=19,
        stopping_reason='Conservative cumulative GPU admission exceeds5h; formal training not started',
        semantic_input_zero_gradient_elements=pre['zero_semantic_gradient_elements'],
        effective_capacity_warning='Matching nominal parameter count does not match effective input connections'))
    fig,ax=plt.subplots(figsize=(11,4.8));ax.set_xlim(0,14);ax.set_ylim(0,6);ax.axis('off')
    def box(x,y,w,h,text,color='#E7ECEF'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#7A858B',lw=.7))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#6D777C',lw=1))
    box(.2,3.35,2.6,1.5,'Frozen development pool\n25 train recordings /7496 targets\n5 validation recordings /998 targets\nHistory8, future12; raw pixels')
    box(3.4,4.35,2.5,1.0,'Historical Semantic baseline\n6 original semantic channels\nCompleted Epoch30', '#DDE7EF')
    box(3.4,2.2,2.5,1.4,'Planned NoSemantic control\n6 exactly zero channels\nSame random initialization\nNot formally trained','#ECE2DE')
    box(6.65,3.3,2.5,1.5,'Same Y-Net architecture\n14 inputs; same loss and Adam\nSame7496 visits /epoch\n1882 updates /epoch')
    box(10,3.3,3.4,1.5,'Planned K20 evaluation\nTTST10000;19 centers +1 mean\nSame seven validation nodes\nActual new full validations:0')
    arrow((2.9,4.3),(3.3,4.8));arrow((2.9,3.65),(3.3,2.9))
    arrow((6,4.8),(6.55,4.4));arrow((6,2.9),(6.55,3.7));arrow((9.25,4),(9.9,4))
    ax.text(.3,1.25,'Engineering preflight passed. Conservative budget admission failed before formal training.',fontsize=10,color='#9F503C')
    ax.text(.3,.55,'Matching nominal architecture is not exact effective-capacity control:1728 input-weight elements receive zero gradient.\nNo semantic-benefit conclusion, no new ADE/FDE, and no fabricated missing curves.',fontsize=8)
    save(fig,REQUESTED[0][0],protocol_source,'Planned matched control; formal training and full validation remain unexecuted','schematic-led composite')

    budget_source = source('S01_preflight_and_budget.json',dict(
        measured=pre, cumulative_actual=actual, projection=budget,
        claim='Small engineering probes cannot support conservative admission within the unchanged5h budget',
        limitations=['Projected epoch is not a measured full epoch.',
                     'KMeans timing depends on learned probability maps; two targets do not bound future validation cost.',
                     'Cost rejection is not evidence that semantic prediction benefits are absent.']))
    fig,axs=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    x=['Mean','Maximum'];y=[pre['mean_K20_target_seconds'],pre['maximum_K20_target_seconds']]
    b=axs[0].bar(x,y,color=['#7299AB','#B17B67']);axs[0].bar_label(b,fmt='%.3f',padding=3)
    axs[0].set_ylabel('K20 target time (s)');axs[0].set_ylim(0,max(y)*1.25);axs[0].set_title('Measured:2 unique targets,6 calls')
    axs[1].bar(['Whole GPU worker'],[actual['actual_GPU_related_process_wall_seconds']],color='#7299AB')
    axs[1].text(0,actual['actual_GPU_related_process_wall_seconds'],f"{actual['actual_GPU_related_process_wall_seconds']:.3f}s",ha='center',va='bottom')
    axs[1].set_ylabel('Cumulative actual wall time (s)');axs[1].set_ylim(0,actual['actual_GPU_related_process_wall_seconds']*1.25)
    axs[1].set_title('Measured preflight; no formal run')
    labels=['Historical timing\noptimistic reference','Slow first validation,\nhistorical rest','Conservative all\nseven validations']
    values=[budget['historical_only_optimistic_seconds'],budget['first_validation_slow_rest_historical_high_seconds'],budget['conservative_all_validations_seconds']]
    b=axs[2].bar(range(3),np.array(values)/3600,color=['#AAB5BA','#B7907D','#B17B67'])
    axs[2].bar_label(b,fmt='%.2f',padding=2,fontsize=7)
    axs[2].axhline(5,color='#555555',ls='--',label='Hard5h');axs[2].axhline(16800/3600,color='#555555',ls=':',label='Safety4.67h')
    axs[2].set_xticks(range(3),labels,fontsize=6);axs[2].set_ylabel('Projected cumulative time (h)')
    axs[2].set_title('Estimates, not executed full-run times');axs[2].legend(fontsize=7)
    save(fig,'S01_preflight_cost_and_budget',budget_source,'Actual smoke cost plus explicitly labelled uncertain full-run estimates')
    available={r['name']:r for r in records}
    requested=[]
    for name,title in REQUESTED:
        requested.append(available[name] if name in available else dict(name=name,title=title,status='NOT_GENERATED_NO_FORMAL_RUN',
            reason='No NoSemantic formal epoch or complete validation; comparative figure would require nonexistent observations',source=None))
    jnew(OUT/'figure_manifest.json',dict(requested=requested,supplemental=[r for r in records if r['name'].startswith('S')],
        requested_count=12,requested_available=1,requested_not_generated=11,
        numerical_panels_fabricated=False,prototype_schema='Budget-rejected protocol; no comparative performance figures',
        backend='Python/matplotlib Agg',export='PNG220dpi + SVG editable text + exact source JSON',
        visual_QA_required=['No clipped labels','Actual versus projected times clearly labelled','No new validation metrics shown']))
    print('Generated protocol and actual engineering budget figures;11 requested comparison figures not generated',flush=True)


if __name__=='__main__':
    main()
