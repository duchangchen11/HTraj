"""Pair the smoke cohort with fresh CVs and estimate budgets from batch128."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import math
import numpy as np
from src.data.preprocessing import sha256,write_json_new
from src.socialvae_causal.data import CausalDataset,array_sha
from src.socialvae_causal.io import read_csv,csv_new

OUT=Path('results/tables/stage5p1')


def finalize():
    smoke=json.load(open(OUT/'smoke_summary.json'))
    curve=read_csv(OUT/'smoke_epoch_curves.csv')
    resource=json.load(open(OUT/'resource_summary.json'))
    comparisons=read_csv(OUT/'smoke_cv_comparison.csv')
    per_step=sum(float(r['train_epoch_seconds']) for r in curve)/smoke['training_optimizer_steps']
    val_seconds=sum(float(r['evaluation_seconds']) for r in curve)
    current_full_epoch=smoke['elapsed_seconds']/5
    raw=json.load(open(OUT/'raw_counterfactual_summary.json'));coverage=json.load(open(OUT/'coverage_gate.json'))
    ready=raw['passed'] and coverage['passed'] and resource['finite_gate'] and smoke['all_finite'] and smoke['all_strict_restore_passed'] and smoke['train_loss_decreased']
    write_json_new(OUT/'research_decision.json',dict(engineering_ready_for_formal_baseline=bool(ready),
        reasons=['raw eight-frame counterfactual inputs/predictions unchanged','coverage gate passed','all measured batch sizes fit','five complete passes finite with decreasing loss','strict model/optimizer/RNG restore passed'],
        smoke_epochs=5,smoke_optimizer_steps=smoke['training_optimizer_steps'],resource_optimizer_steps=sum(r.get('optimizer_steps',0) for r in resource['rows']),
        selected_checkpoint_epoch=smoke['best']['epoch'],best_checkpoint_sha256=smoke['best']['checkpoint_sha256'],
        current_K1_beats_CV_Last2=bool(float(next(r for r in comparisons if r['CV_model']=='CV-Last2')['delta_ADE_m'])<0),
        social_information_independent_benefit_proven=False,formal_performance_reproduction=False,
        next_recommendation='preregister a finite formal baseline budget on fixed recording-isolated causal manifests; no network modifications; do not begin automatically',
        ETH_evaluated=False,FPC=False,new_method_innovation_claim=False,automatic_push=False,
        measurement_summary=dict(five_epoch_wall_seconds=smoke['elapsed_seconds'],mean_complete_epoch_wall_seconds=current_full_epoch,
            mean_training_step_with_data_seconds=per_step,mean_validation_seconds=val_seconds/5)))
    inventory=[]
    for f in sorted(Path('experiments/stage5p1/checkpoints').rglob('*.pth')):
        inventory.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha256(f)))
    csv_new(OUT/'checkpoint_inventory.csv',inventory)


def main():
    smoke=json.load(open(OUT/'smoke_summary.json'));curve=read_csv(OUT/'smoke_epoch_curves.csv')
    resource=json.load(open(OUT/'resource_summary.json'));plan=json.load(open('configs/stage5p1/protocol.json'))
    data=CausalDataset(plan['data_root'],'configs/stage5p1/manifests/crowds_zara02/B.json','validation')
    target=read_csv(OUT/'smoke'/f"epoch_{smoke['best']['epoch']:02d}_validation_targets.csv.gz")
    cv=read_csv(OUT/'cv_paired_targets.csv.gz')
    lookup={(r['model'],r['recording'],int(r['start_frame']),int(r['target_id'])):r for r in cv if r['fold']=='crowds_zara02'}
    paired=[]
    assert len(target)==len(data)==1092
    for i,row in enumerate(target):
        item=data[i];key=(item['recording'],item['start_frame'],item['actor'])
        assert key==(row['recording'],int(row['start_frame']),int(row['target_id']))
        for kind in plan['CV_models']:
            ref=lookup[(kind,*key)]
            assert ref['future_sha256']==array_sha(item['y']) and ref['history_sha256']==array_sha(item['x'][:,:2])
            a,f=float(row['K1_prior_ADE_m']),float(row['K1_prior_FDE_m'])
            ca,cf=float(ref['ADE_m']),float(ref['FDE_m'])
            paired.append(dict(fold='crowds_zara02',recording=key[0],start_frame=key[1],target_id=key[2],CV_model=kind,
                history_sha256=ref['history_sha256'],future_sha256=ref['future_sha256'],checkpoint_sha256=smoke['best']['checkpoint_sha256'],
                neural_K1_ADE_m=a,neural_K1_FDE_m=f,CV_ADE_m=ca,CV_FDE_m=cf,delta_ADE_m=a-ca,delta_FDE_m=f-cf))
    csv_new(OUT/'smoke_cv_paired_best.csv.gz',paired)
    comparisons=[]
    for kind in plan['CV_models']:
        rows=[r for r in paired if r['CV_model']==kind]
        comparisons.append(dict(fold='crowds_zara02',CV_model=kind,target_windows=len(rows),
            neural_K1_ADE_m=np.mean([r['neural_K1_ADE_m'] for r in rows]),neural_K1_FDE_m=np.mean([r['neural_K1_FDE_m'] for r in rows]),
            CV_ADE_m=np.mean([r['CV_ADE_m'] for r in rows]),CV_FDE_m=np.mean([r['CV_FDE_m'] for r in rows]),
            delta_ADE_m=np.mean([r['delta_ADE_m'] for r in rows]),delta_FDE_m=np.mean([r['delta_FDE_m'] for r in rows]),
            target_window_ADE_improvement_fraction=np.mean([r['delta_ADE_m']<0 for r in rows]),not_independent_samples=True))
    csv_new(OUT/'smoke_cv_comparison.csv',comparisons)
    training_steps=smoke['training_optimizer_steps']
    train_seconds=sum(float(r['train_epoch_seconds']) for r in curve)
    val_seconds=sum(float(r['evaluation_seconds']) for r in curve)
    per_step=train_seconds/training_steps
    current_full_epoch=smoke['elapsed_seconds']/5
    budget=[]
    for epochs in [30,100,600,800]:
        budget.append(dict(scope='ZARA2 causal full-pass epochs',unit='full_passes',budget=epochs,optimizer_steps=184*epochs,
            estimated_hours=epochs*current_full_epoch/3600,reference='actual5-epoch batch128 wall time including checkpoint/validation overhead',
            measured=False,ETH_data=False,limitation='estimate not convergence guarantee; no FPC, hardware contention and later training cost may vary'))
    for steps in [60000,80000]:
        budget.append(dict(scope='nonETH pipeline at original-configuration update counts',unit='optimizer_updates',budget=steps,optimizer_steps=steps,
            estimated_hours=(steps*per_step+(steps/100)*(val_seconds/5))/3600,
            reference='actual batch128 entire smoke training step time plus observed validation every100 updates',
            measured=False,ETH_data=False,limitation='update-count comparison only; dataset and validation policy differ from official paper setup; excludes unmeasured long-run I/O'))
    splits=read_csv(OUT/'data_summary.csv')
    hotel=next(r for r in splits if r['fold']=='biwi_hotel' and r['split']=='train' and r['protocol'].startswith('HTraj'))
    for epochs in [30,100]:
        steps=math.ceil(int(hotel['target_windows'])/128)*epochs
        budget.append(dict(scope='HOTEL held-recording fold transferred estimate',unit='full_passes',budget=epochs,optimizer_steps=steps,
            estimated_hours=steps*per_step/3600,reference='ZARA2 fold measured batch128 step cost transferred to HOTEL manifest target count',
            measured=False,ETH_data=False,limitation='HOTEL training/validation runtime not measured; excludes HOTEL validation/checkpoint overhead; context distribution differs'))
    csv_new(OUT/'training_budget_estimates.csv',budget)
    finalize()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--finalize-only',action='store_true',help='Complete decision/checkpoint inventory from already frozen CSVs without rerunning metrics')
    args=parser.parse_args()
    finalize() if args.finalize_only else main()
