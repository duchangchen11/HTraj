"""Post-run descriptive CPU analysis; no prediction or training entry."""
from .common import *
from .data import input_records,truths
import numpy as np,torch
import xml.etree.ElementTree as ET

def main():
    assert not torch.cuda.is_initialized()
    decision=json.loads((OUT/'final_decision.json').read_text());e=decision['best_epoch']
    v=json.loads((OUT/f'validation_epoch{e:03d}.json').read_text())
    pairs=list(csv.DictReader((OUT/f'paired_epoch{e:03d}.csv').open()))
    old=list(csv.DictReader((ROOT/'results/tables/stage11b1/baseline_paired_errors.csv').open()))
    baselines={model:{r['sample_id']:r for r in old if r['model']==model} for model in ['CV-Last2','OLS4','OLS8']}
    combined=[]
    for r in pairs:
        row={k:r[k] for k in ['sample_id','scene','person_id','frame_id','history_SHA','GT_SHA','checkpoint_SHA']}
        row.update({f'YNet_{k}':float(r[k]) for k in ['ADE1','FDE1','minADE20','minFDE20','GoalFDE1','minGoalFDE20']})
        for model,index in baselines.items():
            b=index[r['sample_id']]
            assert b['history_SHA']==r['history_SHA'] and b['GT_SHA']==r['GT_SHA']
            row.update({f'{model}_{k}':float(b[k]) for k in ['ADE1','FDE1']})
        combined.append(row)
    cnew(OUT/'best_paired_baseline_comparison.csv',combined)
    delta=[]
    for model in baselines:
        for scene in ['ALL_WINDOWS']+sorted({r['scene'] for r in pairs}):
            rows=[r for r in combined if scene=='ALL_WINDOWS' or r['scene']==scene]
            item=dict(baseline=model,scene=scene,targets=len(rows))
            for metric in ['ADE1','FDE1']:
                base=np.array([r[f'{model}_{metric}'] for r in rows]);pred=np.array([r[f'YNet_{metric}'] for r in rows])
                item.update({f'{metric}_YNet_minus_baseline':float((pred-base).mean()),
                    f'{metric}_relative_improvement':float((base.mean()-pred.mean())/base.mean()),
                    f'{metric}_targets_lower_error':int((pred<base).sum()),
                    f'{metric}_targets_higher_error':int((pred>base).sum())})
            delta.append(item)
    cnew(OUT/'paired_baseline_deltas.csv',delta)
    pack=torch.load(ROOT/v['prediction_manifest']['path'],map_location='cpu',weights_only=True)
    a=json.loads((OUT/'data_audit.json').read_text());hist={r['sample_id']:r['history'] for r in input_records('validation')};gt=truths('validation')
    rows=[]
    for i,sid in enumerate(pack['sample_ids']):
        scene=next(r['scene'] for r in pairs if r['sample_id']==sid);shape=a['reference'][scene]
        p=pack['paths'][:,i].numpy();g=pack['goals'][:,i].numpy();last=hist[sid][-1]
        step=np.linalg.norm(np.diff(np.concatenate([np.broadcast_to(last,(20,1,2)),p],axis=1),axis=1),axis=-1)
        outside=(p[:,:,0]<0)|(p[:,:,0]>=shape['width'])|(p[:,:,1]<0)|(p[:,:,1]>=shape['height'])
        rows.append(dict(sample_id=sid,scene=scene,out_of_reference_predicted_points=int(outside.sum()),
            total_predicted_points=240,max_predicted_step_pixels=float(step.max()),
            max_GT_step_pixels=float(np.linalg.norm(np.diff(np.vstack([last,gt[sid]]),axis=0),axis=-1).max()),
            max_goal_path_gap_pixels=float(np.linalg.norm(g-p[:,-1],axis=-1).max())))
    cnew(OUT/'prediction_geometry_audit.csv',rows)
    epochs=list(csv.DictReader((OUT/'epochs.csv').open()));prior=json.loads((OUT/'validation_epoch025.json').read_text())
    result=dict(best_epoch=e,best_at_training_boundary=e==30,unit='original reference pixels',
        loss_epoch1=float(epochs[0]['train_loss']),loss_epoch25=float(epochs[24]['train_loss']),loss_epoch30=float(epochs[29]['train_loss']),
        minADE20_epoch25_to30=v['window']['minADE20']-prior['window']['minADE20'],
        ADE1_epoch25_to30=v['window']['ADE1']-prior['window']['ADE1'],FDE1_epoch25_to30=v['window']['FDE1']-prior['window']['FDE1'],
        goal_coverage=v['goal_coverage'],geometry=dict(outside_points=sum(r['out_of_reference_predicted_points'] for r in rows),
            total_predicted_points=len(rows)*240,max_step_pixels=max(r['max_predicted_step_pixels'] for r in rows)),
        peak_allocated_bytes=max(int(r['peak_allocated_bytes']) for r in epochs),peak_reserved_bytes=max(int(r['peak_reserved_bytes']) for r in epochs),
        train_epoch_seconds_mean=float(np.mean([float(r['train_seconds']) for r in epochs])),
        K1_advantage_not_uniform_across_recordings=True,no_statistical_independence_claim=True,
        convergence='Budget truncated; endpoint oracle still improves while K1 regresses fromepoch25 to30',
        baseline_engineering_GO=True,automatic_extension=False,semantic_innovation_GO=False,
        GPU_used_by_this_analysis=False)
    jnew(OUT/'descriptive_analysis.json',result)
    # A preliminary CPU-oriented regression invokes CUDA RNG checkpoint capture.
    # Its pytest elapsed time is real; the whole parent was not timed. Keep this
    # limitation explicit and charge an additional300-second allowance, not a
    # fabricated measurement. No extra model execution is performed.
    budget=json.loads((OUT/'gpu_budget.json').read_text());suite=ET.parse(OUT/'preflight_unit_tests.xml').getroot().find('testsuite')
    budget.update(preliminary_regression_pytest_seconds=float(suite.attrib['time']),
        preliminary_regression_parent_wall_measured=False,
        preliminary_regression_budget_allowance_seconds=300,
        budget_accounted_seconds=budget['GPU_complete_worker_wall_seconds']+300,
        budget_allowance_is_not_actual_GPU_time=True,
        timing_limit='Preliminary24-test CPU regression captures CUDA RNG but had no parent wall timer;300seconds conservatively reserved. Main GPU workers and final full regression are parent-timed.',
        all_startup_loading_checkpoint_and_GPU_preflight_included=False)
    assert budget['budget_accounted_seconds']<27000
    replace_json(OUT/'gpu_budget.json',budget)
    assert not torch.cuda.is_initialized()
    print('Post-run paired comparisons, geometry, convergence and budget accounting saved; CPU only')

if __name__=='__main__':main()
