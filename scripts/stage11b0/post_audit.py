"""CPU post-run update, augmentation-count, timing and budget audit. No new steps."""
from .common import *
from .runtime import build_model,original_modules,state_digest
from .data_audit import load_train
from scripts.stage11ap.adapters import augment_data_adapter
import torch,numpy as np,pandas as pd,math,gc

def main():
    run=json.loads((OUT/'training_execution.json').read_text())
    steps=pd.read_csv(OUT/'training_timing.csv');scenes=pd.read_csv(OUT/'scene_timing.csv')
    measured=steps[steps.phase=='full_pass']
    aug_rows=[];oob=[]
    df=load_train();all_meta_ids=set();torch.set_num_threads(1)
    for scene,g in df.groupby('sceneId',sort=True):
        augmented,images=augment_data_adapter(g.copy(),RES/'data/SDD/train',original_modules().preprocess)
        assert len(augmented)==8*len(g) and augmented.sceneId.nunique()==8
        assert (augmented.groupby('metaId').size()==20).all()
        n=len(g)//20
        for v in augmented.sceneId.unique():
            a=augmented[augmented.sceneId==v];h,w=images[v].shape[:2]
            raw=a[['x','y']].to_numpy();outside=(raw[:,0]<0)|(raw[:,0]>=w)|(raw[:,1]<0)|(raw[:,1]>=h)
            oob.append(dict(scene=v,raw_outside_points=int(outside.sum()),raw_points=len(raw)))
        aug_rows.append(dict(scene=scene,original_targets=n,augmented_scenes=8,augmented_targets=8*n,
                             target_batch4_steps=8*math.ceil(n/4),target_batch8_steps=8*math.ceil(n/8)))
        del augmented,images;gc.collect()
    cnew(OUT/'augmentation_visit_counts.csv',aug_rows);cnew(OUT/'augmentation_outside_points.csv',oob)
    jnew(OUT/'augmentation_audit.json',dict(CPU_only=True,raw_targets=7496,augmented_targets=sum(r['augmented_targets'] for r in aug_rows),
        raw_recordings=25,augmented_scene_variants=200,metadata_lengths20=True,image_and_trajectory_follow_original_transforms=True,
        target_batch4_steps=sum(r['target_batch4_steps'] for r in aug_rows),target_batch8_steps=sum(r['target_batch8_steps'] for r in aug_rows),
        GPU_augmented_training_executed=False,
        coordinate_convention='Author uses W/2,H/2 continuous image-edge-centered transforms. At integer pixel-center indexing,90deg/flip can differ by1 pixel from W-1/H-1; original behavior preserved, not called exact pixel-center registration',
        transformed_outside_points_preserved=True))
    if run.get('complete_pass'):
        net,cfg=build_model(17,'cpu')
        last=torch.load(CACHE/'checkpoints/full_pass_last.pt',map_location='cpu',weights_only=True)
        updates=[]
        for group in ['encoder','goal_decoder','traj_decoder','semantic_segmentation']:
            before={k:v for k,v in net.state_dict().items() if k.startswith(group+'.')}
            after={k:v for k,v in last['model'].items() if k.startswith(group+'.')}
            assert before.keys()==after.keys()
            changed=sum(not torch.equal(v,after[k]) for k,v in before.items())
            updates.append(dict(group=group,tensors=len(before),changed_tensors=changed,before_SHA=state_digest(before),after_SHA=state_digest(after)))
            assert (changed==0 if group=='semantic_segmentation' else changed>0)
        cnew(OUT/'parameter_update_audit.csv',updates)
        jnew(OUT/'parameter_update_audit.json',dict(passed=True,segmentation_weights_and_buffers_exact=True,
             encoder_updated=True,goal_decoder_updated=True,path_decoder_updated=True,checkpoint_sha=sha(CACHE/'checkpoints/full_pass_last.pt'),
             legacy_predictor_weights_loaded=False,CPU_only=True))
    seconds=run.get('full_pass_wall_seconds')
    base=dict(full_pass_completed=bool(run.get('complete_pass')),full_pass_targets=run.get('full_pass_target_visits'),
              full_pass_steps=run.get('full_pass_steps'),observed_full_pass_seconds=seconds,
              all_train_steps_wall_sum=float(measured.wall_seconds.sum()),all_train_GPU_event_span_seconds=float(measured.GPU_event_seconds.sum()),
              average_step_wall_seconds=float(measured.wall_seconds.mean()),median_step_seconds=float(measured.wall_seconds.median()),
              p90_step_seconds=float(measured.wall_seconds.quantile(.9)),p99_step_seconds=float(measured.wall_seconds.quantile(.99)),
              maximum_step_seconds=float(measured.wall_seconds.max()),mean_excluding_first5_steps=float(measured.wall_seconds.iloc[5:].mean()),
              total_loading_seconds=float(scenes.loading_seconds.sum()),total_segmentation_seconds=float(scenes.segmentation_seconds.sum()),
              allocated_peak_bytes=run.get('peak_allocated_bytes'),reserved_peak_bytes=run.get('peak_reserved_bytes'),
              checkpoint_bytes=run.get('last_checkpoint',{}).get('bytes'),checkpoint_save_seconds=run.get('last_checkpoint',{}).get('save_and_hash_seconds'),
              augmented_target_visits=59968,augmented_steps=sum(r['target_batch4_steps'] for r in aug_rows),
              official_capped_epoch='At most26 of200 augmented recording variants in this development pool; count depends on scene permutation/target counts. Not a complete epoch.',
              official_original_pool='Original author30 recordings ->240 augmented variants; separate pool from this train-only25 recording subset',
              confidence='One unaugmented full pass at seed17; no learning/validation convergence evidence, no direct augmented GPU measure',
              validation_TTST_evaluation_cost='UNMEASURED; excluded from training estimates',
              checkpoint_storage_policy_suggestion='Keep initial/last/best/periodic; original weights never overwritten',
              exclusion='No author inference timings used for training extrapolation')
    estimates=[]
    if seconds is not None:
        for epochs in [30,50,100,300]:
            estimates.append(dict(epochs=epochs,unaugmented_training_only_hours=epochs*seconds/3600,
                unaugmented_planning_range_hours=[epochs*seconds*.8/3600,epochs*seconds*1.3/3600],
                eightfold_augmented_linear_training_only_hours=epochs*seconds*8/3600,
                eightfold_augmented_planning_range_hours=[epochs*seconds*8*.8/3600,epochs*seconds*8*1.5/3600],
                note='Planning ranges are assumptions, not confidence intervals. Rotated padding/cache/thermal effects and later validation increase uncertainty.'))
    base['estimates']=estimates
    # Exact sample/step range of a26scene cap on200variants (8 identical target counts per recording).
    visits=sorted([r['original_targets'] for r in aug_rows for _ in range(8)])
    stepcounts=sorted([math.ceil(r['original_targets']/4) for r in aug_rows for _ in range(8)])
    base['26_scene_cap_target_range']=[sum(visits[:26]),sum(visits[-26:])]
    base['26_scene_cap_step_range']=[sum(stepcounts[:26]),sum(stepcounts[-26:])]
    base['26_scene_cap_expected_target_visits_under_uniform_scene_shuffle']=26/200*59968
    jnew(OUT/'budget_estimate.json',base)
    print('Augmentation counts verified on CPU, no extra GPU training. Actual full-pass seconds:',seconds)
if __name__=='__main__':main()
