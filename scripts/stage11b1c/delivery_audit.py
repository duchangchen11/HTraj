"""Post-run CPU-only evidence audit; never constructs or updates a model."""
from scripts.stage11b1c.common import *
from scripts.stage11b1c.data import truths, manifest
from scripts.stage11b0.runtime import state_digest
from collections import defaultdict, Counter
import numpy as np
import torch

def audit():
    assert not torch.cuda.is_initialized()
    run=json.loads((OUT/'training_execution.json').read_text())
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert run['status']=='COMPLETED_30_EPOCHS' and run['completed_epochs']==30
    assert run['completed_validation_epochs']==[1,5,10,15,20,25,30]
    for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    cm=json.loads((OUT/'checkpoint_manifest.json').read_text())
    assert len(cm['checkpoints'])==31 and cm['last']['epoch']==30
    checkpoints=[]
    for c in cm['checkpoints']:
        assert sha(ROOT/c['path'])==c['sha256'] and (ROOT/c['path']).stat().st_size==c['bytes']
        checkpoints.append(dict(epoch=c['epoch'],path=c['path'],sha256=c['sha256'],verified=True))
    initial=torch.load(ROOT/cm['checkpoints'][0]['path'],map_location='cpu',weights_only=True)
    assert not initial['optimizer']['state']
    core=lambda s:{k:v for k,v in s.items() if not k.startswith('semantic_segmentation.')}
    seg=lambda s:{k.removeprefix('semantic_segmentation.'):v for k,v in s.items() if k.startswith('semantic_segmentation.')}
    assert state_digest(core(initial['model']))==run['initial_predictor_state_SHA']
    assert state_digest(seg(initial['model']))==run['initial_segmentation_state_SHA']
    states=[]
    for role in ['best','last']:
        c=cm[role];s=torch.load(ROOT/c['path'],map_location='cpu',weights_only=True)
        assert s['progress']['epoch']==c['epoch'] and s['progress']['global_step']==1882*c['epoch']
        assert s['progress']['epoch_boundary'] and s['config']['train_config_SHA']==frozen['train_config_SHA']
        assert state_digest(seg(s['model']))==run['initial_segmentation_state_SHA']
        changed=[k for k,v in core(s['model']).items() if not torch.equal(v,initial['model'][k])]
        assert changed and all(torch.isfinite(v).all() for v in s['model'].values())
        assert all(float(p['step'])==1882*c['epoch'] for p in s['optimizer']['state'].values())
        assert set(s['rng'])=={'python','numpy','torch','cuda'} and len(s['rng']['cuda'])==1
        states.append(dict(role=role,epoch=c['epoch'],checkpoint_SHA=c['sha256'],segmentation_bitwise_unchanged=True,
            updated_predictor_tensors=len(changed),Adam_state_tensors=len(s['optimizer']['state']),RNG_complete=True))
        del s
    del initial
    epochs=list(csv.DictReader((OUT/'epochs.csv').open()));assert len(epochs)==30
    visits=defaultdict(list);steps=Counter();gradient_values=[]
    with gzip.open(OUT/'training_steps.csv.gz','rt') as f:
        for number,r in enumerate(csv.DictReader(f),1):
            assert int(r['global_step'])==number and float(r['learning_rate'])==.0001
            e=int(r['epoch']);ids=r['sample_ids'].split('|');assert len(ids)==int(r['targets'])
            visits[e].extend(ids);steps[e]+=1;gradient_values.append(float(r['gradient_norm']))
            assert np.isfinite([float(r[k]) for k in ['loss','goal_loss','path_loss','gradient_norm','seconds']]).all()
            assert int(r['nonzero_gradient_elements'])>0
    train_ids={r['sample_id'] for r in manifest('train')};coverage=[]
    for e,r in enumerate(epochs,1):
        assert int(r['epoch'])==e and len(visits[e])==len(set(visits[e]))==7496
        assert set(visits[e])==train_ids and steps[e]==1882
        assert hashlib.sha256('\n'.join(visits[e]).encode()).hexdigest()==r['visit_order_SHA']
        assert int(r['recordings'])==len(set(r['scene_order'].split('|')))==25
        assert r['segmentation_state_SHA']==run['initial_segmentation_state_SHA']
        coverage.append(dict(epoch=e,optimizer_steps=steps[e],visits=7496,unique=7496,manifest_set_identical=True,visit_order_SHA=r['visit_order_SHA']))
    assert sum(steps.values())==run['optimizer_updates']==56460
    assert sum(map(len,visits.values()))==224880
    # Predeclared audit tolerance:1e-8 raw pixels, float64 independent NumPy scorer.
    tolerance=1e-8;gt=truths('validation');identity={r['sample_id']:r for r in manifest('validation')}
    numerical=[];prediction_checks=[]
    for e in run['completed_validation_epochs']:
        v=json.loads((OUT/f'validation_epoch{e:03d}.json').read_text());m=v['prediction_manifest']
        assert v['status']=='COMPLETE' and v['targets_scored']==998 and m['all_predictions_completed_before_GT_scoring']
        assert sha(ROOT/m['path'])==m['sha256'] and sha(OUT/f'prediction_traces_epoch{e:03d}.csv.gz')==m['trace_SHA']
        p=torch.load(ROOT/m['path'],map_location='cpu',weights_only=True)
        assert set(p['sample_ids'])==set(identity) and len(p['sample_ids'])==998
        assert p['paths'].shape==(20,998,12,2) and p['goals'].shape==(20,998,2)
        assert p['K']==20 and p['TTST_proposals']==10000 and p['checkpoint_SHA']==v['checkpoint']['sha256']
        y=np.stack([gt[s] for s in p['sample_ids']]);paths=p['paths'].numpy().astype('float64');goals=p['goals'].numpy().astype('float64')
        assert np.isfinite(paths).all() and np.isfinite(goals).all()
        d=np.sqrt(np.sum((paths-y[None])**2,axis=-1));ge=np.sqrt(np.sum((goals-y[None,:,-1])**2,axis=-1))
        values=dict(ADE1=d[0].mean(-1),FDE1=d[0,:,-1],minADE20=d.mean(-1).min(0),minFDE20=d[:,:,-1].min(0),
            GoalFDE1=ge[0],minGoalFDE20=ge.min(0))
        rows={r['sample_id']:r for r in csv.DictReader((OUT/f'paired_epoch{e:03d}.csv').open())}
        for k,a in values.items():
            actual=np.array([float(rows[s][k]) for s in p['sample_ids']]);error=float(np.max(np.abs(a-actual)))
            assert error<=tolerance,(e,k,error)
            assert abs(float(a.mean())-v['window'][k])<=tolerance
            numerical.append(dict(epoch=e,metric=k,max_absolute_pixel_difference=error,tolerance=tolerance,passed=True))
        for sid,r in rows.items():
            assert r['history_SHA']==identity[sid]['history_SHA'] and r['GT_SHA']==identity[sid]['GT_SHA']
        checks=list(csv.DictReader((OUT/f'reproducibility_epoch{e:03d}.csv').open()))
        assert len(checks)==6 and all(r['bitwise']=='True' for r in checks)
        prediction_checks.append(dict(epoch=e,targets=998,K=20,proposals=10000,prediction_SHA=m['sha256'],
            checkpoint_SHA=p['checkpoint_SHA'],runtime_reproducibility_gates=6,finite=True,identity_GT_history_matched=True))
    assert cm['best_epoch']==min(run['completed_validation_epochs'],key=lambda e:(json.loads((OUT/f'validation_epoch{e:03d}.json').read_text())['window']['minADE20'],e))
    assert not torch.cuda.is_initialized()
    cnew(OUT/'delivery_checkpoint_verification.csv',checkpoints)
    cnew(OUT/'delivery_epoch_coverage.csv',coverage)
    cnew(OUT/'independent_numpy_scoring.csv',numerical)
    jnew(OUT/'delivery_execution_audit.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        completed_epochs=30,total_optimizer_updates=56460,total_training_target_visits=224880,
        checkpoint_count=31,checkpoint_states=states,predictions=prediction_checks,
        independent_scorer='NumPy float64 Euclidean distances; six main metrics, all998 targets, all7 validation epochs',
        prespecified_tolerance_pixels=tolerance,source_freeze_unchanged=True,
        gradient_norm_min=float(min(gradient_values)),gradient_norm_max=float(max(gradient_values)),
        gradient_norm_median=float(np.median(gradient_values)),test_evaluated=False,
        audit_script_SHA=sha(Path(__file__))))
    print('CPU delivery audit PASS:31 checkpoints,224880 target visits,6986 target-epoch predictions,6 paired metrics')

if __name__=='__main__':audit()
