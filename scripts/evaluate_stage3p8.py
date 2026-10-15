"""Execute only the preregistered analytic and frozen-checkpoint evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import json
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.preprocessing import sha256,write_json_new
from src.data.benchmark_audit import BenchmarkDataset,assert_recording_isolation
from src.data.collate import collate_trajectories
from src.models.kinematic_reliability import KinematicPredictor,reliability_features,reliability_groups
from src.models.ped_base import PedBase
from src.models.frozen_ego import CVResidual
from src.models.motion_normalization import HeadingNormalized
from src.evaluation.benchmark_audit import target_errors
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage3p8'); CONF=Path('configs/stage3p8/protocol.json')


def batch_double(b):
    return dict(b,observed=b['observed'].double(),future=b['future'].double())


def verify_reference(ref,fold,p):
    s=ref['selected']; assert sha256(s['summary'])==s['summary_sha256'] and sha256(s['checkpoint'])==s['checkpoint_sha256']
    c=json.loads(Path(s['summary']).with_name('config.json').read_text())
    assert c['manifest_sha256']==fold['manifest_sha256'] and c['source_commit']==s['source_commit']
    protocol_name=c.get('protocol_name',c.get('training',{}).get('protocol_name'))
    assert protocol_name==p['protocol_name']
    for path,digest in c['source_sha256'].items(): assert sha256(path)==digest,path
    # Check that validation recording was excluded from training and selected
    # training indices match frozen data, rather than reuse a leaking fold.
    m=json.loads(Path(fold['manifest']).read_text()); assert_recording_isolation(m)
    assert {w['recording'] for w in m['splits']['validation']}=={ref['recording']}
    if 'pilot_indices_sha256' in c:
        assert sha256(c['protocol']['pilot_indices'])==c['pilot_indices_sha256']
        idx=json.loads(Path(c['protocol']['pilot_indices']).read_text())
        assert idx['manifest']==fold['manifest']
        assert idx['protocols'][p['protocol_name']]['validation']==list(range(len(m['splits']['validation'])))
    else:
        oldfold=next(f for f in c['training']['folds'] if f['recording']==ref['recording'])
        assert oldfold['manifest']==fold['manifest']
    model=PedBase() if s['kind']=='PedBase' else CVResidual() if s['kind']=='CVResidual' else HeadingNormalized(cv_residual=True)
    checkpoint=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']); model.eval()
    return model


@torch.no_grad()
def timed(model,batches,p):
    for _ in range(p['timing']['warmup_passes']):
        for b in batches: model(b['observed'],b['observed_valid'],b['timestamps'])
    values=[]
    for _ in range(p['timing']['repeat_passes']):
        start=time.perf_counter()
        for b in batches: model(b['observed'],b['observed_valid'],b['timestamps'])
        values.append(time.perf_counter()-start)
    return dict(forward_seconds_median=float(np.median(values)),forward_seconds_min=min(values),forward_seconds_max=max(values),
        forward_seconds_mean=float(np.mean(values)),timed_passes=len(values),device='CPU',cpu_threads=1)


@torch.no_grad()
def evaluate(model,batches,recording,name):
    totals={}; individual=[]
    for b in batches:
        prediction=model(b['observed'],b['observed_valid'],b['timestamps'])
        assert torch.isfinite(prediction).all()
        ade,fde,a,f,count=target_errors(prediction,b['future'],b['future_valid'])
        assert torch.equal(a,f) and (count[a]==12).all()
        for g,mask in b['groups'].items():
            target=a&mask; r=totals.setdefault(g,[0.,0.,0]); r[0]+=float(ade[target].sum()); r[1]+=float(fde[target].sum()); r[2]+=int(target.sum())
        for bi,ni in a.nonzero().tolist():
            individual.append(dict(recording=recording,sample_id=b['sample_id'][bi],target_id=int(b['agent_ids'][bi,ni]),
                model=name,ADE_m=float(ade[bi,ni]),FDE_m=float(fde[bi,ni])))
    groups=[dict(recording=recording,model=name,group=g,targets=r[2],ADE_m=r[0]/r[2] if r[2] else None,FDE_m=r[1]/r[2] if r[2] else None) for g,r in totals.items()]
    return groups,individual


def main():
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(): raise RuntimeError('Commit fixed protocol, source and tests first')
    suites=ET.parse(ROOT/'pre_evaluation_tests.xml').getroot(); assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors'))
    p=json.loads(CONF.read_text()); assert not p['new_neural_training'] and not p['ETH_evaluated']
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); torch.set_num_threads(1)
    metrics=[]; groups=[]; errors=[]; history=[]; times=[]; counts=[]; coverage=[]; historical_fit=[]
    for fold in p['folds']:
        assert sha256(fold['manifest'])==fold['manifest_sha256']; manifest=json.loads(Path(fold['manifest']).read_text()); assert_recording_isolation(manifest)
        data=BenchmarkDataset(p['raw_root'],fold['manifest'],'validation',p['protocol_name']); batches=[]; effective=0
        for b in DataLoader(data,batch_size=p['batch_size'],collate_fn=collate_trajectories):
            d=batch_double(b); f=reliability_features(d['observed'],d['observed_valid'],d['timestamps']); gs=reliability_groups(f)
            b['groups']=gs; d['groups']=gs; target=b['future_valid'].all(1)
            assert torch.equal(target,b['future_valid'].any(1)); effective+=int(target.any(-1).sum())
            assert (f['history_frames'][target]==8).all() and (f['contiguous_history_frames'][target]==8).all()
            for bi,ni in target.nonzero().tolist():
                history.append(dict(recording=fold['recording'],sample_id=b['sample_id'][bi],target_id=int(b['agent_ids'][bi,ni]),
                    **{k:float(v[bi,ni]) for k,v in f.items()},**{g:bool(v[bi,ni]) for g,v in gs.items()}))
            # Historical one-step constant-velocity residual, using only x0..x7.
            predicted=d['observed'][:,1:-1]+(d['observed'][:,1:-1]-d['observed'][:,:-2])
            rmse=(predicted-d['observed'][:,2:]).square().sum(-1).mean(1).sqrt()
            for bi,ni in target.nonzero().tolist():
                historical_fit.append(dict(recording=fold['recording'],sample_id=b['sample_id'][bi],target_id=int(b['agent_ids'][bi,ni]),
                    history_last2_one_step_RMS_m=float(rmse[bi,ni]),history_OLS8_in_sample_RMS_m=float(f['linear_fit_RMS_m'][bi,ni]),
                    interpretation='in-sample smoothing vs historical one-step backtest; not true denoised velocity or future test'))
            if target.any(): batches.append((b,d))
        target_count=sum(int(b['future_valid'].all(1).sum()) for b,d in batches)
        counts.append(dict(recording=fold['recording'],manifest_sha256=fold['manifest_sha256'],manifest_windows=len(data),effective_windows=effective,
            target_windows=target_count,unique_pedestrians=len({r['target_id'] for r in history if r['recording']==fold['recording']}),complete8_history=target_count,complete12_future=target_count))
        double=[d for b,d in batches]; single=[b for b,d in batches]
        for name in p['models']:
            model=KinematicPredictor(name).double().eval(); timing=timed(model,double,p)
            gs,es=evaluate(model,double,fold['recording'],name); groups+=gs; errors+=es
            metrics.append(dict(gs[0],status='evaluated',parameters=0,manifest_sha256=fold['manifest_sha256'],source_commit=source,new_training=False))
            times.append(dict(recording=fold['recording'],model=name,target_windows=target_count,
                microseconds_per_target_window=timing['forward_seconds_median']/target_count*1e6,**timing))
        for ref in [r for r in p['references'] if r['recording']==fold['recording']]:
            model=verify_reference(ref,fold,p); gs,es=evaluate(model,single,fold['recording'],ref['label']); groups+=gs; errors+=es
            assert abs(gs[0]['ADE_m']-ref['selected']['historical_best_ADE_m'])<2e-6
            assert abs(gs[0]['FDE_m']-ref['selected']['historical_paired_FDE_m'])<2e-6
            metrics.append(dict(gs[0],status='frozen_checkpoint_evaluated',parameters=sum(x.numel() for x in model.parameters()),
                manifest_sha256=fold['manifest_sha256'],source_commit=ref['selected']['source_commit'],new_training=False))
            coverage.append(dict(recording=fold['recording'],model=ref['label'],status='compatible_checked',checkpoint_sha256=ref['selected']['checkpoint_sha256'],
                checkpoint=ref['selected']['checkpoint'],targets=target_count,source_summary=ref['selected']['summary']))
        for absent in [r for r in p['reference_coverage'] if r['recording']==fold['recording'] and r['status']=='not_available']:
            metrics.append(dict(recording=fold['recording'],model=absent['model'],group='all',targets=None,ADE_m=None,FDE_m=None,status='not_available',reason=absent['reason'],new_training=False))
        print(json.dumps(dict(recording=fold['recording'],targets=target_count,analytic_models=4,new_training=False)),flush=True)
    for name,rows in [('recording_metrics.csv',metrics),('group_results.csv',groups),('target_errors_long.csv',errors),('history_features.csv',history),
        ('runtime.csv',times),('split_counts.csv',counts),('reference_coverage.csv',coverage),('historical_smoothing_diagnostic.csv',historical_fit)]: csv_new(ROOT/name,rows)
    write_json_new(ROOT/'evaluation_provenance.json',dict(source_commit=source,protocol_sha256=sha256(CONF),
        source_sha256={str(x):sha256(x) for x in Path('src').rglob('*.py')},script_sha256=sha256(__file__),
        manifest_hashes={f['recording']:f['manifest_sha256'] for f in p['folds']},source_files=p['source_files'],
        dtype_note=p['analytic_dtype'],target_windows=len(history),analytic_runs=20,frozen_neural_runs=len(coverage),new_neural_training=False,
        ETH_evaluated=False,group_future_input=False,model_future_input=False,independent_confirmation=False))


if __name__=='__main__':main()
