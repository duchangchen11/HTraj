"""Recompute diagnostics from immutable Stage7.4 validation samples only."""
import collections
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.agentformer_innovation.metrics import history_features, motion_metrics, candidate_metrics, social_features, ols_velocity
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
from stage7p5_prepare import digest

OUT=ROOT/'results/tables/stage7p5'


def readcsv(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f: return list(csv.DictReader(f))


def summarize(rows, keys=None):
    if not rows: return dict(target_windows=0,persons=0,recordings=0)
    if keys is None:
        keys=sorted({k for r in rows for k,v in r.items() if isinstance(v,(float,int,np.floating))}-{ 'frame','person_id','context_people'})
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['sequence'],r['person_id'])].append(r)
    result=dict(target_windows=len(rows),persons=len(groups),recordings=len({r['sequence'] for r in rows}),window={},person_equal={},valid_counts={})
    for k in keys:
        a=[r.get(k) for r in rows if r.get(k) is not None]
        if not a: continue
        result['window'][k]=float(np.mean(a));result['valid_counts'][k]=len(a)
        b=[[r.get(k) for r in rr if r.get(k) is not None] for rr in groups.values()]
        b=[float(np.mean(v)) for v in b if v]
        result['person_equal'][k]=float(np.mean(b))
    return result


def clustered_interval(rows,key,seed=7502):
    grouped=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        if r.get(key) is not None:grouped[r['sequence']][r['person_id']].append(r[key])
    rng=np.random.default_rng(seed);means=[]
    for _ in range(200):
        sampled=[]
        for actors in grouped.values():
            a=list(actors.values())
            sampled.extend(a[j] for j in rng.integers(len(a),size=len(a)))
        means.append(np.mean([np.mean(v) for v in sampled]))
    return np.quantile(means,[.025,.975]).tolist() if means else None


def scale_regression(rows):
    """Person-equal least squares. Coefficients descriptive, no iid window tests."""
    sources=sorted({r['sequence'] for r in rows});actors=collections.Counter((r['sequence'],r['person_id']) for r in rows)
    x=np.array([[1,np.log(max(r['GT_path_m'],.5)),*[(r['speed_group']==g)*1. for g in ('low','normal','fast')],*[(r['sequence']==s)*1. for s in sources[1:]]] for r in rows])
    weights=np.array([1/actors[(r['sequence'],r['person_id'])] for r in rows])
    result={}
    for key in ('ADE_oracle_ADE','all_heads_ADE','head0_ADE'):
        y=np.log(np.array([r[key] for r in rows])+.01)
        coef=np.linalg.lstsq(x*np.sqrt(weights)[:,None],y*np.sqrt(weights),rcond=None)[0]
        result[key]=dict(names=['intercept','log_GT_path_floor0.5','low_vs_stationary','normal_vs_stationary','fast_vs_stationary']+sources[1:],coefficients=coef.tolist(),rank=int(np.linalg.matrix_rank(x)),
            interpretation='Descriptive scale adjustment; future path is outcome-scale covariate, not an input/group. Sparse high-speed actors and exposed data limit conclusions')
    return result


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'configs/stage7p5/protocol.json').read_text())
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Unfrozen worktree')
    input_checks=[];all_targets={};curve_rows=[];step_rows=[];head_stats=[]
    official=ROOT/'data/raw/agentformer_official'/protocol['official_commit']
    for index in protocol['prediction_indices']:
        assert digest(ROOT/index['path'])==index['sha256']
        fold=Path(index['path']).name.split('_')[0];targets=[];heads=[]
        curves=collections.defaultdict(list)
        raw_cache={}
        for w,row in enumerate(readcsv(ROOT/index['path'])):
            file=ROOT/row['path'];assert digest(file)==row['sha256']
            input_checks.append(dict(path=row['path'],sha256=row['sha256'],verified=True))
            seq=row['sequence'];frame=int(row['frame'])
            if not seq.endswith('_val'):raise ValueError('Non-validation sequence forbidden')
            if seq not in raw_cache:
                file_raw=official/fold/(seq+'.txt')
                raw_cache[seq]=np.loadtxt(file_raw,usecols=(0,1,13,15))
            with np.load(file,allow_pickle=False) as a:
                x,y,p,ids=[a[k].astype(float) for k in ('history','truth','samples','person_ids')]
            assert x.shape==(len(ids),8,2) and y.shape==(len(ids),12,2) and p.shape==(len(ids),20,12,2)
            for i,identity in enumerate(ids):
                key=dict(fold=fold,sequence=seq,frame=frame,person_id=int(identity))
                hist=history_features(x[i]);social=social_features(raw_cache[seq],frame,int(identity))
                ade,fde,ac,fc,cm=candidate_metrics(x[i],y[i],p[i]);ai=int(ade.argmin());fi=int(fde.argmin())
                per=[motion_metrics(x[i],y[i],p[i,h]) for h in range(20)]
                target={**key,**hist,**social,**cm,'context_people':len(ids),
                    'GT_SHA256':hashlib.sha256(y[i].astype('float32').tobytes()).hexdigest(),
                    'history_SHA256':hashlib.sha256(x[i].astype('float32').tobytes()).hexdigest(),
                    'ADE_winner':ai,'FDE_winner':fi,'GT_speed_mps':per[0]['GT_speed_mps'],'GT_path_m':per[0]['GT_path_m'],'GT_net_m':per[0]['GT_net_m']}
                for name,metrics in [('head0',per[0]),('ADE_oracle',per[ai]),('FDE_oracle',per[fi])]:
                    target.update({name+'_'+k:v for k,v in metrics.items()})
                for k in per[0]:
                    vals=[r[k] for r in per if r[k] is not None]
                    target['all_heads_'+k]=float(np.mean(vals)) if vals else None
                for n in (2,4,8):
                    v=ols_velocity(x[i],n);cv=x[i,-1]+np.arange(1,13)[:,None]*.4*v
                    met=motion_metrics(x[i],y[i],cv)
                    target.update({f'CV{n}_{k}':met[k] for k in ('ADE','FDE','speed_bias_mps','abs_speed_bias_mps')})
                for schedule in protocol['subsets']:
                    order=schedule['heads'];sid=schedule['order_id']
                    for k in protocol['K']:
                        ma,mf=float(ade[order[:k]].min()),float(fde[order[:k]].min())
                        curves[sid,k].append({**key,'ADE':ma,'FDE':mf,'speed_group':hist['speed_group']})
                        if sid==0:target.update({f'prefix_K{k}_ADE':ma,f'prefix_K{k}_FDE':mf})
                for h,m in enumerate(per):
                    heads.append({**key,'speed_group':hist['speed_group'],'density_group':social['density_group'],'head':h,'ADE_credit':float(ac[h]),'FDE_credit':float(fc[h]),**m})
                for t in range(12):
                    er=np.linalg.norm(p[i,:,t]-y[i,t],axis=-1)
                    step_rows.append({**key,'step':t+1,'seconds':(t+1)*.4,'speed_group':hist['speed_group'],
                        'ADE_oracle_error_m':float(er[ai]),'FDE_oracle_error_m':float(er[fi]),'head0_error_m':float(er[0]),'all_heads_error_m':float(er.mean())})
                targets.append(target)
            if w%100==0:print(f'{fold}: {w} windows processed',flush=True)
        csv_new(OUT/f'{fold}_targets.csv.gz',targets)
        csv_new(OUT/f'{fold}_heads.csv.gz',heads)
        keys=['ADE','FDE','speed_bias_mps','abs_speed_bias_mps','predicted_speed_mps','ADE_credit','FDE_credit']
        for h in range(20):
            hh=[r for r in heads if r['head']==h]
            for field in ('all','sequence','speed_group'):
                labels=['all'] if field=='all' else sorted({r[field] for r in hh})
                for label in labels:
                    rr=hh if field=='all' else [r for r in hh if r[field]==label]
                    head_stats.append(dict(fold=fold,head=h,group_field=field,group=label,**summarize(rr,keys)))
        for (sid,k),rows in curves.items():
            for group_field in ('all','sequence','speed_group'):
                labels=['all'] if group_field=='all' else sorted({r[group_field] for r in rows})
                for label in labels:
                    rr=rows if group_field=='all' else [r for r in rows if r[group_field]==label]
                    s=summarize(rr,['ADE','FDE'])
                    curve_rows.append(dict(fold=fold,order_id=sid,K=k,group_field=group_field,group=label,
                        target_windows=s['target_windows'],persons=s['persons'],window_ADE=s['window']['ADE'],window_FDE=s['window']['FDE'],person_ADE=s['person_equal']['ADE'],person_FDE=s['person_equal']['FDE']))
        all_targets[fold]=targets
    csv_new(OUT/'timestep_errors.csv.gz',step_rows)
    csv_new(OUT/'K_subset_curves.csv.gz',curve_rows)
    json_new(OUT/'head_contributions.json',head_stats)
    summaries={};intervals={};regressions={}
    for fold,rows in all_targets.items():
        summaries[fold]={};intervals[fold]={}
        for field in ('all','sequence','speed_group','turn_group','density_group'):
            labels=['all'] if field=='all' else sorted({r[field] for r in rows})
            summaries[fold][field]={}
            for label in labels:
                rr=rows if field=='all' else [r for r in rows if r[field]==label]
                summaries[fold][field][label]=summarize(rr)
                if field in ('speed_group','all'):
                    intervals[fold][label]={key:clustered_interval(rr,key) for key in ('ADE_oracle_speed_bias_mps','all_heads_speed_bias_mps','ADE_oracle_ADE_over_GT_path_floor0.5','ADE_oracle_abs_heading_error_rad','ADE_oracle_turn_magnitude_bias_rad')}
        regressions[fold]=scale_regression(rows)
    json_new(OUT/'group_statistics.json',summaries)
    json_new(OUT/'person_cluster_intervals.json',intervals)
    json_new(OUT/'scale_adjusted_regressions.json',regressions)
    a,b=all_targets.values();ka={(r['sequence'],r['frame'],r['person_id']):r for r in a};kb={(r['sequence'],r['frame'],r['person_id']):r for r in b};common=ka.keys()&kb.keys()
    assert all(ka[k]['GT_SHA256']==kb[k]['GT_SHA256'] and ka[k]['history_SHA256']==kb[k]['history_SHA256'] for k in common)
    json_new(OUT/'cohort_overlap.json',dict(shared_target_windows=len(common),all_shared_history_GT_equal=True,
        source_counts={f:dict(collections.Counter(r['sequence'] for r in rr)) for f,rr in all_targets.items()},
        unique_source_persons={f:len({(r['sequence'],r['person_id']) for r in rr}) for f,rr in all_targets.items()},
        fast_persons={f:sorted({(r['sequence'],r['person_id']) for r in rr if r['speed_group']=='fast'}) for f,rr in all_targets.items()},independent_folds=False))
    csv_new(OUT/'verified_prediction_inputs.csv.gz',input_checks)
    json_new(OUT/'execution.json',dict(source_commit=source_commit,protocol_SHA256=digest(ROOT/'configs/stage7p5/protocol.json'),
        official_commit=protocol['official_commit'],duration_seconds=time.perf_counter()-start,
        prediction_windows=len(input_checks),targets={f:len(r) for f,r in all_targets.items()},new_training=False,new_inference=False,
        test_errors_read=False,python=sys.version,numpy=np.__version__))
    print(json.dumps(dict(seconds=time.perf_counter()-start,targets={f:len(r) for f,r in all_targets.items()})),flush=True)


if __name__=='__main__': main()
