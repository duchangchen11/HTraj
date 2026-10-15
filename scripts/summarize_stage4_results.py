"""Aggregate executed evaluations without selecting weights, heads or scenes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,gzip,statistics,collections,difflib
import numpy as np
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage4')


def main():
    rows=list(csv.DictReader((ROOT/'social_results.csv').open()))
    assert len(rows)==36
    groups=collections.defaultdict(list)
    for r in rows:groups[r['fold'],r['mode']].append(r)
    summary=[]
    for (fold,mode),runs in groups.items():
        assert {int(r['monte_carlo_seed']) for r in runs}=={17,42,2026}
        a=[float(r['ADE_m']) for r in runs];f=[float(r['FDE_m']) for r in runs]
        summary.append(dict(fold=fold,mode=mode,MC_seeds=3,target_windows=int(runs[0]['target_windows']),windows=int(runs[0]['windows']),
            ADE_mean_m=statistics.mean(a),ADE_sample_SD_m=statistics.stdev(a),FDE_mean_m=statistics.mean(f),FDE_sample_SD_m=statistics.stdev(f),
            parameters=int(runs[0]['parameters']),evaluation_seconds_mean=statistics.mean(float(r['evaluation_seconds']) for r in runs)))
    csv_new(ROOT/'social_sampling_summary.csv',summary)
    aggregate=[]
    for mode in sorted({r['mode'] for r in rows}):
        for seed in (17,42,2026):
            runs=[r for r in rows if r['mode']==mode and int(r['monte_carlo_seed'])==seed]
            assert len(runs)==4
            for kind in ('four_benchmark_scene_macro','target_window_weighted'):
                w=[1 if kind=='four_benchmark_scene_macro' else int(r['target_windows']) for r in runs]
                aggregate.append(dict(mode=mode,monte_carlo_seed=seed,aggregation=kind,scene_count=4,
                    target_windows=sum(int(r['target_windows']) for r in runs),ADE_m=sum(float(r['ADE_m'])*v for r,v in zip(runs,w))/sum(w),
                    FDE_m=sum(float(r['FDE_m'])*v for r,v in zip(runs,w))/sum(w)))
    csv_new(ROOT/'social_aggregate_results.csv',aggregate)
    temporal=[]
    for fold,rec in [('hotel','biwi_hotel'),('univ','students001'),('univ','students003'),('zara1','crowds_zara01'),('zara2','crowds_zara02')]:
        a=np.loadtxt(Path('data/external/stage4/assets/social_stgcnn/datasets')/fold/'test'/(rec+'.txt'))
        frames=np.unique(a[:,0]);accepted=0;gaps=0;targets=0;bad_targets=0;max_interval=0
        for start in range(len(frames)-19):
            span=frames[start:start+20];data=a[(a[:,0]>=span[0])&(a[:,0]<=span[-1])]
            ids,count=np.unique(data[:,1],return_counts=True);complete=int((count==20).sum())
            if complete<2:continue
            accepted+=1;targets+=complete;delta=np.diff(span);max_interval=max(max_interval,float(delta.max()))
            if not np.all(delta==10):gaps+=1;bad_targets+=complete
        temporal.append(dict(fold=fold,recording=rec,accepted_windows=accepted,target_windows=targets,
            windows_with_non10_frame_gap=gaps,target_windows_with_non10_frame_gap=bad_targets,max_accepted_frame_interval=max_interval,
            method='source frame+ID occurrence audit matching complete20 and at least2 nodes; no prediction/GT-value grouping'))
    for fold in ('hotel','univ','zara1','zara2'):
        known=next(r for r in rows if r['fold']==fold)
        assert sum(r['accepted_windows'] for r in temporal if r['fold']==fold)==int(known['windows'])
        assert sum(r['target_windows'] for r in temporal if r['fold']==fold)==int(known['target_windows'])
    csv_new(ROOT/'social_temporal_support_audit.csv',temporal)
    old=Path('data/external/stage3p5/social_stgcnn/ebd57aaf34d84763825d05cf9d4eff738d8c96bb');new=Path('third_party/stage4/social_stgcnn')
    files=[]
    for name in ('model.py','utils.py','test.py'):
        a,b=old/name,new/name
        diff=list(difflib.unified_diff(a.read_text().splitlines(),b.read_text().splitlines(),fromfile='CVPR2020/'+name,tofile='officialHEAD/'+name))
        files.append(dict(file=name,original_SHA256=sha256(a),frozen_HEAD_SHA256=sha256(b),byte_identical=a.read_bytes()==b.read_bytes(),unified_diff=diff))
    write_json_new(ROOT/'source_version_comparison.json',dict(CVPR2020_commit=old.name,frozen_HEAD='333d3a57b4d2705e129b21aefefa09c79b2b9ae1',
        actual_evaluation='frozen maintained HEAD and its supplied weights; not exact original CVPR2020 architecture',files=files))
    # Preserve compact events and raw log fingerprints, excluding progress-bar spam.
    events=[];logs=[]
    for name in ('social_execution.log','eqmotion_execution.log'):
        path=Path('experiments/stage4')/name
        if not path.exists():continue
        logs.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path)))
        for line in path.read_text().replace('\r','\n').splitlines():
            if line.startswith('{"fold"'):events.append(dict(log=str(path),event=json.loads(line)))
    write_json_new(ROOT/'execution_events.json',dict(events=events,logs=logs,training=False,ETH_evaluated=False))
    print(json.dumps(dict(scene_mode_rows=len(summary),aggregate_rows=len(aggregate),temporal_support=temporal)))


if __name__=='__main__':main()
