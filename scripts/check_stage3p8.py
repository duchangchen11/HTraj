"""Independent numerical delivery checks, no model training or result edits."""
import csv,json,math,subprocess
from pathlib import Path
from hashlib import sha256 as digest
from collections import defaultdict
import xml.etree.ElementTree as ET

ROOT=Path('results/tables/stage3p8')
def sha(path):return digest(Path(path).read_bytes()).hexdigest()
def read(name):return list(csv.DictReader((ROOT/name).open()))
def js(path):return json.loads(Path(path).read_text())


def main():
    p=js('configs/stage3p8/protocol.json'); protected=js('configs/stage3p8/protected_artifacts.json')
    for f in protected['files']:assert sha(f['path'])==f['sha256'],f['path']
    assert len(p['folds'])==5 and p['models']==['CV-Last2','CV-OLS4','CV-OLS8','CV-Median']
    assert not p['new_neural_training'] and not p['ETH_evaluated'] and not p['learned_gate']
    assert not Path('experiments/stage3p8').exists(), 'No checkpoint/training directory expected'
    for fold in p['folds']:
        assert sha(fold['manifest'])==fold['manifest_sha256']
        m=js(fold['manifest']);train,val=m['splits']['train'],m['splits']['validation']
        assert not {w['recording'] for w in train}&{w['recording'] for w in val}
        assert all(w['scene']!='ETH' and w['end_frame']-w['start_frame']==190 for w in train+val)
    for info in p['source_files'].values():assert info['scene']!='ETH' and sha(Path(p['raw_root'])/info['file'])==info['sha256']
    e=js(ROOT/'evaluation_provenance.json')
    assert e['protocol_sha256']==sha('configs/stage3p8/protocol.json') and not e['new_neural_training'] and not e['model_future_input'] and not e['group_future_input']
    subprocess.run(['git','merge-base','--is-ancestor',e['source_commit'],'HEAD'],check=True)
    for f,h in e['source_sha256'].items():assert sha(f)==h,f
    hist=read('history_features.csv');long=read('target_errors_long.csv');pair=read('paired_target_results.csv')
    assert len(hist)==len(pair)==3582 and len(long)==19970
    ids=lambda r:(r['recording'],r['sample_id'],int(r['target_id']))
    keys={ids(r) for r in hist};assert len(keys)==3582 and {ids(r) for r in pair}==keys
    assert all(float(r['history_frames'])==float(r['contiguous_history_frames'])==8 for r in hist)
    by=defaultdict(list);individual=defaultdict(set)
    for r in long:
        assert math.isfinite(float(r['ADE_m'])) and math.isfinite(float(r['FDE_m']))
        by[r['recording'],r['model']].append(r);individual[r['model']].add(ids(r))
    assert all(individual[n]==keys for n in p['models'])
    metrics=read('recording_metrics.csv'); assert len(metrics)==35
    observed={}
    for r in metrics:
        if r['status']=='not_available':assert not r['ADE_m'] and not r['FDE_m'] and not r['targets'];continue
        rows=by[r['recording'],r['model']];assert len(rows)==int(r['targets'])
        for metric in ('ADE_m','FDE_m'):
            mean=sum(float(x[metric]) for x in rows)/len(rows)
            assert abs(mean-float(r[metric]))<1e-6
        observed[r['recording'],r['model']]=r
    recs=[f['recording'] for f in p['folds']]
    for r in read('aggregate_metrics.csv'):
        for metric in ('ADE_m','FDE_m'):
            if r['weighting']=='recording_macro':value=sum(float(observed[rec,r['model']][metric]) for rec in recs)/5
            elif r['weighting']=='target_window_weighted':value=sum(float(x[metric]) for rec in recs for x in by[rec,r['model']])/3582
            else:
                person=read('per_person_paired_results.csv')
                value=sum(sum(float(x[metric]) for x in person if x['recording']==rec and x['model']==r['model'])/len([x for x in person if x['recording']==rec and x['model']==r['model']]) for rec in recs)/5
            assert abs(value-float(r[metric]))<1e-9
    assert len(read('group_paired_comparison.csv'))==560
    assert all(r['p_value']=='' for r in read('feature_error_associations.csv'))
    times=read('runtime.csv');assert len(times)==20 and all(int(r['timed_passes'])==20 and float(r['forward_seconds_median'])>0 for r in times)
    assert len(read('reference_coverage.csv'))==8
    for ref in p['references']:
        s=ref['selected'];assert sha(s['checkpoint'])==s['checkpoint_sha256'] and sha(s['summary'])==s['summary_sha256']
    d=js(ROOT/'research_decision.json');assert not d['new_training'] and not d['ETH_evaluated'] and not d['analytic_prior_candidates'] and d['stop_adding_estimators']
    assert all(m['raw_ADE_improved_recordings']==m['recording_passes']==1 for m in d['models'])
    tests={}
    for name in ('pre_evaluation_tests.xml','unit_tests.xml'):
        suites=ET.parse(ROOT/name).getroot();assert all(int(s.get(k,0))==0 for s in suites for k in ('errors','failures'))
        tests[name]=sum(int(s.get('tests',0)) for s in suites);assert tests[name]>=201
    for name in ('01_MOTION_RELIABILITY_AUDIT.md','02_KINEMATIC_BASELINE_RESULTS.md','03_CROSS_RECORDING_ANALYSIS.md','04_FAILURE_CASE_ANALYSIS.md','05_RESEARCH_DIRECTION_DECISION.md','STAGE3P8_REPORT.md'):
        assert (Path('reports/stage3p8')/name).stat().st_size>500
    result=dict(status='passed',protected_artifacts=len(protected['files']),old_checkpoints=protected['old_checkpoints'],recordings=5,
        target_windows=3582,recording_person_units=177,analytic_evaluations=20,frozen_neural_evaluations=8,neural_coverage_incomplete=True,
        new_training=False,new_checkpoints=0,ETH_evaluated=False,tests=tests,auto_push=False)
    out=ROOT/'DELIVERY_AUDIT.json'
    if out.exists():assert js(out)==result
    else:out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
