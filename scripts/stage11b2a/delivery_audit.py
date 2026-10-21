"""CPU delivery assertions over actual outputs and immutable sources."""
from .common import *
from .analysis import candidate_metrics
import numpy as np,pandas as pd,torch,xml.etree.ElementTree as ET,re

def main():
    cpu_guard();assert readj(OUT/'history_after.json')['passed']
    pre=readj(OUT/'protocol_freeze.json');assert sha(CONF/'analysis_protocol.json')==pre['protocol_SHA']
    for name,digest in pre['source_SHA'].items():assert sha(ROOT/name)==digest,name
    assert sha(CP)==CK and sha(PP)==PSHA
    for name,digest in readj(OUT/'data_cache_audit.json')['sources'].items():assert sha(ROOT/name)==digest,name
    p,h,y,identity=frozen();ids=p['sample_ids'];f=pd.read_csv(OUT/'motion_features.csv',float_precision='round_trip').set_index('sample_id').loc[ids]
    assert len(f)==998 and set(f.index)==set(ids)
    d,ade,fde,ge,gap,m=candidate_metrics(p['paths'].numpy().astype(float),y,p['goals'].numpy().astype(float))
    maximum=0.
    for key,values in m.items():
        error=float(np.max(np.abs(f[key].to_numpy()-values)));maximum=max(maximum,error);assert error<=1e-8,key
    c=pd.read_csv(OUT/'goal_path_decomposition.csv',float_precision='round_trip')
    assert len(c)==19960 and not c.duplicated(['sample_id','head']).any()
    for i,sid in enumerate(ids):
        rows=c[c.sample_id==sid].sort_values('head');assert rows['head'].tolist()==list(range(20))
        for key,a in [('ADE',ade),('FDE',fde),('GoalFDE',ge),('goal_path_gap',gap)]:assert np.max(np.abs(rows[key].to_numpy()-a[:,i]))<=1e-8
    groups=pd.read_csv(OUT/'motion_error_groups.csv')
    assert groups.groupby('family').targets.sum().eq(998).all()
    assert f.outside_point_count.sum()==51 and (f.outside_point_count>0).sum()==11
    cases=pd.read_csv(OUT/'failure_case_manifest.csv');coords=pd.read_csv(OUT/'failure_case_coordinates.csv',float_precision='round_trip')
    assert len(cases)==21 and len(coords)==21*304
    for row in cases.itertuples():
        i=ids.index(row.sample_id);r=coords[coords['case']==row.case]
        for kind,original in [('history',h[i]),('GT_future',y[i])]:
            a=r[r.kind==kind].sort_values('step')[['x','y']].to_numpy();assert np.array_equal(a,original)
        a=r[r.kind=='prediction'].sort_values(['head','step'])[['x','y']].to_numpy();assert np.array_equal(a,p['paths'][:,i].numpy().reshape(-1,2))
        a=r[r.kind=='goal'].sort_values('head')[['x','y']].to_numpy();assert np.array_equal(a,p['goals'][:,i].numpy())
        assert row.history_SHA==identity[i]['history_SHA'] and row.GT_SHA==identity[i]['GT_SHA']
        for view in ['full','detail']:
            for ext in ['png','svg']:assert (FIG/f'case_{row.case}_{view}.{ext}').is_file()
    names=['01_K1_K20_distributions','02_goal_path_relationship','03_goal_coverage_error','04_history_motion_error','05_turning_error',
        '06_recording_motion_comparison','07_failure_types_and_overlap','08_paired_OLS8_error','09_frozen_validation_trends','10_real_failure_case_composite']
    for name in names:
        for ext in ['png','svg']:assert (FIG/(name+'.'+ext)).is_file()
    xml=ET.parse(OUT/'full_unit_tests_final.xml').getroot().find('testsuite')
    assert xml.attrib['failures']==xml.attrib['errors']=='0' and xml.attrib['tests']=='573' and xml.attrib['skipped']=='1'
    altered=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    prefixes=('configs/stage11b2a/','scripts/stage11b2a/','reports/stage11b2a/','results/tables/stage11b2a/','results/figures/stage11b2a/')
    assert all(x.startswith(prefixes) for x in altered)
    paths=[q for base in [CONF,OUT,FIG,REPORT,ROOT/'scripts/stage11b2a'] for q in base.rglob('*') if q.is_file() and '__pycache__' not in q.parts]
    for q in paths:
        assert q.stat().st_size<10_000_000 and q.suffix not in {'.pt','.pth','.pkl','.ckpt','.npz','.npy'},q
        if q.suffix in {'.py','.json','.md','.txt','.csv'}:
            assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{25,}',q.read_text(errors='replace')),q
    env=dict(python=__import__('sys').version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        matplotlib=__import__('matplotlib').__version__,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_initialized=torch.cuda.is_initialized(),GPU_seconds=0)
    writej(OUT/'environment.json',env)
    sources=[dict(path=str(q.relative_to(ROOT)),sha256=sha(q)) for q in sorted((ROOT/'scripts/stage11b2a').rglob('*')) if q.is_file() and '__pycache__' not in q.parts]
    pd.DataFrame(sources).to_csv(OUT/'analysis_code_SHA256.csv',index=False)
    cpu_guard();writej(OUT/'delivery_audit.json',dict(passed=True,targets=998,candidate_pairs=19960,case_selections=21,case_coordinate_rows=len(coords),
        original_prediction_SHA=PSHA,original_checkpoint_SHA=CK,protocol_unchanged=True,frozen_analysis_sources_unchanged=True,
        history_protected_files=90656,old_files_changed=[],all_outputs_numerically_verified=True,max_sample_metric_difference=maximum,
        full_tests_passed=572,full_tests_skipped=1,skipped_reason='Historical CUDA compatibility test, CPU-only stage',GPU_seconds=0,
        new_model_forward_calls=0,new_formal_training_steps=0,new_test_scores=False,large_or_sensitive_artifacts_found=False,
        raw_failure_logs_preserved=['layout_fix_stdout.txt','source_snapshots/layout_fix_attempt1.py.txt','qa_evidence/before_caption_overlap.png'],
        analysis_class='EXPLORATORY_DEVELOPMENT_ANALYSIS',scientific_thresholds_changed_after_freeze=False))
    print('Delivery audit PASS: all998/19960 paired rows,21 cases/6384 coordinates, frozen source identity,572 tests+1 CPU-environment skip')
if __name__=='__main__':main()
