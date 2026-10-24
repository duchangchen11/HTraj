"""Final CPU-only integrity, resource and repository-scope acceptance."""
from .common import *
import xml.etree.ElementTree as ET

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    frozen=verify_frozen()
    history=json.loads((OUT/'history_after.json').read_text());assert history['passed']
    assert sha(CONF/'protected_files.csv.gz')==history['protection_manifest_SHA']
    diag=json.loads((OUT/'diagnostic_epoch1_execution.json').read_text())
    assert diag['epoch']==1 and diag['target_visits']==diag['unique_targets']==7496
    assert diag['optimizer_updates']==1882 and diag['formal_Stage11C1_epochs']==0
    assert diag['semantic_zero_nonzero_count']==0 and diag['strict_checkpoint_restore_model_optimizer_RNG']
    checkpoints=json.loads((OUT/'diagnostic_checkpoint_manifest.json').read_text())
    for key in ['initial','epoch1']:assert sha(ROOT/checkpoints[key]['path'])==checkpoints[key]['sha256']
    rows=list(csv.DictReader((OUT/'tensor_artifact_manifest.csv').open()))
    assert len(rows)==90
    for row in rows:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    eq=json.loads((OUT/'instrumentation_equivalence.json').read_text());assert eq['passed']
    ledger=json.loads((OUT/'gpu_budget.json').read_text());assert ledger['passed']
    assert ledger['Stage11C1RUsed']<2250 and ledger['CombinedUsed']<=18000
    decision=json.loads((OUT/'final_decision.json').read_text())
    assert decision['FORMAL_TRAINING']=='NOT_AUTHORIZED' and decision['measured_new_metrics'] is False
    figures=json.loads((OUT/'figure_manifest.json').read_text())['available'];assert len(figures)==6
    for row in figures:
        assert sha(ROOT/row['source'])==row['source_SHA']
        for ext,key in [('png','PNG_SHA'),('svg','SVG_SHA')]:assert sha(FIG/(row['name']+'.'+ext))==row[key]
    junit=ET.parse(OUT/'full_tests.xml').getroot().find('testsuite')
    assert junit.get('failures')=='0' and junit.get('errors')=='0'
    skips=[dict(test=c.get('name'),reason=c.find('skipped').get('message'))
        for c in junit.findall('testcase') if c.find('skipped') is not None]
    changed=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    roots=['configs','scripts','reports','results/tables','results/figures']
    assert all(any(n.startswith(x+'/stage11c1r/') for x in roots) for n in changed)
    files=[p for x in roots for p in (ROOT/x/'stage11c1r').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts
        and p.name not in {'delivery_audit.json','delivery_audit_stdout.txt'}]
    bad=[str(p.relative_to(ROOT)) for p in files if p.stat().st_size>10*1024*1024 or p.suffix in {'.pt','.pth','.pkl','.npz','.pem','.key'}]
    assert not bad,bad
    for key in ['initial','epoch1']:
        assert subprocess.run(['git','check-ignore','-q',checkpoints[key]['path']],cwd=ROOT).returncode==0
    jnew(OUT/'delivery_audit.json',dict(passed=True,execution_source_commit='f259323d3af3aeacd635192deae1e33fe43f133f',
        audit_parent_commit=commit(),critical_frozen_source_and_inputs_verified=True,history_files=history['files'],
        history_verified=True,diagnostic_checkpoint_SHA_verified=True,tensor_caches_verified=90,figures_verified=6,
        formal_epochs=0,diagnostic_epochs=1,new_validation_metrics=False,test_loaded=False,
        pytest=dict(tests=int(junit.get('tests')),passed=int(junit.get('tests'))-len(skips),skips=skips,failures=0,errors=0),
        GPU_ledger=ledger,repository_scope_verified=True,forbidden_submission_files=bad,
        metadata_scope_note='measurement_scope_audit.json; six optional original-Predictor RNG fields unmeasured, not passed',
        figure_QA='All six PNGs inspected; observed iteration axis and budget legend corrected; no invented accuracy curves',
        self_artifacts_excluded_from_recursive_manifest=['delivery_audit.json','delivery_audit_stdout.txt'],
        files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)]))
    print('DELIVERY_AUDIT_PASS',len(files),'files',history['files'],'historical files',flush=True)
if __name__=='__main__':main()
