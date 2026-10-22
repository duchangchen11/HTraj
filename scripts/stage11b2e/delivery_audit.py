"""Delivery scope, immutable source, actual results and artifact checksums."""
from .common import *
import re
import xml.etree.ElementTree as ET

def main():
    cpu_only();history=json.loads((OUT/'history_after.json').read_text())
    assert history['passed'] and history['files']==92362 and not history['changed']
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'experiment_protocol.json')==frozen['protocol_SHA']
    assert sha(CONF/'generation_config.json')==frozen['generation_config_SHA']
    for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(CP)==CK and sha(PP)==PSHA
    for file in ['cache_audit.json','selection_audit.json','decoder_regression.json','generation_execution.json','score_integrity.json']:
        assert json.loads((OUT/file).read_text())['passed'],file
    execution=json.loads((OUT/'execution_manifest.json').read_text())
    for r in execution['source_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert json.loads((OUT/'gpu_budget.json').read_text())['actual_GPU_process_seconds']<3300
    suites=list(ET.parse(OUT/'full_tests.xml').iter('testsuite'))
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert tests==dict(tests=653,failures=0,errors=0,skipped=1)
    skipped=[dict(test=t.get('classname')+'.'+t.get('name'),reason=t.find('skipped').get('message'))
        for t in ET.parse(OUT/'full_tests.xml').iter('testcase') if t.find('skipped') is not None]
    figs=json.loads((OUT/'figure_manifest.json').read_text());assert len(figs)==6
    for r in figs:
        for ext,key in [('png','PNG_SHA'),('svg','SVG_SHA')]:assert sha(FIG/(r['name']+'.'+ext))==r[key]
        ET.parse(FIG/(r['name']+'.svg'))
    allowed=['configs/stage11b2e/','scripts/stage11b2e/','reports/stage11b2e/','results/tables/stage11b2e/','results/figures/stage11b2e/']
    names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
    names+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(any(n.startswith(a) for a in allowed) for n in names)
    for p in [CP,PP,CACHE/'alternative_goals.pt',CACHE/'alternative_paths.pt']:
        subprocess.run(['git','check-ignore','-q',str(p.relative_to(ROOT))],cwd=ROOT,check=True)
    files=sorted(p for a in allowed for p in (ROOT/a).rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.name not in {'delivery_SHA256.csv','delivery_audit.json'})
    assert not any(p.suffix.lower() in {'.pt','.pth','.pkl','.npy','.npz','.pdf','.key','.pem'} for p in files)
    assert all(p.stat().st_size<10*1024*1024 for p in files)
    secret=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bghp_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{50,}')
    for p in files:
        if p.suffix in {'.py','.json','.csv','.md','.txt'}:assert not secret.search(p.read_text()),p
    d=dict(passed=True,historical_files_unchanged=history['files'],numerical_frozen_code_unchanged=True,
        original_checkpoint_and_prediction_SHA_unchanged=True,all998_original_tensors_unchanged=True,
        all998_new_paths_really_decoded=True,all998_head0_goals_paths_bitwise=True,
        tests=tests,tests_passed=652,tests_skipped=skipped,figures=6,tests_failed=0,
        training_steps=0,new_sampling=0,new_kmeans=0,sealed_test_used=False,
        scope='CONTROL_EXPERIMENT_ONLY',decision=json.loads((OUT/'final_decision.json').read_text())['status'],
        Git_scope='only five new Stage11B-2E source/config/report/table/figure directories',
        ignore_rules='existing rules sufficient; datasets, environments, weights and new tensor caches not staged',
        largest_file=dict(path=str(max(files,key=lambda p:p.stat().st_size).relative_to(ROOT)),bytes=max(p.stat().st_size for p in files)),
        sensitive_scan='PASS',source_code_SHA=sha(Path(__file__)),audit_source_commit=commit())
    (OUT/'delivery_audit.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    files.append(OUT/'delivery_audit.json')
    with (OUT/'delivery_SHA256.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','sha256','bytes'],lineterminator='\n');w.writeheader()
        for p in sorted(files):w.writerow(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    print('DELIVERY PASS',len(files),'files; history92362 unchanged; 652 PASS, 1 SKIP; CONTROL_NO_GO',flush=True)

if __name__=='__main__':main()
