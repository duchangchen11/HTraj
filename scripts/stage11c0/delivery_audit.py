"""Final real-artifact, immutable-history and Git-scope checks, no model inference."""
from .common import *
import re
import xml.etree.ElementTree as ET
import torch
def main():
    cpu_only();hist=json.loads((OUT/'history_after.json').read_text());assert hist['passed'] and not hist['changed']
    frozen=json.loads((OUT/'protocol_frozen.json').read_text())
    assert sha(CONF/'experiment_protocol.json')==frozen['protocol_SHA'] and sha(CONF/'generation_config.json')==frozen['generation_config_SHA']
    for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256']
    assert sha(CP)==CK and sha(PP)==PSHA
    for name in ['cache_audit.json','precheck_results.json','baseline_regression.json','full_generation.json','fixed_generation.json','fixed_repeat_reverse.json','score_integrity.json']:
        assert json.loads((OUT/name).read_text())['passed'],name
    failed_files=list((CACHE/'S1_failed_attempt1/tensors').glob('*.pt'));assert len(failed_files)==1
    failed=torch.load(failed_files[0],map_location='cpu',weights_only=True)
    resumed=torch.load(CACHE/'S1/tensors'/failed_files[0].name,map_location='cpu',weights_only=True)
    assert failed['sample_id']==resumed['sample_id'] and set(failed['tensors'])==set(resumed['tensors'])
    assert all(torch.equal(failed['tensors'][k],resumed['tensors'][k]) for k in failed['tensors'])
    budget=json.loads((OUT/'gpu_budget.json').read_text());assert abs(parent_seconds()-budget['actual_GPU_process_seconds'])<1e-9 and parent_seconds()<6600
    suites=list(ET.parse(OUT/'full_tests.xml').iter('testsuite'));tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert tests['tests']>=672 and tests['failures']==tests['errors']==0 and tests['skipped']==1
    skips=[dict(test=t.get('classname')+'.'+t.get('name'),reason=t.find('skipped').get('message')) for t in ET.parse(OUT/'full_tests.xml').iter('testcase') if t.find('skipped') is not None]
    figures=json.loads((OUT/'figure_manifest.json').read_text());assert len(figures)==10
    for r in figures:
        for ext,key in [('png','PNG_SHA'),('svg','SVG_SHA')]:assert sha(FIG/(r['name']+'.'+ext))==r[key]
        ET.parse(FIG/(r['name']+'.svg'))
    manifest=json.loads((OUT/'execution_manifest.json').read_text())
    for r in manifest['source_files']:assert sha(ROOT/r['path'])==r['sha256']
    allowed=['configs/stage11c0/','scripts/stage11c0/','reports/stage11c0/','results/tables/stage11c0/','results/figures/stage11c0/']
    names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
    names+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(any(n.startswith(a) for a in allowed) for n in names)
    for p in [CP,PP,CACHE/'S1_full.pt',CACHE/'S1_fixed.pt']:
        subprocess.run(['git','check-ignore','-q',str(p.relative_to(ROOT))],cwd=ROOT,check=True)
    files=sorted(p for a in allowed for p in (ROOT/a).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in {'delivery_audit.json','delivery_SHA256.csv'})
    assert not any(p.suffix.lower() in {'.pt','.pth','.pkl','.npy','.npz','.pdf','.key','.pem'} for p in files)
    large=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in files if p.stat().st_size>=10*1024*1024];assert not large,large
    secret=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bghp_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{50,}')
    for p in files:
        if p.suffix in {'.py','.json','.csv','.md','.txt'}:assert not secret.search(p.read_text()),p
    audit=dict(passed=True,old_historical_files_unchanged=hist['files'],original_checkpoint_and_prediction_SHA_unchanged=True,
        original_semantic_caches_and_sampling_code_unchanged=True,all_frozen_numerical_code_SHA_unchanged=True,
        actual_new_full_groups=3,actual_new_fixed_goal_groups=3,targets_per_group=998,model_parameters_unchanged=True,
        all_Goal_and_Path_shapes_verified=True,GT_only_after_complete_generation=True,training_steps=0,test_access=False,
        GPU_process_seconds=parent_seconds(),failed_attempt_preserved_and_charged=True,failed_first_prediction_equals_resumed_bitwise=True,tests=tests,skips=skips,
        actual_figures=10,sensitive_scan='PASS',large_model_data_cache_ignored=True,stage_scope='only five new Stage11C-0 directories',
        source_code_SHA=sha(Path(__file__)),source_commit=commit())
    (OUT/'delivery_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');files.append(OUT/'delivery_audit.json')
    with (OUT/'delivery_SHA256.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','sha256','bytes'],lineterminator='\n');w.writeheader()
        for p in sorted(files):w.writerow(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    print('DELIVERY PASS',len(files),'files; tests',tests,'history',hist['files'],'unchanged',flush=True)
if __name__=='__main__':main()
