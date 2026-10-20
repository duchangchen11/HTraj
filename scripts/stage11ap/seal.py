"""Verify actual evidence, protected history and stage-only delivery before commit."""
from common import *
import xml.etree.ElementTree as ET

def main():
    history=json.loads((OUT/'history_after.json').read_text())
    assert history['passed'] and not history['failures']
    suite=ET.parse(OUT/'full_unit_tests.xml').getroot().find('testsuite')
    assert int(suite.get('tests'))==532 and int(suite.get('errors'))==0 and int(suite.get('failures'))==0
    roots=['configs/stage11ap/','scripts/stage11ap/','reports/stage11ap/','results/tables/stage11ap/']
    paths=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    paths+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(any(p.startswith(r) for r in roots) for p in paths), paths
    assert subprocess.run(['git','diff','--check'],cwd=ROOT).returncode==0
    for path in ['ynet_additional_files.zip','author_resources/ynet_additional_files/data/SDD/train_trajnet.pkl','author_resources/ynet_additional_files/pretrained_models/sdd_trajnet_weights.pt']:
        assert subprocess.run(['git','check-ignore','-q',str(CACHE/path)],cwd=ROOT).returncode==0
    resources=[]
    for row in csv.DictReader((OUT/'extracted_resource_SHA256.csv').open()):
        p=CACHE/'author_resources'/row['path']
        assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256']
        resources.append(row)
    source=json.loads((OUT/'frozen_source_manifest.json').read_text())
    for r in source['files']:assert sha(CACHE/'official'/r['path'])==r['sha256']
    alignment=json.loads((OUT/'alignment_cases.json').read_text())
    assert alignment['real_cases']==6 and alignment['test_cases']==0
    for r in alignment['cases']:assert sha(ROOT/r['figure'])==r['figure_SHA']
    pipe=json.loads((OUT/'full_pipeline_admission.json').read_text())
    assert pipe['passed'] and pipe['strict_load'] and pipe['checkpoint_keys']==776
    assert pipe['optimizer_updates']==0 and not pipe['metrics_computed'] and pipe['weights_unchanged']
    isolation=json.loads((OUT/'real_train_future_isolation.json').read_text())
    assert isolation['passed'] and isolation['max_abs_prediction_difference']==0
    protocol=json.loads((CONF/'protocol_candidate.json').read_text())
    assert protocol['status']=='NOT_READY' and not protocol['formal_training_authorized']
    jnew(OUT/'final_checks.json',dict(passed=True,full_unit_tests=532,historical_files=history['files'],history_unchanged=True,
         official_source_unchanged=True,author_resources_SHA_verified=len(resources),alignment_real_train_cases=6,
         actual_train_forward_passed=True,strict_load=True,predictor_checkpoint_keys=776,GT_isolation_passed=True,
         no_YNet_train_or_test_metrics=True,no_YNet_optimizer_updates=True,legacy_toy_optimizer_unit_tests_run=True,protocol_status='NOT_READY',
         decision='NO-GO formal training/protocol freeze and novel algorithm development; real-input engineering PASS',
         research_claims='No new model or claimed performance gain',push_authorization='Current attachment explicitly requests pushing this new branch'))
    files=sorted(p for r in roots for p in (ROOT/r).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in files:
        assert p.stat().st_size<12<<20, p
        assert p.suffix.lower() not in {'.pkl','.pickle','.pt','.pth','.ckpt','.npz','.zip','.jpg','.pdf'},p
    code=[p for p in files if str(p.relative_to(ROOT)).startswith(('scripts/stage11ap/','configs/stage11ap/'))]
    cnew(OUT/'code_config_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in code])
    files.append(OUT/'code_config_SHA256.csv')
    cnew(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)])
    print('Verified Stage11A+:',len(files),'delivery files; 532 tests;',history['files'],'historical files unchanged')
if __name__=='__main__':main()
