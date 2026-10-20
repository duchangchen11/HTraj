from .common import *
import xml.etree.ElementTree as ET
def main():
    history=json.loads((OUT/'history_after.json').read_text());assert history['passed'] and not history['failures']
    test=ET.parse(OUT/'full_unit_tests.xml').getroot().find('testsuite')
    assert int(test.get('tests'))==544 and int(test.get('failures'))==int(test.get('errors'))==0
    run=json.loads((OUT/'training_execution.json').read_text())
    assert run['status']=='ENGINEERING_TRAINING_PASS' and run['complete_pass']
    assert run['full_pass_target_visits']==run['full_pass_unique_targets']==7496 and run['full_pass_recordings']==25
    assert run['full_pass_steps']==1882 and run['total_optimizer_steps']==1887
    assert run['checkpoint_restore_and_next_step_exact'] and run['frozen_segmentation_weights_and_BN_unchanged']
    assert not run['predictor_pretrained_weights_loaded'] and not run['failures']
    budget=json.loads((OUT/'final_gpu_budget.json').read_text());assert budget['budget_pass'] and budget['conservative_total_charge_seconds']<3600
    targets={r['sample_id'] for r in csv.DictReader((OUT/'train_target_manifest.csv').open())}
    visited=[r['sample_id'] for r in csv.DictReader((OUT/'actual_target_visits.csv').open())]
    assert len(visited)==len(set(visited))==7496 and set(visited)==targets
    critical=['common.py','runtime.py','data_audit.py','training_adapter.py','run_admission.py']
    critical_rows=[]
    for name in critical:
        p='scripts/stage11b0/'+name
        frozen=subprocess.check_output(['git','show',run['started_code_commit']+':'+p],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest()==sha(ROOT/p),p
        critical_rows.append(dict(path=p,sha256=sha(ROOT/p)))
    checkpoints=[]
    for c in run['checkpoints']:
        path=ROOT/c['path'];assert path.stat().st_size==c['bytes'] and sha(path)==c['sha256']
        assert subprocess.run(['git','check-ignore','-q',str(path)],cwd=ROOT).returncode==0
        checkpoints.append(c)
    source=json.loads((OUT/'source_and_environment.json').read_text())
    for v in source['official_files']:assert sha(CACHE/'official'/v['path'])==v['sha256']
    allowed=['reports/stage11b0/','configs/stage11b0/','scripts/stage11b0/','results/tables/stage11b0/']
    paths=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    paths+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(any(p.startswith(a) for a in allowed) for p in paths),paths
    jnew(OUT/'final_checks.json',dict(passed=True,unit_tests=544,old_files=history['files'],old_files_unchanged=True,
        source_commit=run['started_code_commit'],official_commit=OFFICIAL,training_critical_source_unchanged_since_freeze=critical_rows,
        complete_pass_targets=7496,complete_pass_steps=1882,total_real_updates=1887,checkpoint_restore_exact=True,
        engineering_GO=True,formal_training_approved=False,full_protocol_ready=False,innovation_GO=False,
        GPU_conservative_seconds=budget['conservative_total_charge_seconds'],no_real_val_or_test_scores=True,
        semantic_class_mapping='SEMANTIC_CLASS_MAPPING_UNVERIFIED',segmentation_exposure='SEGMENTATION_EXPOSURE_UNVERIFIED',
        author_segmentation_train_only_claim_read_in_supplement=True,old_results_not_reinterpreted=True))
    cnew(OUT/'checkpoint_index.csv',[dict(path=c['path'],sha256=c['sha256'],bytes=c['bytes'],save_and_hash_seconds=c['save_and_hash_seconds']) for c in checkpoints])
    files=sorted(p for a in allowed for p in (ROOT/a).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in files:
        assert p.stat().st_size<12<<20,p
        assert p.suffix not in {'.pkl','.pt','.pth','.pdf','.zip','.npz'},p
    cnew(OUT/'code_config_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in files if str(p.relative_to(ROOT)).startswith(('scripts/stage11b0/','configs/stage11b0/'))])
    files.append(OUT/'code_config_SHA256.csv')
    cnew(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)])
    print('Stage11B-0 sealed; engineering GO, formal protocol/training NOT_APPROVED;',len(files),'delivery files')
if __name__=='__main__':main()
