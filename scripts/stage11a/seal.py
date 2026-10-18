"""Validate source/data bytes, stage-only scope and all delivered evidence."""
from common import *
import xml.etree.ElementTree as ET

def main():
    history=json.loads((OUT/'history_after.json').read_text())
    assert history['passed'] and not history['failures']
    suite=ET.parse(OUT/'final_unit_tests.xml').getroot().find('testsuite')
    assert int(suite.get('tests'))==516 and int(suite.get('failures'))==0 and int(suite.get('errors'))==0
    base='521c53a0d5f4ec2a76685c40ab519fc2ea3c8299'
    allowed=['scripts/stage11a/','configs/stage11a/','reports/stage11a/','results/tables/stage11a/']
    paths=subprocess.check_output(['git','diff','--name-only',base],cwd=ROOT,text=True).splitlines()
    paths+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(any(p.startswith(a) for a in allowed) for p in paths)
    code=json.loads((OUT/'source_download.json').read_text())
    for r in code['files']:
        assert r['verified'] and sha(CACHE/'official'/r['path'])==r['sha256']
    license=json.loads((OUT/'license_audit.json').read_text())
    assert sha(CACHE/'official/LICENSE')==license['sha256']
    sample=json.loads((OUT/'sdd_sample_download.json').read_text())
    for r in sample['rows']:
        if 'path' in r:assert sha(ROOT/r['path'])==r['sha256']
    figures=json.loads((OUT/'alignment_results.json').read_text())
    assert figures['real_cases']==6 and not figures['semantic_maps_available']
    for r in figures['cases']:assert sha(ROOT/r['overlay_path'])==r['overlay_SHA']
    draft=json.loads((CONF/'protocol_draft.json').read_text())
    assert draft['status']=='DRAFT_NOT_FROZEN' and not draft['formal_training_authorized']
    assert draft['splits']['test'] is None and draft['splits']['validation'] is None
    blob_rows=[dict(path='experiments/stage11a/cache/official/'+r['path'],sha256=r['sha256'],bytes=r['bytes']) for r in code['files']]
    for r in sample['rows']:
        if 'path' in r:blob_rows.append(dict(path=r['path'],sha256=r['sha256'],bytes=r['bytes']))
    for r in figures['cases']:blob_rows.append(dict(path=r['overlay_path'],sha256=r['overlay_SHA'],bytes=(ROOT/r['overlay_path']).stat().st_size))
    cnew(OUT/'source_data_figure_SHA256.csv',blob_rows)
    jnew(OUT/'final_decision.json',dict(decision='NO-GO formal Stage11B training and new algorithm development',
        conditional_baseline='Y-Net once original resources, split and isolated compatibility verified',
        backup_reference='PECNet; different inputs/evaluator, not independent semantic ablation',
        strongest_question='Q2 semantic reliability impact on goals and constrained multimodal coverage, not yet a new mechanism',
        data_complete=False,official_semantics_available=False,precise_registration_proven=False,
        full_YNet_pipeline_passed=False,CPU_synthetic_core_passed=True,
        GPU_synthetic_core_forward_passed=True,formal_training=False,new_test_metrics=False,
        optimizer_updates=0,proposal_models_implemented=0,protocol_frozen=False,
        formal_training_GPU_hours=None,next_engineering_budget_ceiling_GPU_hours=1,
        actual_core_GPU_probe='gpu_probe.json; excludes segmentation/backward/TTST/CWS, not a full training estimate',
        unresolved_recent_full_texts=['CINet2025','McTrans2026','GDS-TGTF2026'],
        current_base_commit=commit(),official_model_commit=code['commit'],
        full_unit_tests=516,history_protected_files=history['files']))
    files=[]
    for a in allowed:
        files.extend(p for p in (ROOT/a).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in files:
        assert p.stat().st_size<12*1024*1024,str(p)
        assert p.suffix not in ('.jpg','.pt','.pth','.pkl','.pickle','.npz','.pdf')
    assert subprocess.run(['git','check-ignore','-q',str(CACHE/'sdd_samples/bookstore/video0/reference.jpg')],cwd=ROOT).returncode==0
    jnew(OUT/'delivery_checks.json',dict(passed=True,unit_tests=516,history_files=history['files'],
        old_files_unchanged=True,stage11a_only=True,vendor_code_unchanged=True,
        raw_data_cache_ignored=True,no_checkpoints_or_secrets_for_git=True,
        largest_delivery_file_bytes=max(p.stat().st_size for p in files),
        all_failures_retained=['baseline_probe.json','paper_access.json','additional_paper_access.json','network_access.json','recent_preview_access.json','openalex_discovery_stderr.txt'],
        source_and_data_SHA_verified=True,model_training=False,test_evaluation=False))
    code_files=[p for p in files if str(p.relative_to(ROOT)).startswith(('scripts/stage11a/','configs/stage11a/'))]
    cnew(OUT/'final_code_config_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(code_files)])
    files.extend([OUT/'delivery_checks.json',OUT/'final_code_config_SHA256.csv'])
    cnew(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))])
    print('Stage11A sealed: tests516, history',history['files'],'files; NO-GO research training')

if __name__=='__main__':main()
