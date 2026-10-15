"""Validate real equal-budget training, immutable assets, and reviewable delivery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,json,gzip,subprocess,re,xml.etree.ElementTree as ET
import numpy as np
from src.data.preprocessing import sha256,write_json_new

ROOT=Path('results/tables/stage4p1')


def main():
    protection=json.loads(Path('configs/stage4p1/protected_artifacts.json').read_text())
    for entry in protection['files']:assert sha256(entry['path'])==entry['sha256'],entry['path']
    p=json.loads(Path('configs/stage4p1/protocol.json').read_text());assert not p['ETH_evaluated'] and not p['pretrained_initialization']
    manifests={}
    for name,info in p['manifests'].items():
        assert sha256(info['path'])==info['sha256'];m=json.loads(Path(info['path']).read_text());manifests[name]=m
        assert not ({w['recording'] for w in m['splits']['train']}&{w['recording'] for w in m['splits']['validation']})
        assert set(m['source_files'])=={'biwi_hotel','students001','students003','crowds_zara01','crowds_zara02'}
        for rec,source in m['source_files'].items():assert sha256(source['path'])==source['sha256']
        for split,windows in m['splits'].items():
            for w in windows:
                assert len(w['frames'])==20 and all(d==10 for d in np.diff(w['frames']))
                if split=='train':assert w['frames'][-1]<m['train_cutoffs'][w['recording']]['train_end_exclusive']
    assert sha256(p['cache'])==p['cache_sha256']
    A=manifests['A_offline_complete20'];assert len(A['splits']['train'])==1488 and len(A['splits']['validation'])==921
    outputs=[]
    for kind in ('Full','NoSocial-ConsistentNorm'):
        folder=ROOT/kind;run=json.loads((folder/'execution.json').read_text());smoke=json.loads((folder/'smoke.json').read_text())
        rows=list(csv.DictReader((folder/'curve_30.csv').open()));assert len(rows)==30
        assert [int(r['epoch']) for r in rows]==list(range(1,31))
        assert run['parameters']==7563 and run['actual_epochs']==30 and smoke['gate_pass'] and smoke['checkpoint_restore_output_exact']
        assert run['protocol_sha256']==sha256('configs/stage4p1/protocol.json')
        assert run['manifest_sha256']==p['manifests']['A_offline_complete20']['sha256']
        assert not run['ETH_evaluated'] and not run['pretrained_initialization']
        for row in rows:
            assert int(row['train_windows'])==1488 and int(row['train_targets'])==23349 and int(row['optimizer_updates'])==24
            assert int(row['validation_target_windows'])==5833
            for key,value in row.items():
                if key not in ('model',):assert np.isfinite(float(value)),(kind,key,value)
        best=min(rows,key=lambda r:float(r['validation_NLL']));assert run['best_epoch']==int(best['epoch'])
        for ck in run['checkpoints']:assert sha256(ck['path'])==ck['sha256']
        outputs.append(dict(model=kind,best_epoch=run['best_epoch'],training_seconds=run['training_seconds']))
    errors=list(csv.DictReader(gzip.open(ROOT/'paired_target_errors.csv.gz','rt')))
    assert len(errors)==5833*3*2
    assert len({(r['label'],r['mode'],r['window_id'],r['agent_id']) for r in errors})==len(errors)
    for row in errors:
        assert abs(float(row['NoSocial_minus_Full_ADE'])-(float(row['NoSocial_ADE'])-float(row['Full_ADE'])))<1e-12
    failed=json.loads((ROOT/'nosocial_failed_pilot.json').read_text());assert failed['status']=='stopped' and not failed['used_for_final_model_comparison']
    assert sha256(failed['last_saved_checkpoint'])==failed['last_checkpoint_sha256']
    corrected=json.loads((ROOT/'nosocial_correction_parameter_check.json').read_text())
    assert len(corrected['rows'])==4 and all(r['shared_tensors_exact'] for r in corrected['rows'])
    diagnosis=json.loads((ROOT/'nosocial_failure_diagnosis_v2.json').read_text())
    assert diagnosis['checkpoint_unchanged'] and diagnosis['max_gaussian_parameter_shift']>0
    amend=json.loads(Path('configs/stage4p1/nosocial_amendment.json').read_text())
    assert amend['run_script_sha256']==sha256('scripts/stage4p1/run_consistent.py')
    assert amend['new_code_sha256']==sha256('src/public_training/model_consistent.py')
    inventory=list(csv.DictReader((ROOT/'checkpoint_inventory.csv').open()));assert len(inventory)==64
    for ck in inventory:assert sha256(ck['path'])==ck['sha256']
    permutation=json.loads((ROOT/'permutation_execution.json').read_text());assert permutation['windows']==64 and permutation['targets']==235
    assert sha256(permutation['checkpoint'])==permutation['checkpoint_sha256']
    modules=list(csv.DictReader((ROOT/'permutation_modules.csv').open()))
    assert float(modules[0]['maximum_abs'])<1e-5 and float(modules[2]['maximum_abs'])>1e-3
    tests=ET.parse(ROOT/'final_tests.xml').getroot().find('testsuite')
    assert tests is not None and int(tests.attrib['tests'])==219
    assert all(int(tests.attrib[k])==0 for k in ('failures','errors','skipped'))
    reports=['01_TRAINING_PROTOCOL.md','02_PUBLIC_BASELINE_TRAINING.md','03_PERMUTATION_AUDIT.md','04_SOCIAL_ABLATION.md','05_RESEARCH_DECISION.md','STAGE4P1_REPORT.md']
    for name in reports:assert (Path('reports/stage4p1')/name).stat().st_size>1000
    changes=subprocess.check_output(['git','diff','--name-status',p['base_commit']],text=True).splitlines()
    assert all(row.startswith('A\t') for row in changes),changes
    paths=[row.split('\t')[1] for row in changes]
    pattern=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}')
    for path in paths:
        assert not path.startswith(('data/','experiments/'))
        assert Path(path).stat().st_size<8*1024*1024
        if not path.endswith('.gz'):assert not pattern.search(Path(path).read_text()),path
    subprocess.run(['git','diff','--check',p['base_commit']],check=True)
    for path in ('experiments/stage4p1/Full/epoch030.pt','experiments/stage4p1/NoSocial-ConsistentNorm/epoch030.pt',p['cache'],'.env','credentials.json'):
        subprocess.run(['git','check-ignore','-q',path],check=True)
    record=dict(status='pass',old_asset_hashes_verified=len(protection['files']),data_manifest_sha256=p['manifests'],cache_sha256=p['cache_sha256'],
        models=outputs,paired_rows=len(errors),unit_tests=219,unit_seconds=float(tests.attrib['time']),permutation_windows=64,
        failed_control_retained=True,NoSocial_amendment_disclosed=True,ETH_evaluated=False,auto_push=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        files_checked=len(paths),whitespace_check=True,sensitive_pattern_matches=0,
        logs=[dict(path=str(q),sha256=sha256(q),bytes=q.stat().st_size) for q in sorted(Path('experiments/stage4p1').glob('*.log'))],
        artifacts=[dict(path=str(q),sha256=sha256(q)) for q in sorted(ROOT.rglob('*')) if q.is_file() and q.name!='delivery_checks.json'])
    if '--record' in sys.argv:write_json_new(ROOT/'delivery_checks.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('logs','artifacts','data_manifest_sha256')}))


if __name__=='__main__':main()
