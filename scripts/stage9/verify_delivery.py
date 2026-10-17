"""Final real-result, source preservation, budget and publication checks."""
import re,xml.etree.ElementTree as ET
from common import *

def run():
    cfg=json.loads((ROOT/'configs/stage9/training.json').read_text())
    summary=json.loads((OUT/'training_summary.json').read_text())
    assert len(summary['models'])==3
    audits=[]
    for item in summary['models']:
        k=item['model'];assert item['epochs']==cfg['epochs']==200
        assert item['updates']==46400 and item['target_visits']==5935200
        assert sha(ROOT/item['final_checkpoint'])==item['final_SHA']
        with (OUT/(k+'_training_curve.csv')).open() as f:curves=list(csv.DictReader(f))
        assert len(curves)==200 and [int(r['epoch']) for r in curves]==list(range(1,201))
        sums={e:[0.,0,0] for e in range(1,201)};counter=0
        with gzip.open(OUT/(k+'_training_steps.csv.gz'),'rt') as f:
            for r in csv.DictReader(f):
                counter+=1;assert int(r['step'])==counter
                assert np.isfinite(float(r['loss'])) and np.isfinite(float(r['gradient_norm'])) and float(r['seconds'])>0
                e=int(r['epoch']);n=int(r['targets']);sums[e][0]+=float(r['loss'])*n;sums[e][1]+=n;sums[e][2]+=1
        assert counter==46400
        for r in curves:
            s,n,count=sums[int(r['epoch'])]
            assert n==29676 and count==232 and abs(s/n-float(r['loss']))<1e-9 and r['finite']=='True'
        audits.append(dict(model=k,epochs=200,updates=counter,target_visits=sum(r[1] for r in sums.values()),step_loss_curve_consistent=True,checkpoint_SHA_verified=True))
    csv_new(OUT/'training_accounting_audit.csv',audits)
    with (OUT/'checkpoint_index.csv').open() as f:index=list(csv.DictReader(f))
    assert len(index)==30
    for r in index:assert sha(ROOT/r['path'])==r['sha256']
    csv_new(OUT/'all_checkpoint_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,role='formal_or_engineering_as_filename') for p in sorted((CACHE/'checkpoints').glob('*.pt'))])
    preservation=json.loads((OUT/'historical_preservation_final.json').read_text());assert preservation['passed']
    structure=json.loads((OUT/'structure_tests.json').read_text());assert structure['passed']
    tests=ET.parse(OUT/'unit_tests.xml');cases=tests.findall('.//testcase')
    assert len(cases)>=458 and not tests.findall('.//failure') and not tests.findall('.//error') and not tests.findall('.//skipped')
    # The model and numeric protocol are identical across execution repair commits.
    frozen_model=subprocess.check_output(['git','show','c5d2fb2:scripts/stage9/common.py'],cwd=ROOT)
    frozen_cfg=subprocess.check_output(['git','show','c5d2fb2:configs/stage9/training.json'],cwd=ROOT)
    assert hashlib.sha256(frozen_model).hexdigest()==sha(ROOT/'scripts/stage9/common.py')
    assert hashlib.sha256(frozen_cfg).hexdigest()==sha(ROOT/'configs/stage9/training.json')
    (OUT/'executor_recovery_diff.txt').write_bytes(subprocess.check_output(['git','diff','--unified=0','c5d2fb2','837da7b','--','scripts/stage9/execute.py'],cwd=ROOT))
    sources=json.loads((OUT/'source_versions.json').read_text())
    for r in sources['repositories']:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/r['path'],text=True).strip()==r['commit']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/r['path'],text=True).strip()
    elapsed=used();assert elapsed<43200
    json_new(OUT/'resource_budget_audit.json',dict(charged_GPU_task_wall_seconds=elapsed,maximum_seconds=43200,passed=True,
        basis='Conservative sum including held-CUDA CPU/I-O and failed probes; overlaps double charged. Some explicitly marked conservative upper bounds.',epochs_not_extended=True))
    prefixes=('configs/stage9/','scripts/stage9/','reports/stage9/','results/tables/stage9/','tests/test_stage9_controlled_pilot.py')
    changed=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith(prefixes) for p in changed),changed
    assert len(list((ROOT/'reports/stage9').glob('*.md')))==9
    files=[]
    for directory in ['configs/stage9','scripts/stage9','reports/stage9','results/tables/stage9']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.append(ROOT/'tests/test_stage9_controlled_pilot.py')
    secret=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'AKIA[0-9A-Z]{16}']
    for p in files:
        assert p.stat().st_size<10_000_000,(p,p.stat().st_size)
        assert p.suffix not in {'.pt','.pth','.ckpt','.pkl','.npy','.npz','.pdf','.pem','.key'},p
        if p.suffix!='.gz':assert not any(re.search(s,p.read_bytes()) for s in secret),p
    for p in [CACHE/'checkpoints/A_epoch200.pt',CACHE/'A_validation_predictions.npz',CACHE/'train_data.pt']:
        assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=ROOT).returncode==0
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    csv_new(OUT/'delivery_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files) if p.name not in {'delivery_SHA256.csv','delivery_checks.json'}])
    decision=json.loads((OUT/'final_decision.json').read_text())
    json_new(OUT/'delivery_checks.json',dict(passed=True,decision=decision['decision'],base_commit=BASE,
        initial_freeze_commit='c5d2fb2',restoration_fix_commit='837da7b',model_code_unchanged_between_runs=True,config_unchanged_between_runs=True,
        full_tests_passed=len(cases),historical_SHA_checks=preservation['checked_files'],reports=9,
        GPU_charged_seconds=elapsed,local_commit_only=True,no_push=True,no_test_metrics=True,no_extra_models_seeds_or_epochs=True))
    print(json.dumps(dict(passed=True,tests=len(cases),GPU_charged_seconds=elapsed,decision=decision['decision']),indent=2))

if __name__=='__main__':run()
