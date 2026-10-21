"""CPU-only final acceptance: immutable lineage, all targets, budgets, and tests."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
import numpy as np,pandas as pd,torch,xml.etree.ElementTree as ET

def main():
    assert not torch.cuda.is_initialized()
    frozen=json.loads((OUT/'protocol_frozen.json').read_text());execution=json.loads((OUT/'training_execution.json').read_text());decision=json.loads((OUT/'final_decision.json').read_text())
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in frozen['executable_files'])
    assert sha(CP)==CK
    assert json.loads((OUT/'history_after.json').read_text())['passed']
    assert json.loads((OUT/'resume_admission.json').read_text())['passed']
    cm=json.loads((OUT/'checkpoint_manifest.json').read_text());checks=[]
    for r in cm['checkpoints']:
        p=ROOT/r['path'];assert sha(p)==r['sha256'];v=torch.load(p,map_location='cpu',weights_only=True)
        ep=r['epoch'];assert v['progress']['epoch']==ep and v['progress']['global_step']==ep*1882 and v['progress']['next_epoch']==ep+1
        assert v['config']['original_epoch30_SHA']==CK and v['config']['train_config_SHA']==sha(CONF/'train.json')
        assert v['config']['evaluation_config_SHA']==sha(CONF/'evaluation.json')
        assert v['official_commit']==OFFICIAL and v['scheduler'] is None
        assert len(v['optimizer']['state'])==90 and all(float(s['step'])==ep*1882 for s in v['optimizer']['state'].values())
        assert v['optimizer']['param_groups'][0]['lr']==.0001
        checks.append(dict(epoch=ep,checkpoint_SHA=r['sha256'],all90_Adam_states_steps=ep*1882,original_checkpoint_SHA=CK,source_commit=v['source_commit']))
    cnew(OUT/'checkpoint_state_verification.csv',checks)
    assert len(checks)==execution['completed_epochs']-30<=20 and execution['completed_epochs']<=50
    epochs=pd.read_csv(OUT/'epochs.csv');from scripts.stage11b1c.data import load
    scenes=sorted(load('train').sceneId.unique())
    for r in epochs.to_dict('records'):
        order=[scenes[i] for i in torch.randperm(25,generator=torch.Generator().manual_seed(17+int(r['epoch'])-1)).tolist()]
        assert '|'.join(order)==r['scene_order'] and hashlib.sha256('|'.join(order).encode()).hexdigest()==r['scene_order_SHA']
    vstatus=pd.read_csv(OUT/'validation_status.csv');new=vstatus[(vstatus.epoch>30)&(vstatus.status=='COMPLETE')].epoch.tolist()
    gates=[]
    for ep in new:
        r=pd.read_csv(OUT/f'reproducibility_epoch{ep:03d}.csv');assert len(r)==6 and r.bitwise.all();gates.extend(r.to_dict('records'))
        p=pd.read_csv(OUT/f'paired_epoch{ep:03d}.csv');assert len(p)==998 and p.sample_id.nunique()==998
        h=pd.read_csv(OUT/f'candidate_errors_epoch{ep:03d}.csv.gz');assert len(h)==19960 and set(h['head'])==set(range(20))
        v=json.loads((OUT/f'validation_epoch{ep:03d}.json').read_text());assert v['status']=='COMPLETE' and v['targets_scored']==998
        assert v['prediction_manifest']['all_predictions_completed_before_GT_scoring']
    fixed=pd.read_csv(OUT/'a25_fixed_cohort.csv');assert (fixed.groupby('epoch').sample_id.nunique()==164).all()
    tr=pd.read_csv(OUT/'a25_transition.csv');assert (tr[['recovered','new_failures','persistent_failure','persistent_success']].sum(1)==998).all()
    tests=ET.parse(OUT/'full_tests_final.xml').getroot();suites=list(tests.iter('testsuite'));assert all(int(s.get('failures',0))==int(s.get('errors',0))==0 for s in suites)
    skip=[dict(test=c.get('name'),reason=c.find('skipped').get('message')) for c in tests.iter('testcase') if c.find('skipped') is not None]
    changed=subprocess.check_output(['git','diff',BASE,'--name-only'],text=True,cwd=ROOT).splitlines()
    allowed=['scripts/stage11b2b/','configs/stage11b2b/','reports/stage11b2b/','results/tables/stage11b2b/','results/figures/stage11b2b/']
    assert all(any(s.startswith(a) for a in allowed) for s in changed),changed
    newfiles=[p for base in allowed for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    disallowed=[str(p) for p in newfiles if p.suffix.lower() in {'.pt','.pth','.pkl','.ckpt','.pem','.key'} or p.name=='.env'];assert not disallowed
    large=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in newfiles if p.stat().st_size>10*1024*1024]
    assert not large,large
    assert decision['new_GPU_seconds']<16200 and not decision['semantic_direct_causal_evidence'] and not decision['innovation_algorithm_GO']
    jnew(OUT/'delivery_audit.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,source_commit=commit(),
        frozen_runtime_SHA_unchanged=True,all_history_SHA_unchanged=True,old_epoch30_SHA=CK,checkpoint_count=len(checks),completed_new_validations=new,
        all_new_K20_reverse_cache_repeat_bitwise=True,all_prediction_GT_separation=True,all998_targets_kept=True,
        test_counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']},skipped_tests=skip,
        files=newfiles.__len__(),large_files=large,sensitive_or_weight_files=disallowed,no_test_scoring=True,new_GPU_seconds=decision['new_GPU_seconds'],
        scope='Full protected file SHA + executable snapshot + complete checkpoint state + target/step/gate + test XML checks'))
    rows=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(newfiles) if p.name!='delivery_SHA256.csv']
    rows.append(dict(path=str((OUT/'delivery_audit.json').relative_to(ROOT)),sha256=sha(OUT/'delivery_audit.json'),bytes=(OUT/'delivery_audit.json').stat().st_size))
    cnew(OUT/'delivery_SHA256.csv',rows)
    print('DELIVERY PASS: checkpoints',len(checks),'full validations',new,'newGPU',decision['new_GPU_seconds'],flush=True)
if __name__=='__main__':main()
