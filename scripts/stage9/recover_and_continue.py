"""Only repair device-sensitive verification AFTER complete A training.
Never retries/retrains A, never changes model/data/loss/budget or completed weights.
Requires exact known CPU/CUDA comparison error and a complete 200-epoch A curve.
"""
from common import *
import traceback

def main():
    cfg=json.loads((ROOT/'configs/stage9/training.json').read_text())
    assert not (OUT/'B_training_complete.json').exists() and not (OUT/'training_summary.json').exists()
    failures=list(OUT.glob('train_failure_*.json'));assert failures
    known=[p for p in failures if 'cpu and cuda' in json.loads(p.read_text())['error'] and 'equal' in json.loads(p.read_text())['error']]
    assert known,'Not the pre-audited Adam CPU/CUDA comparison issue; do not continue.'
    with (OUT/'A_training_curve.csv').open() as f:curve=list(csv.DictReader(f))
    assert [int(r['epoch']) for r in curve]==list(range(1,cfg['epochs']+1))
    assert all(int(r['targets'])==29676 and int(r['optimizer_updates'])==232 for r in curve)
    final=CACHE/'checkpoints'/('A_epoch'+str(cfg['epochs'])+'.pt')
    assert sha(final)==curve[-1]['checkpoint_SHA']
    before_sha=sha(ROOT/'scripts/stage9/execute.py');before_common=sha(ROOT/'scripts/stage9/common.py')
    path=ROOT/'scripts/stage9/execute.py';s=path.read_text()
    assert "state=torch.load(p,map_location='cuda',weights_only=False);m.load_state_dict" in s
    s=s.replace("state=torch.load(p,map_location='cuda',weights_only=False);m.load_state_dict", "state=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict")
    s=s.replace('def train():','def train(completed_prefix=None):')
    s=s.replace("stage_start=time.monotonic();already=used();data=dataset('train');n=len(data['hist']);summary=[];checkpoints=[]", "stage_start=time.monotonic();already=used();data=dataset('train');n=len(data['hist']);summary=list(completed_prefix or []);checkpoints=[]\n    for item in summary:\n        with (OUT/(item['model']+'_training_curve.csv')).open() as f:\n            for row in csv.DictReader(f):\n                if row['checkpoint']:checkpoints.append(dict(model=item['model'],epoch=int(row['epoch']),path=row['checkpoint'],sha256=row['checkpoint_SHA'],final=int(row['epoch'])==cfg['epochs']))")
    # Exact loop is after data/start initialization, so timing/evaluation loops stay unchanged.
    old="    for kind in 'ABC':\n        m=model(kind,'cuda');seed(42);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);m.train()\n        initial="
    new="    for kind in 'ABC':\n        if any(item['model']==kind for item in summary):continue\n        m=model(kind,'cuda');seed(42);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);m.train()\n        initial="
    assert old in s;s=s.replace(old,new)
    s=s.replace("dict(models=summary,seconds=time.monotonic()-stage_start,code_commit=", "dict(models=summary,seconds=sum(r['seconds'] for r in summary),continuation_wall_seconds=time.monotonic()-stage_start,code_commit=")
    # File serialization only; optimizer and target weighting unchanged.
    s=s.replace("fieldnames=['epoch','step','loss','gradient_norm','seconds','targets'])", "fieldnames=['epoch','step','loss','gradient_norm','seconds','targets'],lineterminator='\\n')")
    s=s.replace('fieldnames=list(row))','fieldnames=list(row),lineterminator="\\n")')
    path.write_text(s)
    protocol=dict(reason='CPU/CUDA scalar Adam-step verification mismatch, not corrupt state or nonfinite training',A_training_repeated=False,
        A_epochs=cfg['epochs'],A_checkpoint_SHA=sha(final),original_executor_SHA=before_sha,repaired_executor_SHA=sha(path),model_code_SHA_unchanged=before_common,
        config_SHA_unchanged=sha(ROOT/'configs/stage9/training.json'),changes=['CPU map checkpoint restoration (preserves scalar Adam step device)','Skip independently completed full A run; keep A final checkpoint and curve','LF serialization for new CSV journals'],
        remaining_runs=['B','C'],same_initialization=True,no_validation_error_seen=True,no_new_seed_or_epoch=True,evidence=[str(p.relative_to(ROOT)) for p in known])
    json_new(ROOT/'configs/stage9/recovery_protocol.json',protocol)
    # All experiment code must be tracked/frozen before B/C. Do not stage weights or mutable training logs.
    subprocess.run(['git','add','scripts/stage9','tests/test_stage9_controlled_pilot.py','configs/stage9/recovery_protocol.json'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','fix: make Stage9 optimizer restoration device-safe without repeating training'],cwd=ROOT,check=True)
    assert sha(ROOT/'scripts/stage9/common.py')==before_common
    import execute
    m=model('A','cuda');opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);state=execute.restore(m,opt,final)
    assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in state['model'].items())
    assert torch.equal(torch.get_rng_state(),state['torch_rng'])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda_rng']))
    assert state['epoch']==cfg['epochs']
    # All Adam values restored onto the expected devices. No optimizer update here.
    original=state['optimizer']['state'];restored=opt.state_dict()['state']
    for key in original:
        for name,value in original[key].items():
            if isinstance(value,torch.Tensor):assert torch.equal(value,restored[key][name].cpu())
    with (OUT/'gpu_usage.csv').open() as f:ledger=list(csv.DictReader(f))
    duration=float([r for r in ledger if r['task']=='train'][-1]['wall_seconds'])
    prefix=dict(model='A',epochs=cfg['epochs'],updates=cfg['epochs']*232,target_visits=cfg['epochs']*29676,seconds=duration,
        seconds_basis='Original complete A process wall including failed device-only comparison; preserved, no repeated updates',parameters=params(m),
        initial_SHA=sha(CACHE/'checkpoints/formal_A_initial.pt'),final_checkpoint=str(final.relative_to(ROOT)),final_SHA=sha(final),restored_model_optimizer_RNG=True,
        allocated=max(int(r['allocated']) for r in curve),reserved=max(int(r['reserved']) for r in curve),restoration_verified_after_device_fix=True)
    json_new(OUT/'A_training_complete.json',prefix)
    json_new(OUT/'A_restoration_after_fix.json',dict(passed=True,strict=True,Adam_values_exact=True,RNG_exact=True,no_additional_optimizer_update=True,checkpoint_SHA=sha(final),source_SHA=sha(path)))
    del m,opt,restored,original,state;torch.cuda.empty_cache()
    execute.train(completed_prefix=[prefix])

if __name__=='__main__':
    t=time.monotonic();torch.set_num_threads(1)
    try:main()
    except Exception as e:
        json_new(OUT/('continuation_failure_'+str(time.time_ns())+'.json'),dict(error=str(e),traceback=traceback.format_exc()));raise
    finally:gpu_ledger('formal_B_C_continuation_and_restoration',time.monotonic()-t)
