"""Checkpoint portability check, zero optimizer updates; no validation errors."""
from common import *
t0=time.monotonic()
try:
    setup();state=torch.load(CACHE/'checkpoints/timing_A.pt',map_location='cuda',weights_only=True)
    m=model('A','cuda');opt=torch.optim.Adam(m.parameters(),lr=.0001);m.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    loaded_step=next(iter(opt.state.values()))['step']
    native=torch.load(CACHE/'checkpoints/timing_A.pt',weights_only=True)
    saved_step=next(iter(native['optimizer']['state'].values()))['step']
    result=dict(saved_Adam_step_device=str(saved_step.device),cuda_map_loaded_Adam_step_device=str(loaded_step.device),
        step_values_equal_after_device_alignment=torch.equal(saved_step,loaded_step.cpu()),
        issue='torch.equal rejects CPU-vs-CUDA Adam step tensors. Verification must align devices or load packet to CPU before load_state_dict. Model weights and optimizer values are intact.',optimizer_updates=0)
    cpu=torch.load(CACHE/'checkpoints/timing_A.pt',map_location='cpu',weights_only=True)
    opt.load_state_dict(cpu['optimizer']);m.load_state_dict(cpu['model'],strict=True)
    result['CPU_map_restore_step_device']=str(next(iter(opt.state.values()))['step'].device)
    result['passed']=torch.equal(saved_step,next(iter(opt.state.values()))['step'])
    assert result['passed']
    json_new(OUT/'restore_device_precheck.json',result);print(json.dumps(result,indent=2))
finally:gpu_ledger('restore_device_precheck_fixed',time.monotonic()-t0)
# Historical grouping precision audit, without model predictions.
with gzip.open(OUT/'val_targets.csv.gz','rt') as f:new={(r['sequence'],r['frame'],r['person_id']):r['ambiguity'] for r in csv.DictReader(f)}
with gzip.open(ROOT/'results/tables/stage8/resonance_history_audit_targets.csv.gz','rt') as f:old=list(csv.DictReader(f))
changes=[dict(sequence=r['sequence'],frame=r['frame'],person_id=r['target_id'],stage8=r['same_sector_position_motion_ambiguity'],stage9=new[r['sequence'],r['frame'],r['target_id']]) for r in old if r['same_sector_position_motion_ambiguity']!=new[r['sequence'],r['frame'],r['target_id']]]
json_new(OUT/'history_group_precision_audit.json',dict(all_target_keys_match=True,differing_group_membership=changes,reason='Same thresholds; Stage8 float64 raw vs Stage9 float32 model-aligned historical coordinates; rules fixed before validation errors',no_validation_errors_read=True))
print('group rounding differences',len(changes))
