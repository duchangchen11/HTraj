"""CPU-only static restore/schema/data admission before any GPU work."""
from .common import *
import torch,os,numpy as np
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b1c.data import load,manifest
from scripts.stage11b0.run_admission import optimizer_digest

def immutable_record(path,value):
    if path.exists():assert json.loads(path.read_text())==value
    else:jnew(path,value)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    configure_torch();assert CP.is_file() and sha(CP)==CK
    saved=torch.load(CP,map_location='cpu',weights_only=True)
    assert set(['model','optimizer','rng','progress','config','official_commit']).issubset(saved)
    assert saved['official_commit']==OFFICIAL
    p=saved['progress'];assert p['epoch']==30 and p['global_step']==56460 and p['next_epoch']==31 and p['next_scene_seed']==47 and p['epoch_boundary']
    net,cfg=build_model(17,'cpu');res=net.load_state_dict(saved['model'],strict=True);assert not res.missing_keys and not res.unexpected_keys
    names=[n for n,v in net.named_parameters() if v.requires_grad];opt=torch.optim.Adam([v for v in net.parameters() if v.requires_grad],lr=.0001)
    assert len(names)==90 and saved['optimizer']['param_groups'][0]['params']==list(range(90))
    opt.load_state_dict(saved['optimizer']);assert len(opt.state)==90 and all(float(s['step'])==56460 for s in opt.state.values())
    assert all(k in saved['rng'] for k in ['python','numpy','torch','cuda']) and len(saved['rng']['cuda'])==1
    old=json.loads((ROOT/'results/tables/stage11b1c/protocol_frozen.json').read_text())
    assert saved['config']['train_config_SHA']==sha(ROOT/'configs/stage11b1c/train.json')
    assert saved['config']['evaluation_config_SHA']==sha(CONF/'evaluation.json')
    assert saved['config']['data_manifest_SHA']==old['data_manifest_SHA']
    for split in ['train','validation']:
        assert sha(OUT/f'{split}_target_manifest.csv')==old['data_manifest_SHA'][split]
    groups={type(m).__name__ for n,m in net.named_modules() if not n.startswith('semantic_segmentation')}
    immutable_record(OUT/'checkpoint_static_audit.json',dict(passed=True,checkpoint_SHA=CK,model_state_SHA=state_digest(saved['model']),progress=p,
        optimizer_parameter_names=names,optimizer_SHA=optimizer_digest(opt),optimizer_states=90,Adam_step_values=sorted({float(s['step']) for s in opt.state.values()}),
        config=saved['config'],core_module_types=sorted(groups),RNG_keys=list(saved['rng']),CUDA_RNG_states=1,weights_only_safe_load=True,
        no_optimizer_reset=True,CPU_only=True,CUDA_initialized=False,old_each_batch_set_training=True))
    budget=dict(reference_epoch_seconds=461.9348,reference_validation_seconds=568,point_seconds=20*461.9348+4*568+600,
        conservative_seconds=20*461.9348*1.25+4*568*1.25+600,hard_seconds=16200,safety_seconds=15000,
        fit_extension=20*461.9348*1.25+4*568*1.25+600<15000,assumptions='Prior same pipeline full epoch mean; conservativeTTST reference;1.25x +600s load/admission/save; guarded actuals supersede estimate')
    assert budget['fit_extension'];immutable_record(OUT/'budget_preestimate.json',budget)
    print('CPU static restore PASS',p,'budget',budget,flush=True)
if __name__=='__main__':main()
