"""One unmodified official train() epoch with read-only instrumentation."""
import ast
import contextlib
import gc
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from src.agentformer_feasibility.common import *
from src.agentformer_feasibility.diagnosis import step_summary
from src.socialvae_baseline.artifacts import json_new,state_info
from src.socialvae_causal.io import read_csv,csv_new
from scripts.stage7p1.runtime import ResourceMeter


def initialize(stage):
    from model.model_lib import model_dict
    from utils.config import Config
    from utils.torch import get_scheduler
    from utils.utils import prepare_seed
    cfg=Config('hotel_agentformer_pre' if stage=='CVAE' else 'hotel_agentformer')
    assert cfg.lr_scheduler=='step'
    if stage=='DLow':
        assert cfg.pred_epoch==1 and cfg.pred_cfg=='hotel_agentformer_pre'
        previous=json.loads((OUT/'CVAE_epoch.json').read_text())
        assert previous['success'] and digest(previous['checkpoint'])==previous['checkpoint_SHA256']
    prepare_seed(cfg.seed)
    model=model_dict[cfg.model_id](cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    scheduler=get_scheduler(optimizer,policy='step',decay_step=cfg.decay_step,decay_gamma=cfg.decay_gamma)
    model.set_device(torch.device('cuda:0'));model.train()
    if stage=='DLow':
        own={id(p) for g in optimizer.param_groups for p in g['params']}
        assert not own.intersection(id(p) for p in model.pred_model[0].parameters())
        assert not model.pred_model[0].training
    return cfg,model,optimizer,scheduler


def finite_norm(parameters,grad=False):
    values=[p.grad if grad else p for p in parameters if not grad or p.grad is not None]
    if not values:return None
    with torch.no_grad():
        norm=float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(values,2))).item())
    if not math.isfinite(norm):raise FloatingPointError('Nonfinite '+('gradient' if grad else 'parameter')+' norm')
    return norm


def original_train(namespace):
    tree=ast.parse((SOURCE/'train.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'train','logging'}]
    assert len(nodes)==2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/'train.py'),'exec'),namespace)
    namespace['train'](0)


class TensorboardReceipt:
    def __init__(self):self.calls=0
    def add_scalar(self,*args,**kwargs):self.calls+=1


def resume_probe(stage,path,generator_state):
    """One engineering update on first valid next-epoch sample; no extra full epoch."""
    from data.dataloader import data_generator
    begin=time.perf_counter()
    cp=torch.load(path,map_location='cpu',weights_only=False)
    cfg,model,opt,scheduler=initialize(stage)
    model.load_state_dict(cp['model_dict'],strict=True)
    opt.load_state_dict(cp['opt_dict']);scheduler.load_state_dict(cp['scheduler_dict'])
    assert nested_equal(model.state_dict(),cp['model_dict'])
    assert nested_equal(opt.state_dict(),cp['opt_dict']) and nested_equal(scheduler.state_dict(),cp['scheduler_dict'])
    generator=data_generator(cfg,io.StringIO(),split='train',phase='training')
    generator.sample_list=list(generator_state['sample_list']);generator.index=generator_state['index']
    restore_rng(cp['rng_state'])
    assert nested_equal(rng_state(),cp['rng_state'])
    before=state_info(model)
    fixed_base=state_info(model.pred_model[0]) if stage=='DLow' else None
    original_steps={k:float(s['step']) for k,s in opt.state_dict()['state'].items()}
    generator.shuffle()
    packet=None
    while packet is None and not generator.is_epoch_end():packet=generator()
    assert packet is not None and packet['seq'].endswith('_train')
    model.set_data(packet);model();loss,terms,raw=model.compute_loss()
    assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();gradient=finite_norm(list(model.parameters()),grad=True);opt.step()
    finite_norm(list(model.parameters()))
    after=state_info(model)
    new_steps={k:float(s['step']) for k,s in opt.state_dict()['state'].items()}
    assert after['parameter_state_sha256']!=before['parameter_state_sha256']
    assert any(new_steps[k]==v+1 for k,v in original_steps.items())
    if stage=='DLow':assert state_info(model.pred_model[0])==fixed_base
    row=dict(strict_model_restore=True,optimizer_state_exact=True,scheduler_state_exact=True,RNG_exact=True,
        optimizer_entries=len(opt.state),optimizer_update_probe=1,not_an_extra_epoch=True,
        sequence=packet['seq'],frame=int(packet['frame']),people=len(packet['valid_id']),
        finite_loss=float(loss),gradient_norm=gradient,model_state_before=before,model_state_after=after,
        optimizer_step_min_before=min(original_steps.values()),optimizer_step_max_before=max(original_steps.values()),
        optimizer_step_min_after=min(new_steps.values()),optimizer_step_max_after=max(new_steps.values()),
        base_CVAE_unchanged=(True if stage=='DLow' else None),seconds=time.perf_counter()-begin)
    del model,opt,scheduler,cp;gc.collect();torch.cuda.empty_cache()
    return row


def run(stage):
    from data.dataloader import data_generator
    from utils.utils import AverageMeter,print_log,convert_secs2time
    cfg,model,optimizer,scheduler=initialize(stage)
    expected={(r['sequence'],int(r['last_observed_frame'])):int(r['people']) for r in
        read_csv(ROOT/'configs/stage7p4/train_windows.csv.gz')}
    expected_ids={(r['sequence'],int(r['last_observed_frame'])):set(json.loads(r['target_ids'])) for r in read_csv(ROOT/'configs/stage7p4/train_windows.csv.gz')}
    steps,reads,seen=[],[],set()
    cache=CACHE/stage;cache.mkdir(parents=True,exist_ok=True)
    checkpoint=WORK/'results'/cfg.id/'models/model_0001.p'
    assert not checkpoint.exists()
    result=dict(stage=stage,success=False,full_epochs_completed=0,optimizer_updates=0,
        random_initialization=True,author_initialization=False,context_truncated=False,
        original_config=cfg.id,learning_rate=cfg.lr,seed=cfg.seed,
        official_configured_epochs=cfg.num_epochs,stage_execution_epochs=1)
    base_before=state_info(model.pred_model[0]) if stage=='DLow' else None
    initial=state_info(model)
    meter=None;current={};stream=None;update_count=[0]
    try:
        with (cache/'official_train_stdout.txt').open('x') as log, (cache/'steps.jsonl').open('x') as stream, contextlib.redirect_stdout(log):
            loader_begin=time.perf_counter()
            generator=data_generator(cfg,log,split='train',phase='training')
            result['loader_initialization_seconds']=time.perf_counter()-loader_begin
            class InstrumentedGenerator:
                def __getattr__(self,name):return getattr(generator,name)
                def shuffle(self):return generator.shuffle()
                def is_epoch_end(self):return generator.is_epoch_end()
                def __call__(self):
                    start=time.perf_counter();packet=generator();end=time.perf_counter()
                    reads.append(dict(slot_visit=len(reads)+1,seconds=end-start,valid=packet is not None))
                    if packet is not None:
                        key=packet['seq'],int(packet['frame'])
                        assert key in expected and key not in seen and len(packet['valid_id'])==expected[key]
                        assert set(map(int,packet['valid_id']))==expected_ids[key]
                        assert all(torch.all(m==1) for m in [*packet['pre_motion_mask'],*packet['fut_motion_mask']])
                        seen.add(key)
                        current.clear();current.update(step=len(steps)+1,sequence=key[0],frame=key[1],people=expected[key],
                            slot_visit=len(reads),data_seconds=end-start,_start=start,_model_start=end)
                    return packet
            compute=model.compute_loss
            def checked_loss():
                value,terms,raw=compute()
                scalar=float(value.detach())
                if not math.isfinite(scalar) or not all(math.isfinite(float(v)) for v in [*terms.values(),*raw.values()]):
                    raise FloatingPointError('Nonfinite original loss; abort without retry')
                assert int(model.data['agent_num'])==current['people'],'Unexpected actor truncation'
                current.update(loss=scalar,weighted_terms=terms,unweighted_terms=raw)
                return value,terms,raw
            model.compute_loss=checked_loss
            def pre_step(opt,args,kwargs):
                t=time.perf_counter()
                current['gradient_norm']=finite_norm(list(model.parameters()),grad=True)
                if stage=='DLow':finite_norm(list(model.pred_model[0].parameters()),grad=True)
                current['gradient_check_seconds']=time.perf_counter()-t
            def post_step(opt,args,kwargs):
                update_count[0]+=1
                t=time.perf_counter();finite_norm(list(model.parameters()));torch.cuda.synchronize();end=time.perf_counter()
                current['parameter_check_seconds']=end-t
                row={k:v for k,v in current.items() if not k.startswith('_')}
                row.update(step_seconds=end-current['_start'],model_and_optimizer_seconds=end-current['_model_start'],
                    allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
                steps.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if len(steps)%100==0:
                    print(f'{stage}: {len(steps)}/{len(expected)} valid updates, last N={current["people"]}, loss={current["loss"]:.4f}',file=sys.__stdout__,flush=True)
            pre=optimizer.register_step_pre_hook(pre_step);post=optimizer.register_step_post_hook(post_step)
            tb=TensorboardReceipt()
            env=dict(time=time,generator=InstrumentedGenerator(),model=model,optimizer=optimizer,scheduler=scheduler,
                cfg=cfg,AverageMeter=AverageMeter,tb_ind=0,tb_logger=tb,log=log,print_log=print_log,
                convert_secs2time=convert_secs2time,args=type('Args',(),dict(cfg=cfg.id))())
            with ResourceMeter() as meter:original_train(env)
            pre.remove();post.remove();model.compute_loss=compute
            assert seen==expected.keys() and len(steps)==len(expected)==update_count[0]
            assert len(reads)==int(generator.num_total_samples)
            after=state_info(model)
            assert initial['parameter_state_sha256']!=after['parameter_state_sha256']
            if stage=='DLow':assert state_info(model.pred_model[0])==base_before
            generator_state=dict(sample_list=list(generator.sample_list),index=generator.index)
            cp=dict(model_dict=model.state_dict(),opt_dict=optimizer.state_dict(),scheduler_dict=scheduler.state_dict(),
                rng_state=rng_state(),generator_state=generator_state,epoch=1,stage=stage,official_commit=protocol()['official_commit'],
                freeze_commit=json.loads((OUT/'execution_start.json').read_text())['freeze_commit'],
                external_CVAE_checkpoint=(str(WORK/'results/hotel_agentformer_pre/models/model_0001.p') if stage=='DLow' else None))
            begin=time.perf_counter()
            with checkpoint.open('xb') as f:torch.save(cp,f)
            result.update(full_epochs_completed=1,optimizer_updates=len(steps),candidate_frame_slots=len(reads),
                valid_windows=len(steps),full_target_windows=sum(r['people'] for r in steps),
                max_context_people=max(r['people'] for r in steps),resource=meter.result,
                checkpoint=str(checkpoint),checkpoint_SHA256=digest(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                checkpoint_save_seconds=time.perf_counter()-begin,initial_state=initial,final_state=after,
                scheduler_state=scheduler.state_dict(),RNG_saved=True,
                external_CVAE_state_before=base_before,external_CVAE_unchanged=(True if stage=='DLow' else None),
                tensorboard_scalar_calls=tb.calls,**step_summary(steps),
                total_data_read_seconds=sum(r['seconds'] for r in reads),
                total_gradient_and_parameter_check_seconds=sum(r['gradient_check_seconds']+r['parameter_check_seconds'] for r in steps))
            # Preserve official train math and scheduler; release graph references before shadow restoration.
            env.clear();model.data=None
            if stage=='DLow':model.pred_model[0].data=None
            del cp,model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
            result['resume_probe']=resume_probe(stage,checkpoint,generator_state)
            assert digest(checkpoint)==result['checkpoint_SHA256']
            result['success']=True
    except Exception:
        result.update(failure=traceback.format_exc(),optimizer_updates=update_count[0],completed_valid_windows=len(steps),
            failing_window={k:v for k,v in current.items() if not k.startswith('_')},
            no_retry=True,OOM=isinstance(sys.exc_info()[1],torch.cuda.OutOfMemoryError))
        if meter is not None and hasattr(meter,'result'):result['resource']=meter.result
        try:
            failure_path=cache/'failure_state.pth'
            with failure_path.open('xb') as f:
                torch.save(dict(model_dict=model.state_dict(),opt_dict=optimizer.state_dict(),scheduler_dict=scheduler.state_dict(),
                    rng_state=rng_state(),current=current),f)
            result.update(failure_checkpoint=str(failure_path),failure_checkpoint_SHA256=digest(failure_path))
        except Exception:result['failure_preservation_exception']=traceback.format_exc()
        print('STOP',stage,result['failure'],file=sys.__stdout__,flush=True)
    finally:
        flat=[]
        for r in steps:
            flat.append({k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in r.items()})
        csv_new(OUT/f'{stage}_steps.csv.gz',flat,fieldnames=list(flat[0]) if flat else ['step','loss'])
        csv_new(OUT/f'{stage}_data_reads.csv.gz',reads,fieldnames=['slot_visit','seconds','valid'])
        result['official_training_source_SHA256']=digest(SOURCE/'train.py')
        result['raw_log']=str(cache/'official_train_stdout.txt')
        if (cache/'official_train_stdout.txt').exists():
            result['raw_log_SHA256']=digest(cache/'official_train_stdout.txt')
            with (OUT/f'{stage}_official_train.txt').open('x') as f:
                f.write('\n'.join(line.rstrip() for line in (cache/'official_train_stdout.txt').read_text().splitlines())+'\n')
        json_new(OUT/f'{stage}_epoch.json',result)
    return result
