"""One amended NoSocial run; same data/seed/optimizer/loss/budget, consistent normalization. Never imports official train driver."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,csv,gzip,json,time,subprocess
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.public_training.data import network_input
from src.public_training.model import forward,nll
from src.public_training.model_consistent import make_model
from src.public_training.engine import evaluate,truth_velocity
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage4p1');EXP=Path('experiments/stage4p1')


def checkpoint(path,model,optimizer,epoch):
    assert not path.exists()
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)


def restored_model(path,kind,lr):
    saved=torch.load(path,map_location='cpu',weights_only=True);model=make_model(kind).cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=lr);model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return model,optimizer,saved['epoch']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['NoSocial-ConsistentNorm'],required=True);parser.add_argument('--phase',choices=['smoke','finish'],required=True);args=parser.parse_args()
    p=json.loads(Path('configs/stage4p1/protocol.json').read_text());kind=args.model
    amendment=json.loads(Path('configs/stage4p1/nosocial_amendment.json').read_text())
    assert amendment['immutable_protocol_sha256']==sha256('configs/stage4p1/protocol.json')
    assert amendment['new_code_sha256']==sha256('src/public_training/model_consistent.py')
    assert not p['ETH_evaluated'] and not p['pretrained_initialization']
    assert sha256(p['cache'])==p['cache_sha256']
    for info in p['manifests'].values():assert sha256(info['path'])==info['sha256']
    data=torch.load(p['cache'],weights_only=True);train,val=data['train'],data['validation']
    assert {s['recording'] for s in val}=={'crowds_zara02'}
    assert not ({s['recording'] for s in train}&{s['recording'] for s in val})
    assert not any(s['recording']=='biwi_eth' for s in train+val)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    folder=EXP/kind;folder.mkdir(parents=True,exist_ok=True);result=ROOT/kind;result.mkdir(parents=True,exist_ok=True)
    if args.phase=='smoke':
        assert not (folder/'epoch001.pt').exists()
        model=make_model(kind).cuda();optimizer=torch.optim.SGD(model.parameters(),lr=p['learning_rate']);first,last=1,5
        initial=sha256_model(model);rows=[];best_loss=float('inf');best_epoch=None
    else:
        smoke=json.loads((result/'smoke.json').read_text());assert smoke['gate_pass']
        assert not (folder/'epoch006.pt').exists()
        model,optimizer,previous=restored_model(folder/'epoch005.pt',kind,p['learning_rate']);assert previous==5
        first,last=6,30;rows=list(csv.DictReader((result/'curve_smoke.csv').open()));best_loss=min(float(r['validation_NLL']) for r in rows)
        best_epoch=min(int(r['epoch']) for r in rows if float(r['validation_NLL'])==best_loss);initial=smoke['initial_state_sha256']
    trace=[];handles=[]
    if first==1:
        def hook(name):
            def record(module,inputs,output):
                tensors=[output] if torch.is_tensor(output) else list(output)
                trace.append(dict(module=name,input_shapes=[list(v.shape) for v in inputs if torch.is_tensor(v)],output_shapes=[list(v.shape) for v in tensors if torch.is_tensor(v)]))
            return record
        for name,module in model.named_modules():handles.append(module.register_forward_hook(hook(name)))
    unused=None
    for epoch in range(first,last+1):
        model.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(100000+p['seed']+epoch)).tolist()
        epoch_loss=0.;norms=[];n_targets=0
        for chunk_start in range(0,len(order),p['accumulation_windows']):
            chunk=order[chunk_start:chunk_start+p['accumulation_windows']];optimizer.zero_grad(set_to_none=True)
            for i in chunk:
                sample=train[i];v,a=network_input(sample,'cuda');pred=forward(model,v,a)
                if handles:
                    for handle in handles:handle.remove()
                    handles=[]
                loss=nll(pred,truth_velocity(sample).cuda(),sample['target_mask'].cuda())
                if not torch.isfinite(loss):raise FloatingPointError('Training NLL nonfinite; stop, no adaptive retry')
                (loss/len(chunk)).backward();epoch_loss+=float(loss);n_targets+=len(sample['target_ids'])
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True);norms.append(float(norm))
            if unused is None:unused=[name for name,parameter in model.named_parameters() if parameter.grad is None]
            optimizer.step()
        torch.cuda.synchronize();train_seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
        metrics,_=evaluate(model,val,p['evaluation_seed'])
        row=dict(epoch=epoch,seed=p['seed'],model=kind,train_NLL_window_mean=epoch_loss/len(train),
            gradient_norm_mean=float(np.mean(norms)),gradient_norm_p95=float(np.quantile(norms,.95)),gradient_norm_max=max(norms),
            train_seconds=train_seconds,train_windows_per_second=len(train)/train_seconds,train_target_windows_per_second=n_targets/train_seconds,
            train_windows=len(train),train_targets=n_targets,optimizer_updates=len(norms),peak_train_cuda_allocated_MiB=peak,
            **{'validation_'+k:v for k,v in metrics.items()})
        assert np.isfinite([v for v in row.values() if isinstance(v,(int,float))]).all()
        rows.append(row);path=folder/f'epoch{epoch:03d}.pt';checkpoint(path,model,optimizer,epoch)
        if metrics['NLL']<best_loss:best_loss=metrics['NLL'];best_epoch=epoch
        print(json.dumps(row),flush=True)
    curve='curve_smoke.csv' if args.phase=='smoke' else 'curve_30.csv';csv_new(result/curve,rows)
    common=dict(model=kind,seed=p['seed'],initial_state_sha256=initial,parameters=sum(v.numel() for v in model.parameters()),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        protocol_sha256=sha256('configs/stage4p1/protocol.json'),manifest_sha256=p['manifests']['A_offline_complete20']['sha256'],
        amendment_sha256=sha256('configs/stage4p1/nosocial_amendment.json'),model_code_sha256=sha256('src/public_training/model_consistent.py'),engine_sha256=sha256('src/public_training/engine.py'),unused_parameter_names=unused,
        ETH_evaluated=False,pretrained_initialization=False,optimizer='SGD',learning_rate=p['learning_rate'])
    if args.phase=='smoke':
        model.eval();v,a=network_input(val[0],'cuda')
        with torch.no_grad():before=forward(model,v,a).clone()
        loaded,opt,e=restored_model(folder/'epoch005.pt',kind,p['learning_rate']);loaded.eval()
        with torch.no_grad():after=forward(loaded,v,a)
        exact=torch.equal(before,after);assert exact and opt.state_dict()['param_groups']==optimizer.state_dict()['param_groups']
        gate=exact and all(float(r['gradient_norm_max'])>0 for r in rows)
        write_json_new(result/'smoke.json',dict(**common,gate_pass=gate,checkpoint_restore_output_exact=exact,optimizer_restore_exact=True,
            checkpoint_sha256=sha256(folder/'epoch005.pt'),best_epoch=best_epoch,actual_epochs=5,layer_shapes=trace))
        train_metrics,_=evaluate(model,train,p['evaluation_seed']);write_json_new(result/'train_epoch5.json',train_metrics)
    else:
        errors=[];summaries=[];checkpoint_rows=[]
        for label,ep in [('best',best_epoch),('last',30)]:
            path=folder/f'epoch{ep:03d}.pt';current,_,_=restored_model(path,kind,p['learning_rate'])
            checkpoint_rows.append(dict(label=label,epoch=ep,path=str(path),sha256=sha256(path)))
            for split,samples in [('train',train),('validation',val)]:
                metrics,targets=evaluate(current,samples,p['evaluation_seed'],return_targets=split=='validation')
                summaries.append(dict(model=kind,label=label,epoch=ep,split=split,**metrics))
                errors.extend(dict(model=kind,label=label,epoch=ep,**r) for r in targets)
        csv_new(result/'best_last.csv',summaries)
        with gzip.open(result/'target_errors.csv.gz','xt',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
        write_json_new(result/'execution.json',dict(**common,actual_epochs=30,best_epoch=best_epoch,best_validation_NLL=best_loss,
            checkpoints=checkpoint_rows,training_seconds=sum(float(r['train_seconds']) for r in rows),
            evaluation_seconds=sum(float(r['validation_evaluation_seconds']) for r in rows),
            initial_weights='from scratch seed17; Full and NoSocial same source initialization, NoSocial TXP weights reshaped',
            training_resume='exact epoch5 weights, optimizer and RNG; no repeated epochs or warm start from pretrained model'))


def sha256_model(model):
    import hashlib
    h=hashlib.sha256()
    for name,parameter in model.state_dict().items():
        h.update(name.encode());h.update(str((tuple(parameter.shape),parameter.dtype)).encode());h.update(parameter.detach().cpu().numpy().tobytes())
    return h.hexdigest()


if __name__=='__main__':main()
