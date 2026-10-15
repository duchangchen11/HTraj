"""Immutable epoch checkpoints; a five-epoch gate resumes the same run to30."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,subprocess,time,random,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.graph_attribution.data import pack
from src.graph_attribution.model import GraphAttributionModel
from src.graph_attribution.engine import train_epoch,evaluate


def state_hash(state):
    digest=hashlib.sha256()
    for key,value in state.items():
        digest.update(key.encode());digest.update(str((value.dtype,tuple(value.shape))).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True,choices=['crowds_zara02','biwi_hotel'])
    parser.add_argument('--model',required=True,choices=['G0','G1','G2']);parser.add_argument('--phase',required=True,choices=['smoke','finish'])
    parser.add_argument('--attempt',default='deterministic_scan',choices=['deterministic_scan']);args=parser.parse_args()
    config=json.loads(Path('configs/stage4p2/protocol.json').read_text());torch.set_num_threads(config['cpu_threads'])
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    assert sha256(config['cache'])==config['cache_sha256']
    for manifest in config['manifests'].values():assert sha256(manifest['path'])==manifest['sha256']
    if args.model=='G2':
        decision=json.loads(Path('results/tables/stage4p2/first_round_decision.json').read_text());assert decision['run_G2']
    cache=torch.load(config['cache'],weights_only=False)
    samples={split:[cache['records'][r][i] for r,i in pairs] for split,pairs in cache['folds'][args.fold].items()}
    result=Path('results/tables/stage4p2')/args.fold/args.model/args.attempt;checkpoints=Path('experiments/stage4p2')/args.fold/args.model/args.attempt
    model=GraphAttributionModel(args.model,config['width'],config['graph']['gamma']).cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    initial_hash=state_hash(model.state_dict());source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--','src/graph_attribution','scripts/stage4p2','configs/stage4p2','tests/test_graph_attribution.py'],text=True)
    if dirty:raise RuntimeError('Training code/config must be committed before training: '+dirty)
    start=1;previous=[]
    if args.phase=='smoke':
        assert not result.exists() and not checkpoints.exists();result.mkdir(parents=True);checkpoints.mkdir(parents=True)
        write_json_new(result/'run_config.json',dict(**config,fold=args.fold,model=args.model,attempt=args.attempt,source_commit=source_commit,initial_state_sha256=initial_hash,
            protocol_sha256=sha256('configs/stage4p2/protocol.json'),effective_parameters=sum(p.numel() for p in model.parameters()),
            entrypoint_sha256=sha256(__file__),environment=sys.prefix,python=sys.version,torch=str(torch.__version__),GPU=torch.cuda.get_device_name(),
            code_files={str(p):sha256(p) for p in sorted(Path('src/graph_attribution').glob('*.py'))}))
        end=config['smoke_epochs']
    else:
        gate=json.loads((result/'smoke_gate.json').read_text());assert gate['passed']
        saved=torch.load(checkpoints/'epoch005.pt',weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);start=6;end=config['max_epochs']
        previous=[json.loads((result/f'epoch{epoch:03}.json').read_text()) for epoch in range(1,6)]
        assert saved['initial_state_sha256']==initial_hash and saved['source_commit']==source_commit
    phase_start=time.perf_counter()
    try:
        for epoch in range(start,end+1):
            training=train_epoch(model,samples['train'],optimizer,config,epoch)
            validation,_=evaluate(model,samples['validation'],config['batch_windows'])
            row=dict(epoch=epoch,**training,**validation)
            path=checkpoints/f'epoch{epoch:03}.pt';assert not path.exists()
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                epoch=epoch,model_kind=args.model,fold=args.fold,source_commit=source_commit,initial_state_sha256=initial_hash,
                protocol_sha256=sha256('configs/stage4p2/protocol.json'),metrics=row),path)
            row.update(checkpoint=str(path),checkpoint_sha256=sha256(path));write_json_new(result/f'epoch{epoch:03}.json',row)
            previous.append(row);print(json.dumps(dict(fold=args.fold,model=args.model,**row)),flush=True)
        if args.phase=='smoke':
            probe=pack(samples['validation'][:3],args.model);model.eval()
            with torch.no_grad():expected=model(probe['v'],probe['relation'],probe['valid']).clone()
            restored=GraphAttributionModel(args.model,config['width'],config['graph']['gamma']).cuda().eval()
            saved=torch.load(checkpoints/'epoch005.pt',weights_only=False);restored.load_state_dict(saved['model'])
            with torch.no_grad():actual=restored(probe['v'],probe['relation'],probe['valid'])
            assert torch.equal(expected,actual) and state_hash(restored.state_dict())==state_hash(model.state_dict())
            write_json_new(result/'smoke_gate.json',dict(passed=True,epoch=5,checkpoint=str(checkpoints/'epoch005.pt'),exact_restore=True,
                max_restore_difference=float((actual-expected).abs().max()),seconds=time.perf_counter()-phase_start))
        else:
            csv_new(result/'curves.csv',previous)
            best=min(previous,key=lambda r:(r['K1_distribution_mean_ADE'],r['epoch']));last=previous[-1]
            for label,row in [('best',best),('last',last)]:
                model.load_state_dict(torch.load(row['checkpoint'],weights_only=False)['model'])
                metrics,targets=evaluate(model,samples['validation'],config['batch_windows'],True)
                for mode in ('K1_distribution_mean','K1_first_sample','K20_independent_min'):
                    assert abs(metrics[mode+'_ADE']-row[mode+'_ADE'])<1e-7 and abs(metrics[mode+'_FDE']-row[mode+'_FDE'])<1e-7
                csv_new(result/(label+'_targets.csv'),targets)
            write_json_new(result/'summary.json',dict(model=args.model,fold=args.fold,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                best=best,last=last,min_NLL_epoch=min(previous,key=lambda r:(r['NLL'],r['epoch']))['epoch'],source_commit=source_commit,
                initial_state_sha256=initial_hash,training_seconds=sum(r['train_seconds'] for r in previous),
                evaluation_seconds=sum(r['evaluation_seconds'] for r in previous),phase_seconds=time.perf_counter()-phase_start,
                total_optimizer_steps=sum(r['optimizer_steps'] for r in previous),smoke_passed=True))
    except Exception as error:
        write_json_new(result/(args.phase+'_failure.json'),dict(error=repr(error),completed_epochs=[r['epoch'] for r in previous],source_commit=source_commit))
        raise


if __name__=='__main__':main()
