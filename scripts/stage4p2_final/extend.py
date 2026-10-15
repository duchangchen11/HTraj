"""One frozen continuation31..100 for every existing run, no model changes."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,time
import torch
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new
from src.graph_attribution.model import GraphAttributionModel
from src.graph_attribution.engine import train_epoch,evaluate
from src.evaluation.graph_final_audit import earliest_best


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True);parser.add_argument('--model',required=True,choices=['G0','G1']);args=parser.parse_args()
    protocol=json.loads(Path('configs/stage4p2_final/protocol.json').read_text());assert protocol['extend_to100'] and args.fold in protocol['folds']
    old=json.loads(Path(protocol['old_protocol_path']).read_text());assert sha256(protocol['old_protocol_path'])==protocol['old_protocol_sha256']
    torch.set_num_threads(1);torch.manual_seed(old['seed']);torch.cuda.manual_seed_all(old['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    dirty=subprocess.check_output(['git','status','--porcelain','--','src/graph_attribution','src/evaluation/graph_final_audit.py','scripts/stage4p2_final/extend.py','configs/stage4p2_final/protocol.json'],text=True)
    if dirty:raise RuntimeError('Commit frozen training implementation before continuation')
    assert sha256(old['cache'])==old['cache_sha256'];cache=torch.load(old['cache'],weights_only=False)
    samples={split:[cache['records'][r][i] for r,i in pairs] for split,pairs in cache['folds'][args.fold].items()}
    parent=protocol['extension']['resume'][args.fold+'__'+args.model];assert sha256(parent['path'])==parent['sha256']
    saved=torch.load(parent['path'],weights_only=False);assert saved['epoch']==30
    root=Path('results/tables/stage4p2_final/extension')/args.fold/args.model;ckpt=Path('experiments/stage4p2_final/extension')/args.fold/args.model
    assert not root.exists() and not ckpt.exists();root.mkdir(parents=True);ckpt.mkdir(parents=True)
    model=GraphAttributionModel(args.model,old['width'],old['graph']['gamma']).cuda();model.load_state_dict(saved['model'])
    optimizer=torch.optim.SGD(model.parameters(),lr=old['lr'],momentum=old['momentum'],weight_decay=old['weight_decay']);optimizer.load_state_dict(saved['optimizer'])
    assert optimizer.param_groups[0]['lr']==.01
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    write_json_new(root/'run_config.json',dict(protocol=protocol,original_config=old,parent_checkpoint=parent,source_commit=source,seed=17,
        initial_state_sha256=saved['initial_state_sha256'],script_sha256=sha256(__file__),
        audit_helper_sha256=sha256('src/evaluation/graph_final_audit.py'),
        model_code={str(p):sha256(p) for p in sorted(Path('src/graph_attribution').glob('*.py'))},parameters=sum(p.numel() for p in model.parameters())))
    prefix=Path('results/tables/stage4p2')/args.fold/args.model/'deterministic_scan'
    rows=[json.loads((prefix/f'epoch{epoch:03}.json').read_text()) for epoch in range(1,31)]
    restored,_=evaluate(model,samples['validation'],old['batch_windows'])
    for key in ('NLL','K1_distribution_mean_ADE','K1_distribution_mean_FDE'):assert abs(restored[key]-rows[-1][key])<1e-7
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    write_json_new(root/'restore_check.json',dict(passed=True,epoch=30,metrics=restored,parent_sha256=sha256(parent['path'])))
    start=time.perf_counter()
    try:
        for epoch in range(31,101):
            training=train_epoch(model,samples['train'],optimizer,old,epoch);validation,_=evaluate(model,samples['validation'],old['batch_windows'])
            row=dict(epoch=epoch,**training,**validation);path=ckpt/f'epoch{epoch:03}.pt';assert not path.exists()
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                epoch=epoch,model_kind=args.model,fold=args.fold,source_commit=source,initial_state_sha256=saved['initial_state_sha256'],
                parent_checkpoint_sha256=parent['sha256'],protocol_sha256=sha256('configs/stage4p2_final/protocol.json'),metrics=row),path)
            row.update(checkpoint=str(path),checkpoint_sha256=sha256(path));write_json_new(root/f'epoch{epoch:03}.json',row);rows.append(row)
            print(json.dumps(dict(fold=args.fold,model=args.model,**row)),flush=True)
        csv_new(root/'curve_31to100.csv',rows[30:]);csv_new(root/'curve_1to100.csv',rows)
        write_json_new(root/'summary.json',dict(fold=args.fold,model=args.model,budget=100,completed=True,best=earliest_best(rows),last=rows[-1],best_extension_only=earliest_best(rows[30:]),
            original_budget30_best=earliest_best(rows[:30]),source_commit=source,parent_checkpoint=parent,phase_wall_seconds=time.perf_counter()-start,
            additional_train_seconds=sum(r['train_seconds'] for r in rows[30:]),additional_evaluation_seconds=sum(r['evaluation_seconds'] for r in rows[30:]),
            total_train_seconds=sum(r['train_seconds'] for r in rows),parameters=1553,seed=17))
    except Exception as error:
        write_json_new(root/'failure.json',dict(error=repr(error),last_completed_epoch=rows[-1]['epoch'],source_commit=source,completed=False));raise


if __name__=='__main__':main()
