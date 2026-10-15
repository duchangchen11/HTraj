"""Read-only official Social-STGCNN test function on four non-ETH folds.

The upstream driver (which globs ETH, opens stored metrics, divides by5) is
not executed. Its test() AST is executed unchanged. Only nx API alias and
evaluation-only no_grad are added, with three fixed Monte Carlo seeds.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy,json,time,random,subprocess,gzip,csv
import numpy as np
import torch
import networkx as nx
from torch.utils.data import DataLoader
import torch.distributions.multivariate_normal as torchdist
from src.evaluation.public_baseline import non_eth_fold,load_args,load_module,only_function,sample_modes
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage4');OFFICIAL=Path('third_party/stage4/social_stgcnn');ASSETS=Path('data/external/stage4/assets/social_stgcnn')


def main():
    p=json.loads(Path('configs/stage4/protocol.json').read_text())
    assert not p['ETH_evaluated'] and not p['training']
    if (ROOT/'social_results.csv').exists():raise FileExistsError('No overwriting experiment')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not hasattr(nx,'from_numpy_matrix'):nx.from_numpy_matrix=nx.from_numpy_array
    utils=load_module('stage4_official_utils',OFFICIAL/'utils.py');metrics=load_module('stage4_official_metrics',OFFICIAL/'metrics.py')
    model_module=load_module('stage4_official_model',OFFICIAL/'model.py')
    namespace=dict(torch=torch,np=np,torchdist=torchdist,copy=copy)
    namespace.update({k:getattr(utils,k) for k in dir(utils) if not k.startswith('_')})
    namespace.update({k:getattr(metrics,k) for k in dir(metrics) if not k.startswith('_')})
    test,fn_sha=only_function(OFFICIAL/'test.py','test',namespace)
    results=[];targets=[];inventory=[]
    for fold in p['social_folds']:
        non_eth_fold(fold);checkpoint_dir=ASSETS/'checkpoint'/('social-stgcnn-'+fold)
        args=load_args(checkpoint_dir/'args.pkl');assert args.dataset==fold
        t0=time.perf_counter()
        dataset=utils.TrajectoryDataset(str(ASSETS/'datasets'/fold/'test'),obs_len=args.obs_seq_len,pred_len=args.pred_seq_len,skip=1,norm_lap_matr=True)
        loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
        load_seconds=time.perf_counter()-t0
        model=model_module.social_stgcnn(n_stgcnn=args.n_stgcnn,n_txpcnn=args.n_txpcnn,output_feat=args.output_size,
            seq_len=args.obs_seq_len,kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda().eval()
        state=torch.load(checkpoint_dir/'val_best.pth',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        parameters=sum(x.numel() for x in model.parameters());namespace.update(model=model,loader_test=loader)
        inventory.append(dict(fold=fold,windows=len(dataset),target_windows=len(dataset.obs_traj),parameters=parameters,max_pedestrians=max(end-start for start,end in dataset.seq_start_end),
            loader_seconds=load_seconds,checkpoint_sha256=sha256(checkpoint_dir/'val_best.pth'),args_sha256=sha256(checkpoint_dir/'args.pkl'),
            architecture={k:getattr(args,k) for k in ('n_stgcnn','n_txpcnn','kernel_size','input_size','output_size','obs_seq_len','pred_seq_len')},
            checkpoint_selection='official supplied val_best; no Stage4 model fitting or selection'))
        for seed in p['monte_carlo_seeds']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
            means=[]
            def hook(module,inputs,output):means.append(output[0].detach().permute(0,2,3,1)[0,:,:,:2].cpu().numpy().copy())
            handle=model.register_forward_hook(hook)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():ad,fd,raw=test(KSTEPS=20)
            torch.cuda.synchronize();seconds=time.perf_counter()-start;handle.remove()
            modes={}
            for index,entry in raw.items():
                mean=metrics.nodes_rel_to_nodes_abs(means[index-1],entry['obs'][-1].copy())
                scores=sample_modes(np.asarray(entry['pred']),entry['trgt'],mean)
                for label,(a,f) in scores.items():
                    modes.setdefault(label,[[],[]]);modes[label][0].extend(a.tolist());modes[label][1].extend(f.tolist())
                    for n in range(len(a)):targets.append(dict(fold=fold,monte_carlo_seed=seed,window_index=index-1,agent_index=n,mode=label,ADE_m=float(a[n]),FDE_m=float(f[n])))
            assert abs(np.mean(modes['K20_independent_min'][0])-ad)<1e-6 and abs(np.mean(modes['K20_independent_min'][1])-fd)<1e-6
            for label,(a,f) in modes.items():
                results.append(dict(model='Social-STGCNN',fold=fold,mode=label,monte_carlo_seed=seed,ADE_m=float(np.mean(a)),FDE_m=float(np.mean(f)),
                    target_windows=len(a),windows=len(dataset),parameters=parameters,evaluation_seconds=seconds,loader_seconds=load_seconds,
                    peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,official_function_numeric_match=True,ETH_evaluated=False,new_training=False))
            print(json.dumps(dict(fold=fold,seed=seed,K20_ADE=ad,K20_FDE=fd,targets=len(modes['K20_independent_min'][0]),seconds=seconds)),flush=True)
            del raw,means
        del model,dataset,loader;torch.cuda.empty_cache()
    csv_new(ROOT/'social_results.csv',results)
    with gzip.open(ROOT/'social_target_errors.csv.gz','xt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(targets[0]),lineterminator='\n');writer.writeheader();writer.writerows(targets)
    write_json_new(ROOT/'social_execution.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protocol_sha256=sha256('configs/stage4/protocol.json'),
        script_sha256=sha256(__file__),official_test_function_sha256=fn_sha,official_sources_sha256=sha256('configs/stage4/official_sources.json'),
        source_hashes={str(f):sha256(f) for f in OFFICIAL.glob('*.py')},inventory=inventory,
        compatibility=['nx.from_numpy_matrix alias to nx.from_numpy_array','only official test function executed; nonETH whitelist driver, no stored metrics.pkl','num_workers1->0; no_grad inference; no numerical architecture/loss change'],
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,networkx=nx.__version__,GPU=torch.cuda.get_device_name(),cuda=torch.version.cuda),
        ETH_evaluated=False,training=False,paper_full_reproduction=False))


if __name__=='__main__':main()
