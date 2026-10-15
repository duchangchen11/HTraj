"""Official frozen zara2 checkpoint, first64 development windows, no sampling."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,time,subprocess
from collections import defaultdict
import numpy as np
import torch
from src.public_training.data import network_input,permute_graph_input,restore_predictions
from src.public_training.model import make_model,forward
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new


def main():
    torch.set_num_threads(1);p=json.loads(Path('configs/stage4p1/protocol.json').read_text());torch.manual_seed(p['permutation']['seed'])
    checkpoint=Path('data/external/stage4/assets/social_stgcnn/checkpoint/social-stgcnn-zara2/val_best.pth')
    model=make_model('Full').cuda().eval();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'),strict=True)
    data=torch.load(p['cache'],weights_only=True)['validation'][:p['permutation']['windows']]
    rng=np.random.default_rng(p['permutation']['seed']);records=[];layers=defaultdict(list);start=time.perf_counter()
    for sample in data:
        v,a=network_input(sample,'cuda');order=rng.permutation(v.shape[-1]);vv,aa=permute_graph_input(v,a,order)
        hooks=[];outputs={}
        for name in ('st_gcns.0.gcn','st_gcns.0','tpcnns.0','tpcnns.1','tpcnns.2','tpcnns.3','tpcnn_ouput'):
            module=dict(model.named_modules())[name]
            def hook(module,inputs,output,key=name):outputs[key]=(output[0] if isinstance(output,tuple) else output).detach().clone()
            hooks.append(module.register_forward_hook(hook))
        with torch.no_grad():original=forward(model,v,a);baseline=dict(outputs);changed=forward(model,vv,aa)
        for h in hooks:h.remove()
        for name,value in outputs.items():
            recovered=restore_predictions(value,order);difference=(recovered-baseline[name]).abs()
            layers[name].append(dict(window_id=sample['window_id'],max_abs=float(difference.max()),mean_abs=float(difference.mean())))
        changed=restore_predictions(changed.permute(2,0,1)[None],order)[0].permute(1,2,0)
        last=sample['observed'][-1].cuda();truth=sample['future'].cuda()
        original_xy=original[...,:2].cumsum(0)+last;changed_xy=changed[...,:2].cumsum(0)+last
        delta=(changed_xy-original_xy).norm(dim=-1);original_error=(original_xy-truth).norm(dim=-1);changed_error=(changed_xy-truth).norm(dim=-1)
        for j,pid in enumerate(sample['agent_ids']):
            records.append(dict(window_id=sample['window_id'],agent_id=pid,people=len(sample['agent_ids']),
                permutation='|'.join(map(str,order.tolist())),prediction_mean_difference_m=float(delta[:,j].mean()),prediction_max_difference_m=float(delta[:,j].max()),
                original_ADE=float(original_error[:,j].mean()),permuted_ADE=float(changed_error[:,j].mean()),
                original_FDE=float(original_error[-1,j]),permuted_FDE=float(changed_error[-1,j])))
    folder=Path('results/tables/stage4p1');csv_new(folder/'permutation_targets.csv',records)
    layer_rows=[dict(module=name,windows=len(values),maximum_abs=max(r['max_abs'] for r in values),mean_window_abs=float(np.mean([r['mean_abs'] for r in values]))) for name,values in layers.items()]
    csv_new(folder/'permutation_modules.csv',layer_rows)
    grouped=[]
    for people in ['all']+sorted({r['people'] for r in records}):
        rows=records if people=='all' else [r for r in records if r['people']==people]
        grouped.append(dict(people=people,target_windows=len(rows),windows=len({r['window_id'] for r in rows}),
            prediction_mean_difference_m=float(np.mean([r['prediction_mean_difference_m'] for r in rows])),prediction_max_difference_m=max(r['prediction_max_difference_m'] for r in rows),
            original_ADE=float(np.mean([r['original_ADE'] for r in rows])),permuted_ADE=float(np.mean([r['permuted_ADE'] for r in rows])),
            original_FDE=float(np.mean([r['original_FDE'] for r in rows])),permuted_FDE=float(np.mean([r['permuted_FDE'] for r in rows]))))
    csv_new(folder/'permutation_summary.csv',grouped)
    write_json_new(folder/'permutation_execution.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),windows=len(data),targets=len(records),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(__file__),
        protocol_sha256=sha256('configs/stage4p1/protocol.json'),seconds=time.perf_counter()-start,seed=p['permutation']['seed'],
        mode='K1 Gaussian mean, fixed official checkpoint inference only',ETH_evaluated=False,training=False,independent_samples=False))
    print(json.dumps(dict(summary=grouped,modules=layer_rows)))


if __name__=='__main__':main()
