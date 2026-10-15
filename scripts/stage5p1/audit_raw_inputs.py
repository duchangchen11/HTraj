"""Raw-file counterfactuals, including the exact Stage5 entrant example."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy
import json
import subprocess
import numpy as np
import torch
from src.data.preprocessing import Recording,sha256,write_json_new
from src.socialvae_causal.data import raw_example,collate_examples,array_sha,history_context
from src.socialvae_causal.official import load_official,EXPECTED,SOURCE
from src.socialvae_causal.offline_reference import official_examples
from src.socialvae_causal.io import csv_new

OUT=Path('results/tables/stage5p1')
CACHE=Path('experiments/stage5p1/cache/counterfactuals')


def scenario(name):
    rows=[]
    for t in range(20):
        speed=.005 if name=='low_speed' else 0. if name=='stationary' else .1
        rows.append([10*t,0,t*speed,0.])
        if name=='stage5_entrant':
            if t>=7:rows.append([10*t,1,1.7+(t-7)*.2,0.])
        elif name in ['complete_neighbor','boundary_time','stationary','low_speed']:
            rows.append([10*t,1,t*speed+1.,.1])
        elif name=='partial_neighbor':
            if t>=3:rows.append([10*t,1,t*.15+1.,.2])
        elif name=='reappearing_neighbor':
            if t not in [4,5]:rows.append([10*t,1,t*.15+1.,.2])
        elif name=='multi_person':
            rows.append([10*t,1,t*.1+1.,.1])
            if t>=3:rows.append([10*t,2,t*.1+.3,.8])
            if t>=7:rows.append([10*t,3,t*.1-.5,.1])
        elif name!='single_person':raise ValueError(name)
    return rows


def change(rows,variant):
    rows=copy.deepcopy(rows)
    if variant=='target_future':
        for r in rows:
            if r[0]>=80 and r[1]==0:r[2]+=2*(r[0]/10-7);r[3]-=.5
    elif variant=='neighbor_future':
        for r in rows:
            if r[0]>=80 and r[1]!=0:r[2]+=.8*(r[0]/10-7)
    elif variant=='future_only_person':
        rows.extend([[10*t,99,t*.2+.1,.4] for t in range(8,20)])
    elif variant=='extra_prewindow':
        first={int(r[1]):r for r in rows if r[0]==0}
        rows.extend([[-10,actor,r[2]-5.,r[3]+3.] for actor,r in first.items()])
    else:raise ValueError(variant)
    return rows


def write_raw(path,rows):
    with path.open('x') as f:
        for frame,actor,x,y in sorted(rows):f.write(f'{frame} {actor} {x:.9f} {y:.9f}\n')


def offline_fixed_target(path):
    items,keys=official_examples(path)
    i=next(i for i,k in enumerate(keys) if k['start_frame']==0 and k['target_id']==0)
    x,y,n=items[i]
    return dict(x=x,y=y,neighbor=n)


def prediction(model,item):
    x,y,n=collate_examples([item],'cuda')
    with torch.no_grad():p=model(x,n)
    assert torch.isfinite(p).all()
    return p[:,0].cpu().numpy()


def main():
    torch.set_num_threads(1);torch.manual_seed(17)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert not (OUT/'raw_counterfactual_summary.json').exists()
    CACHE.mkdir(parents=True,exist_ok=True)
    ckpt=Path('experiments/stage5/assets/socialvae/models/hotel/ckpt-best')
    model=load_official().SocialVAE(12,2,256).cuda().eval()
    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['model'],strict=True)
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    rows=[];full=[]
    names=['stage5_entrant','complete_neighbor','partial_neighbor','reappearing_neighbor',
           'single_person','multi_person','boundary_time','stationary','low_speed']
    for name in names:
        original=scenario(name);path=CACHE/(name+'_baseline.txt');write_raw(path,original)
        record=Recording.read(path,'synthetic');base=raw_example(record,0,0)
        obase=offline_fixed_target(path);bp=prediction(model,base);op=prediction(model,obase)
        for variant in ['target_future','neighbor_future','future_only_person','extra_prewindow']:
            modified=change(original,variant);file=CACHE/(name+'_'+variant+'.txt');write_raw(file,modified)
            changed=raw_example(Recording.read(file,'synthetic'),0,0)
            ochanged=offline_fixed_target(file);cp=prediction(model,changed);ocp=prediction(model,ochanged)
            equal=np.array_equal(base['x'],changed['x']) and np.array_equal(base['neighbor'],changed['neighbor'])
            equal_ids=np.array_equal(base['context_ids'],changed['context_ids'])
            error=np.linalg.norm(cp-bp,axis=-1);oerror=np.linalg.norm(ocp-op,axis=-1)
            passed=equal and equal_ids and np.array_equal(bp,cp)
            rows.append(dict(scenario=name,intervention=variant,B_input_bitwise_identical=equal,
                B_context_ids_identical=equal_ids,B_prediction_bitwise_identical=np.array_equal(bp,cp),
                B_prediction_mean_difference_m=float(error.mean()),B_prediction_max_difference_m=float(error.max()),
                A_prediction_mean_difference_m=float(oerror.mean()),A_prediction_max_difference_m=float(oerror.max()),
                baseline_raw_sha256=sha256(path),modified_raw_sha256=sha256(file),
                baseline_x_sha256=array_sha(base['x']),modified_x_sha256=array_sha(changed['x']),
                baseline_neighbor_sha256=array_sha(base['neighbor']),modified_neighbor_sha256=array_sha(changed['neighbor']),
                passed=passed))
            full.append(dict(scenario=name,intervention=variant,B_baseline_prediction_m=bp.tolist(),B_modified_prediction_m=cp.tolist(),
                A_baseline_prediction_m=op.tolist(),A_modified_prediction_m=ocp.tolist(),
                target_velocity_valid=base['target_velocity_valid'].tolist(),target_acceleration_valid=base['target_acceleration_valid'].tolist(),
                neighbor_position_valid=base['position_valid'].tolist(),neighbor_velocity_valid=base['velocity_valid'].tolist()))
    assert all(r['passed'] for r in rows),rows
    stage5=next(r for r in rows if r['scenario']=='stage5_entrant' and r['intervention']=='neighbor_future')
    assert abs(stage5['A_prediction_mean_difference_m']-.4874526262)<3e-6
    assert all(torch.equal(frozen[k],v) for k,v in model.state_dict().items())
    # Separate padding diagnostic keeps the same raw sample and weight but
    # changes only padded batching context; tolerate floating summation error.
    single=raw_example(Recording.read(CACHE/'single_person_baseline.txt','synthetic'),0,0)
    multi=raw_example(Recording.read(CACHE/'multi_person_baseline.txt','synthetic'),0,0)
    sx,sy,sn=collate_examples([single],'cuda');mx,my,mn=collate_examples([single,multi],'cuda')
    with torch.no_grad():a=model(sx,sn)[:,0];b=model(mx,mn)[:,0]
    padding_error=float((a-b).abs().max())
    assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)
    csv_new(OUT/'raw_counterfactual_checks.csv',rows)
    write_json_new(OUT/'raw_counterfactual_predictions.json',full)
    write_json_new(OUT/'raw_counterfactual_summary.json',dict(cases=len(rows),scenarios=names,
        passed=all(r['passed'] for r in rows),B_max_prediction_difference_m=max(r['B_prediction_max_difference_m'] for r in rows),
        stage5_original_A_mean_difference_m=stage5['A_prediction_mean_difference_m'],
        stage5_new_B_mean_difference_m=stage5['B_prediction_mean_difference_m'],
        fixed_checkpoint_sha256=sha256(ckpt),weights_unchanged=True,strict_loading=True,
        padding_max_abs_error=padding_error,padding_atol=2e-5,padding_rtol=2e-5,
        official_source_sha256={k:sha256(SOURCE/k) for k in EXPECTED},
        script_sha256=sha256(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        ETH_evaluated=False,training_steps=0,synthetic_not_performance=True))
    print(json.dumps(dict(cases=len(rows),passed=True,stage5_A_mean=stage5['A_prediction_mean_difference_m'],stage5_B_mean=0.,padding_error=padding_error)),flush=True)


if __name__=='__main__':main()
