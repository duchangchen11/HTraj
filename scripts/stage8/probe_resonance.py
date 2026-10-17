"""Frozen Resonance original-network probes, not a new model or benchmark run."""
import hashlib
import json
import os
import plistlib
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.agentformer_benchmark.official import DATA,read_official_file,enumerate_windows
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
from contracts import require_validation_file
SOURCE=ROOT/'experiments/stage8/cache/Re'
OUT=ROOT/'results/tables/stage8'
CACHE=ROOT/'experiments/stage8/cache/probe'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_qpid(output_steps='3_7_11'):
    """Provide only annotation metadata, never invoke the author train/test loop."""
    CACHE.mkdir(parents=True,exist_ok=True)
    directory=CACHE/'dataset_configs/Stage8Metadata'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'validation_smoke.plist'
    if not path.exists():
        with path.open('xb') as f:
            plistlib.dump(dict(dataset='Stage8Metadata',type='meter',scale=1.,scale_vis=1.,
                dimension=2,anntype='coordinate',train=[],val=[],test=[]),f)
    os.chdir(CACHE)
    sys.path.insert(0,str(SOURCE))
    import qpid
    import resonance
    from qpid.args import Args
    args=Args(['main.py','--model','re','--split','validation_smoke',
        '--feature_dim','128','--partitions','8','--T','haar','--Tr','haar',
        '--output_pred_steps',output_steps,'--preprocess','100','--gpu','0',
        '--max_agents','73','--K','1','--K_train','1'])
    structure=resonance.ResonanceStructure(args)
    structure.create_model()
    return structure


def scan_validation():
    records=[];arrays={};groups=[]
    for path in sorted((DATA/'hotel').glob('*_val.txt')):
        require_validation_file(path)
        a=read_official_file(path);arrays[path.stem]=a
        positions={(int(f),int(i)):np.array([x,y]) for f,i,x,y in a}
        for w in enumerate_windows(a):
            frame=w['last_observed_frame'];ids=w['target_ids']
            hist=np.array([[positions[t,i] for t in range(frame-7,frame+1)] for i in ids])
            for target,i in enumerate(ids):
                relative=hist[:,-1]-hist[target,-1]
                angle=np.arctan2(relative[:,0],relative[:,1])%(2*np.pi)
                part=np.floor(angle/(2*np.pi/8)).astype(int)
                velocities=(hist[:,-1]-hist[:,0])/(7*.4)
                any_pair=False;pair_count=0
                for p in range(8):
                    indexes=[j for j in range(len(ids)) if j!=target and part[j]==p]
                    for k,j in enumerate(indexes):
                        for l in indexes[k+1:]:
                            distance_gap=abs(np.linalg.norm(relative[j])-np.linalg.norm(relative[l]))
                            velocity_gap=np.linalg.norm(velocities[j]-velocities[l])
                            if distance_gap>=.5 and velocity_gap>=.3:
                                any_pair=True;pair_count+=1
                groups.append(dict(sequence=path.stem,frame=frame,target_id=i,context_people=len(ids),
                    same_sector_position_motion_ambiguity=any_pair,qualifying_pairs=pair_count))
            records.append(dict(sequence=path.stem,path=str(path.relative_to(ROOT)),source_SHA256=sha(path),**w))
    if not records:raise RuntimeError('No official validation windows')
    selected=[]
    for name,condition in [('single',lambda r:r['people']==1),('two',lambda r:r['people']==2),
                           ('multiple',lambda r:r['people']>=5)]:
        matches=[r for r in records if condition(r)]
        if matches:selected.append(dict(case=name,**matches[0]))
    selected.append(dict(case='maximum_validation_context',**max(records,key=lambda r:r['people'])))
    csv_new(OUT/'resonance_history_audit_targets.csv.gz',groups)
    json_new(OUT/'resonance_validation_manifest.json',dict(
        source='AgentFormer official HOTEL fold validation only; not HOTEL test recording',
        protocol='Existing complete20 qualification; adapter for engineering only, not author-loader reproduction',
        rules='Full8/full12 IDs; every context passed, ego included as author neighbor list permits; no neighbor truncation',
        group_thresholds='Same final angular sector of8; absolute radial-distance gap>=0.5m and historical mean velocity-vector gap>=0.3m/s. Fixed before scan. Descriptive exposure only.',
        windows=len(records),target_windows=len(groups),selected=selected,
        sources=[dict(path=str((DATA/'hotel'/(s+'.txt')).relative_to(ROOT)),sha256=sha(DATA/'hotel'/(s+'.txt'))) for s in sorted(arrays)],
        no_test_metrics=True))
    return selected,arrays,groups


def payload(record,a,device):
    frame=record['last_observed_frame'];ids=record['target_ids']
    positions={(int(f),int(i)):np.array([x,y]) for f,i,x,y in a}
    hist=torch.tensor(np.array([[positions[t,i] for t in range(frame-7,frame+1)] for i in ids]),dtype=torch.float32,device=device)
    gt=torch.tensor(np.array([[positions[t,i] for t in range(frame+1,frame+13)] for i in ids]),dtype=torch.float32,device=device)
    # Author NEIGHBOR_TRAJ is already relative to each target's final observed position.
    neighbors=hist[None].repeat(len(ids),1,1,1)-hist[:,-1,None,None,:]
    return hist,neighbors,gt


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(17);np.random.seed(17)
    summary=dict(kind='Original Resonance engineering/structure probes, not trained prediction metrics',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        torch=torch.__version__,numpy=np.__version__,python=sys.version,environment=sys.executable,
        dependency_reuse='New stage8 venv; read-only import path to existing frozen stage4 packages, not author exact torch2.0 environment',
        formal_epochs=0,new_modules=False,test_metrics=False,cases=[])
    try:
        structure=prepare_qpid();model=structure.model
        summary['parameters']=sum(p.numel() for p in model.parameters())
        model.eval()
        # A known invariance: swap centered motion histories between fixed positions in the same sector.
        target=torch.stack([torch.arange(8)*.4,torch.zeros(8)],-1)[None]
        anchors=torch.tensor([[1.,2.],[2.,4.]])
        v=torch.tensor([[.4,0.],[-.4,0.]])
        t=torch.arange(-7,1,dtype=torch.float32)
        nei=anchors[:,None,:]+t[None,:,None]*v[:,None,:]
        swapped=anchors[:,None,:]+t[None,:,None]*v.flip(0)[:,None,:]
        m1,_=model.rc(target,nei[None]);m2,_=model.rc(target,swapped[None])
        summary['pairing_alias']=dict(synthetic=True,position_anchors=anchors.tolist(),
            velocities_m_per_frame=v.tolist(),relative_histories=nei.tolist(),swapped_histories=swapped.tolist(),
            original_matrix_shape=list(m1.shape),max_matrix_difference=float((m1-m2).abs().max()),
            first_candidate_or_ADE_not_computed=True,
            interpretation='An architectural non-identifiability, not proof of predictive harm or a novel remedy')
        selected,arrays,groups=scan_validation()
        summary['history_exposure']=dict(target_windows=len(groups),
            eligible_target_windows=sum(r['same_sector_position_motion_ambiguity'] for r in groups),
            interpretation='Potential ambiguous association contexts; no counterfactual GT or accuracy improvement claim')
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device);summary['device']=device
        if device.startswith('cuda'):
            summary['GPU']=torch.cuda.get_device_name(0)
            summary['GPU_total_bytes']=torch.cuda.get_device_properties(0).total_memory
        for index,record in enumerate(selected):
            hist,neighbors,gt=payload(record,arrays[record['sequence']],device)
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            before=time.monotonic()
            model.eval();torch.manual_seed(8100+index)
            with torch.no_grad():pred=model.implement([hist,neighbors],training=False)[0]
            if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite prediction')
            # Reorder neighbor storage while preserving target inputs and reset identical random noise.
            permutation=torch.arange(neighbors.shape[1]-1,-1,-1,device=device)
            torch.manual_seed(8100+index)
            with torch.no_grad():reordered=model.implement([hist,neighbors[:,permutation]],training=False)[0]
            permutation_difference=float((pred-reordered).abs().max())
            # Target translation and an unrelated future label packet cannot enter input arguments.
            torch.manual_seed(8100+index)
            with torch.no_grad():shifted=model.implement([hist+torch.tensor([9.,-4.],device=device),neighbors],training=False)[0]
            translation_difference=float((shifted-pred-torch.tensor([9.,-4.],device=device)).abs().max())
            model.train();torch.manual_seed(8200+index);model.zero_grad(set_to_none=True)
            outputs=model.implement([hist,neighbors],training=True)
            # Original qpid default l2 loss; append only its declared target loss weight.
            from qpid.training.loss.__ade import ADE_2D
            loss=ADE_2D(outputs[0],gt)
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf')))
            if not torch.isfinite(loss) or not np.isfinite(norm):raise RuntimeError('Nonfinite loss/gradient')
            if device.startswith('cuda'):torch.cuda.synchronize()
            row=dict(case=record['case'],sequence=record['sequence'],frame=record['last_observed_frame'],
                target_ids=record['target_ids'],context_people=record['people'],target_batch=len(hist),
                output_shape=list(pred.shape),finite=True,functional_random_init_loss=float(loss.detach()),gradient_norm=norm,
                neighbor_permutation_max_difference=permutation_difference,
                translation_equivariance_max_difference=translation_difference,
                seconds=time.monotonic()-before)
            if device.startswith('cuda'):
                row.update(allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())
                smi=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                row['whole_card_memory_MiB_snapshot']=smi.stdout.strip()
            summary['cases'].append(row)
        path=CACHE/'random_init_roundtrip.pt'
        torch.save(model.state_dict(),path)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
        summary['checkpoint_restore']=dict(strict=True,path=str(path.relative_to(ROOT)),sha256=sha(path),
                                           author_checkpoint=False)
        summary['optimizer_updates']=0
        summary['passed']=True
    except Exception as exc:
        summary['passed']=False;summary['failure']=str(exc);summary['traceback']=traceback.format_exc()
    summary['total_seconds']=time.monotonic()-started
    json_new(OUT/'resonance_probe.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':run()
