"""Independent minimal case, calling original frozen functions without patches."""
from .common import *
from scripts.stage11b1.data import load
from scripts.stage11b1.evaluation import Predictor, sample_seed
from scripts.stage11b0.runtime import build_model,original_modules,seed_all,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics,trajectory_maps
import torch,numpy as np,random
import torch.nn.functional as F

def rng_hashes():
    n=np.random.get_state()
    return dict(torch_CPU=state_digest({'s':torch.get_rng_state()}),
                torch_CUDA=state_digest({'s':torch.cuda.get_rng_state()}),
                numpy=hashlib.sha256(repr((n[0],n[1].tolist(),n[2:])).encode()).hexdigest(),
                python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())

def main(guard):
    configure_torch();assert torch.cuda.is_available()
    net,cfg=build_model(17,'cuda:0');net.eval();templates=make_templates(cfg,'cuda:0')
    rows=json.loads((ROOT/'results/tables/stage11b1/precheck_selected_targets.json').read_text())['targets']
    r=rows[0];df=load('train');g=df[df.metaId==int(r['meta_id'])]
    history=torch.as_tensor(g[['x','y']].iloc[:8].to_numpy(dtype='float32')).unsqueeze(0)
    image=preprocess_scene(r['scene'],cfg,'cuda:0');semantic=fixed_semantics(net,image)
    H,W=semantic.shape[-2:];km=original_modules().model.evaluate.__globals__['kmeans']
    with torch.no_grad():
        observed=trajectory_maps(history.cuda()*.25,H,W,templates[0])
        features=net.pred_features(torch.cat([semantic,observed],1))
        logits=net.pred_goal(features)[:,[11]];prob=net.sigmoid(logits/1.)
    digest=lambda t:state_digest({'tensor':t})
    records=[];proposals=[];centers=[];decoded=[]
    for i in range(5):
        guard();seed_all(sample_seed(r['sample_id']));before=rng_hashes();t=time.perf_counter()
        torch.use_deterministic_algorithms(False)
        try:
            p=original_modules().image.sampling(prob,10000,replacement=True,rel_threshold=.01)
        finally:torch.use_deterministic_algorithms(True)
        torch.cuda.synchronize();proposals.append(p.detach().cpu())
        records.append(dict(component='original_GPU_multinomial',repeat=i,probability_SHA=digest(prob),
             proposal_SHA=digest(p),seconds=time.perf_counter()-t,max_coordinate_diff=float((p.cpu()-proposals[0]).abs().max()),
             RNG_before=before,RNG_after=rng_hashes()))
    # Same proposals and RNG, so any new center differences cannot be blamed on input sampling.
    fixed=proposals[0][0,0].cuda()
    for i in range(5):
        guard();seed_all(sample_seed(r['sample_id']));before=rng_hashes();t=time.perf_counter()
        _,c=km(X=fixed,num_clusters=19,distance='euclidean',device=fixed.device,tqdm_flag=False,tol=.001,iter_limit=1000)
        torch.cuda.synchronize();centers.append(c.cpu())
        records.append(dict(component='original_KMeans_fixed_proposals',repeat=i,goal_SHA=digest(c),
             seconds=time.perf_counter()-t,max_coordinate_diff=float((c.cpu()-centers[0]).abs().max()),RNG_before=before,RNG_after=rng_hashes()))
    fixed_goals=torch.cat([net.softargmax(logits).unsqueeze(0),centers[0].cuda()[:,None,None,:]],0)
    for i in range(5):
        guard();out=[];t=time.perf_counter()
        with torch.no_grad():
            for goal in fixed_goals:
                wm=trajectory_maps(goal,H,W,templates[0])
                pyramid=[wm]+[F.avg_pool2d(wm,2**j,2**j) for j in range(1,len(features))]
                out.append(net.softargmax(net.pred_traj([torch.cat([f,w],1) for f,w in zip(features,pyramid)])))
        p=torch.stack(out)/.25;torch.cuda.synchronize();decoded.append(p.cpu())
        records.append(dict(component='original_decoder_fixed_goals',repeat=i,path_SHA=digest(p),
             seconds=time.perf_counter()-t,max_coordinate_diff=float((p.cpu()-decoded[0]).abs().max())))
    old=Predictor(net,cfg,templates);full=[]
    for i in range(5):
        guard();p,goals,timing=old.predict(r['sample_id'],history,semantic=semantic)
        full.append((p,goals))
        records.append(dict(component='old_full_K20',repeat=i,path_SHA=digest(p),goal_SHA=digest(goals),
            max_path_diff=float((p-full[0][0]).abs().max()),max_goal_diff=float((goals-full[0][1]).abs().max()),**timing))
    artifact=CACHE/'minimal_original_case.pt'
    torch.save(dict(sample_id=r['sample_id'],history=history,probability=prob.detach().cpu(),
                    original_proposals=proposals,original_KMeans_centers=centers,original_fixed_goal_paths=decoded,
                    old_full_paths=[p for p,g in full],old_full_goals=[g for p,g in full]),artifact)
    jnew(OUT/'minimal_original_case.json',dict(sample_id=r['sample_id'],scene=r['scene'],records=records,
        artifact_path=str(artifact.relative_to(ROOT)),artifact_SHA=sha(artifact),
        probability_SHA=digest(prob),source_hist_SHA=r['history_SHA'],source_GT_never_used=True,
        gpu_sampling_repeated_bitwise=len({digest(p) for p in proposals})==1,
        fixed_proposal_KMeans_bitwise=len({digest(c) for c in centers})==1,
        fixed_goal_decoder_bitwise=len({digest(p) for p in decoded})==1,
        full_goal_bitwise=len({digest(g) for p,g in full})==1,
        official_source_changed=False,monkeypatch_used=False))
    print('Original minimal case saved; proposals bitwise:',len({digest(p) for p in proposals})==1,
          'KMeans fixed-input:',len({digest(c) for c in centers})==1,
          'decoder fixed-input:',len({digest(p) for p in decoded})==1,flush=True)

if __name__=='__main__':timed_run('diagnose_original',main)
