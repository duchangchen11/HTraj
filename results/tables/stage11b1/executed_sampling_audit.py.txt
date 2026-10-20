from scripts.stage11b1.common import *
from scripts.stage11b1.data import load,manifest
from scripts.stage11b1.evaluation import Predictor,history_record
from scripts.stage11b0.runtime import build_model,original_modules,state_digest
from scripts.stage11b0.training_adapter import make_templates,preprocess_scene,fixed_semantics
import numpy as np,torch
configure_torch();net,cfg=build_model(17,'cuda');templates=make_templates(cfg,'cuda');model=Predictor(net,cfg,templates)
r=json.loads((OUT/'precheck_selected_targets.json').read_text())['targets'][0]
df=load('train');g=df[df.metaId==int(r['meta_id'])];xy=g[['x','y']].to_numpy(dtype='float32').reshape(20,2)
raw=dict(sample_id=r['sample_id'],xy=xy);sid,h=history_record(raw)
image=preprocess_scene(r['scene'],cfg,'cuda');semantic=fixed_semantics(net,image)
module=original_modules().image;original=module.sampling
captures=[]
def spy(prob,*args,**kw):
    prob0=prob.clone();p=original(prob,*args,**kw);captures.append((prob0.detach().cpu(),p.detach().cpu()));return p
module.sampling=spy
rows=[];results=[]
for i in range(5):
    p,g,t=model.predict(sid,h,semantic=semantic)
    results.append((p,g));prob,samp=captures[-1]
    rows.append(dict(iteration=i,history_SHA=state_digest({'history':h}),prob_SHA=state_digest({'p':prob}),proposal_SHA=state_digest({'s':samp}),path_SHA=state_digest({'p':p}),goal_SHA=state_digest({'g':g}),max_path_diff_first=float((p-results[0][0]).abs().max()),max_goal_diff_first=float((g-results[0][1]).abs().max()),torch_deterministic=torch.are_deterministic_algorithms_enabled()))
    print(rows[-1],flush=True)
module.sampling=original
jnew(OUT/'repeat_probe.json',dict(train_only=True,no_GT_used=True,rows=rows))
