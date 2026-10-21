"""Extra CPU-only diagnostics without changing any frozen model or metrics."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from .common import *
from .metrics import candidate_metrics
from scripts.stage11b1c.data import truths
import pandas as pd,numpy as np,torch

def main():
    assert not torch.cuda.is_initialized()
    tr=truths('validation');f=pd.read_csv(ROOT/'results/tables/stage11b2a/motion_features.csv');original=set(f.loc[f.GOAL_PATH_MISMATCH,'sample_id']);rows=[];summary=[]
    for ep in [30,35,40,45,50]:
        pp=ROOT/f'experiments/{"stage11b1c" if ep==30 else "stage11b2b"}/cache/predictions/epoch{ep:03d}.pt'
        if not pp.exists():continue
        pack=torch.load(pp,map_location='cpu',weights_only=True);ids=pack['sample_ids'];y=np.stack([tr[s] for s in ids]);d,a,z,ge,gap,m=candidate_metrics(pack['paths'].numpy(),y,pack['goals'].numpy())
        middle=((ge<=10)&(a>=25)&(z<=10)&(d[:,:,:11].mean(-1)>=25)).any(0)
        now=set(np.asarray(ids)[m['GOAL_PATH_MISMATCH']]);summary.append(dict(epoch=ep,original_C7_recovered=len(original-now),original_C7_persistent=len(original&now),new_C_mismatch=len(now-original),current_C_mismatch=len(now),middle_mismatch=int(middle.sum())))
        for i,sid in enumerate(ids):rows.append(dict(epoch=ep,sample_id=sid,original_C7=sid in original,current_mismatch=bool(m['GOAL_PATH_MISMATCH'][i]),middle_mismatch=bool(middle[i])))
    pd.DataFrame(rows).to_csv(OUT/'goal_path_mismatch_id_persistence.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'goal_path_mismatch_persistence.csv',index=False)
    epochs=pd.read_csv(OUT/'epochs.csv');old=pd.read_csv(OLD/'epochs.csv');v=pd.read_csv(OUT/'validation_metrics.csv');cv=v[v.epoch>=30]
    jnew(OUT/'convergence_descriptions.json',dict(epoch30_train_loss=float(old.loc[old.epoch==30,'train_loss'].iloc[0]),last_train_loss=float(epochs.train_loss.iloc[-1]),
        new_first5_train_loss_mean=float(epochs.train_loss.iloc[:5].mean()),new_last5_train_loss_mean=float(epochs.train_loss.iloc[-5:].mean()),
        new_validation_minADE20_values=cv[['epoch','minADE20']].to_dict('records'),new_best_worse_than_epoch30=bool(cv[cv.epoch>30].minADE20.min()>cv[cv.epoch==30].minADE20.iloc[0]),
        scope='Descriptive convergence audit, not proof of sufficient convergence or cause of generalization degradation'))
    print(summary,flush=True)
if __name__=='__main__':main()
