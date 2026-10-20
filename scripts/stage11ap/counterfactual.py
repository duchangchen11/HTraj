"""One train sample, fixed noise: changing future labels cannot change predictor outputs."""
from common import *
from pipeline_admission import predicted_paths,RES
from checkpoint_io import load_author_segmentation
from safe_data import load_data_pickle
import torch,numpy as np,yaml,cv2

def main():
    t=time.perf_counter();torch.set_num_threads(1)
    sys.path.insert(0,str(CACHE/'official/ynet'))
    from model import YNetTorch
    cfg=yaml.safe_load((RES/'config/sdd_trajnet.yaml').read_text())
    seg,err=load_author_segmentation(RES/cfg['segmentation_model_fp'])
    m=YNetTorch(8,12,None,False,6,cfg['encoder_channels'],cfg['decoder_channels'],1)
    m.semantic_segmentation=seg
    m.load_state_dict(torch.load(RES/'pretrained_models/sdd_trajnet_weights.pt',map_location='cpu',weights_only=True),strict=True)
    m=m.eval().cuda();before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    data,_=load_data_pickle(RES/'data/SDD/train_trajnet.pkl')
    g=data[data.metaId==data.metaId.iloc[0]]
    original=g[['x','y']].to_numpy(dtype=np.float32)
    mutated=original.copy();mutated[8:]=1e6  # train GT altered outside prediction input
    scene=g.sceneId.iloc[0];image=cv2.imread(str(RES/'data/SDD/train'/scene/'reference.jpg'))
    def run(x):
        torch.manual_seed(113);np.random.seed(113)
        return predicted_paths(m,image,x[:8].copy()[None],cfg,torch.device('cuda:0'))[:2]
    first=run(original);second=run(mutated)
    assert all(torch.equal(a,b) for a,b in zip(first,second))
    assert all(torch.equal(before[k],v.detach().cpu()) for k,v in m.state_dict().items())
    log=dict(passed=True,scene=scene,meta_id=int(g.metaId.iloc[0]),future_only_mutation=True,
             future_values_sent_to_model=False,first_and_second_goals_identical=True,first_and_second_K20_paths_identical=True,
             max_abs_prediction_difference=float((first[0]-second[0]).abs().max()),weights_unchanged=True,optimizer_updates=0,
             future_mutation_used_for_metrics=False,torch_seed=113,numpy_seed=113,wall_seconds=time.perf_counter()-t)
    jnew(OUT/'real_train_future_isolation.json',log)
    print(json.dumps(log,indent=2))
if __name__=='__main__':main()
