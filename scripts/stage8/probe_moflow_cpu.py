"""CPU-only original MoFlow component admission; not flow sampling or training."""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/stage8/cache/MoFlow'))
sys.path.append(str(ROOT/'experiments/stage7/cache/deps'))
sys.path.append(str(ROOT))
import torch
import yaml
from easydict import EasyDict
from src.socialvae_baseline.artifacts import json_new


class Logger:
    def info(self,*args):pass


def main():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(17)
    row=dict(device='cpu',GPU_seconds=0,formal_training=False,loss_backward=False,
        scope='Original teacher/student backbones, no trained flow/student performance')
    try:
        from models.backbone_eth_ucy import ETHMotionTransformer,ETHIMLETransformer
        cfg=EasyDict(yaml.safe_load((ROOT/'experiments/stage8/cache/MoFlow/cfg/eth_ucy/cor_fm.yml').read_text()))
        row['models']=[]
        for agents in (1,2):
            cfg.agents=agents;cfg.MODEL.CONTEXT_ENCODER.AGENTS=agents
            cfg.objective='set';cfg.num_to_gen=1
            teacher=ETHMotionTransformer(cfg.MODEL,Logger(),cfg).eval()
            student=ETHIMLETransformer(cfg.MODEL,Logger(),cfg).eval()
            x=torch.randn(2,agents,8,6);noise=torch.randn(2,20,agents,24);t=torch.ones(2)*.5
            data={'past_traj':x,'past_traj_original_scale':x}
            with torch.no_grad():out,score=teacher(noise,t,data)
            r=dict(agents=agents,teacher_parameters=sum(p.numel() for p in teacher.parameters()),
                student_parameters=sum(p.numel() for p in student.parameters()),teacher_output_shape=list(out.shape),
                finite=bool(torch.isfinite(out).all()),classification_shape=list(score.shape))
            if agents==2:
                with torch.no_grad():permuted,_=teacher(noise.flip(2),t,{'past_traj':x.flip(1),'past_traj_original_scale':x.flip(1)})
                r['synchronized_agent_permutation_max_difference']=float((out-permuted.flip(2)).abs().max())
                r['scope_note']='N2 is a synthetic fixed-agent configuration, not proof official original A1 dataset is social'
            row['models'].append(r)
        row['passed']=True
    except Exception as exc:row.update(passed=False,error=str(exc),traceback=traceback.format_exc())
    row['seconds']=time.monotonic()-start
    json_new(ROOT/'results/tables/stage8/moflow_cpu_probe.json',row)
    print(json.dumps(row,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
