"""Hard commit gate, validation-only author diagnosis, then two full epochs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import os
import shutil
import time
import traceback
from unittest.mock import patch
import torch
from src.agentformer_feasibility.common import *
from src.socialvae_baseline.artifacts import json_new


def main():
    commit=freeze_gate();verified=verify_inputs()
    assert shutil.disk_usage(ROOT).free>=protocol()['stage_minimum_free_bytes']
    json_new(OUT/'execution_start.json',dict(freeze_commit=commit,git_clean=True,inputs=verified,
        start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_evaluation_forbidden=True))
    install_compatibility();torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.cuda.set_device(0)
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
    from scripts.stage7p4.validation import run as validation
    from scripts.stage7p4.train_epoch import run as train
    completed=[]
    try:
        validation();completed.append('official validation: HOTEL and ZARA2')
        workspace();os.chdir(WORK)
        loads=[]
        original_load=torch.load
        def guarded_load(path,*args,**kwargs):
            resolved=Path(path).resolve()
            assert resolved.is_relative_to(WORK),f'Forbidden pretrained/smoke load in training: {resolved}'
            loads.append(dict(path=str(resolved),sha256=digest(resolved)))
            return original_load(path,*args,**kwargs)
        for stage in ['CVAE','DLow']:
            print('START ONE FULL',stage,'EPOCH',flush=True)
            with patch.object(torch,'load',guarded_load):row=train(stage)
            json_new(OUT/f'{stage}_checkpoint_load_audit.json',dict(loads=list(loads),all_paths_within_new_training_workspace=True,author_or_smoke_weights_loaded=False))
            if not row['success']:
                json_new(OUT/'execution_completion.json',dict(success=False,completed=completed,stopped_at=stage,reason='No retry after failed full epoch/restore',test_metrics_recomputed=False))
                return
            completed.append(stage)
            print('COMPLETE',stage,row['optimizer_updates'],'epoch updates;',row['resource']['seconds'],'seconds',flush=True)
        json_new(OUT/'execution_completion.json',dict(success=True,completed=completed,
            full_epochs=2,engineering_resume_probe_updates=2,test_metrics_recomputed=False))
    except Exception:
        json_new(OUT/'execution_failure.json',dict(traceback=traceback.format_exc(),completed=completed,
            stopped=True,no_retry=True,test_metrics_recomputed=False))
        print(traceback.format_exc(),flush=True)
    finally:os.chdir(ROOT)


if __name__=='__main__':main()
