"""Only bytes/config/loader metadata, no forward or optimization before freeze."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import io
import json
import os
import platform
import subprocess
import torch
import numpy as np
import yaml
from src.agentformer_feasibility.common import *
from src.socialvae_baseline.artifacts import json_new

verify_inputs();install_compatibility();workspace()
from data.dataloader import data_generator
from utils.config import Config
original=Path.cwd();os.chdir(WORK)
try:
    cases=[]
    for fold,split in [('hotel','train'),('hotel','val'),('zara2','val')]:
        from easydict import EasyDict
        source=SOURCE/f'cfg/eth_ucy/{fold}/{fold}_agentformer_pre.yml'
        cfg=EasyDict(yaml.safe_load(source.read_text()))
        cfg.data_root_ethucy=str(DATA)
        log=io.StringIO();gen=data_generator(cfg,log,split=split,phase='training' if split=='train' else 'testing')
        assert all(s.endswith('_'+split) for s in gen.sequence_to_load)
        cases.append(dict(fold=fold,split=split,sequences=gen.sequence_to_load,
            candidate_slots=int(gen.num_total_samples),source_file_SHA256=digest(source),loader_log=log.getvalue()))
    pre=Config('hotel_agentformer_pre');dlow=Config('hotel_agentformer')
    a=yaml.safe_load((SOURCE/'cfg/eth_ucy/hotel/hotel_agentformer.yml').read_text())
    b=yaml.safe_load((ROOT/'configs/stage7p4/training/hotel_agentformer.yml').read_text())
    assert {k for k in a if a[k]!=b[k]}=={'pred_epoch'} and b['pred_epoch']==1
    assert digest(ROOT/'configs/stage7p4/training/hotel_agentformer_pre.yml')==digest(SOURCE/'cfg/eth_ucy/hotel/hotel_agentformer_pre.yml')
finally:os.chdir(original)
json_new(OUT/'preflight.json',dict(passed=True,cases=cases,no_forward=True,no_training=True,no_test_metrics=True,
    CVAE_config_byte_identical=True,DLow_only_reference_index_changed=True,
    python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
    GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()))
print('Byte/config/official train and validation loader preflight passed; no model execution',flush=True)
