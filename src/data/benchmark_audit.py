"""Explicit complete-target A versus archived partial-mask B views.

Future availability chooses supervised benchmark targets only. Context is
chosen exclusively from history, never from future trajectories/availability.
"""
import copy
import torch
from .dataset import TrajectoryDataset

PROTOCOLS=('A_complete8x12_history_context','B_partial_mask_v1')


def apply_protocol(sample,protocol):
    if protocol not in PROTOCOLS:raise ValueError(protocol)
    out=dict(sample)
    if protocol==PROTOCOLS[0]:
        complete_history=sample['observed_valid'].all(0)
        out['observed_valid']=sample['observed_valid'] & complete_history[None]
        complete_future=sample['future_valid'].all(0)
        out['future_valid']=sample['future_valid'] & (complete_history & complete_future)[None]
    # B is exactly the archived mask and coordinates; model forward still
    # whitelists observations. No coordinate interpolation/normalization.
    return out


class BenchmarkDataset(TrajectoryDataset):
    def __init__(self,raw_root,manifest_path,split,protocol):
        super().__init__(raw_root,manifest_path,split);self.protocol=protocol

    def __getitem__(self,index):return apply_protocol(super().__getitem__(index),self.protocol)


def assert_recording_isolation(manifest):
    train=manifest['splits']['train'];val=manifest['splits']['validation']
    if {w['recording'] for w in train}&{w['recording'] for w in val}:
        raise ValueError('Training/validation recordings overlap')
    if any(w['scene']=='ETH' for w in train+val):raise ValueError('ETH is sealed')
    if any(w['end_frame']-w['start_frame']!=190 for w in train+val):raise ValueError('Incomplete20-frame support')
    return True
