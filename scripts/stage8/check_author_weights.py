"""Strict load of one predeclared Resonance author checkpoint, no score reading."""
import json
import sys
import time
from pathlib import Path

from probe_resonance import ROOT,OUT,prepare_qpid,sha,payload
import numpy as np
import torch
from src.agentformer_benchmark.official import read_official_file
from src.socialvae_baseline.artifacts import json_new


def main():
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(17)
    weights=ROOT/'experiments/stage8/cache/Re_weights/rezara2'
    args=json.loads((weights/'args.json').read_text())
    # Filename fixed by author release listing; no best score file/event log read.
    cp=weights/'P8_haar_fixed_epoch129.pt'
    before=sha(cp)
    structure=prepare_qpid(args['output_pred_steps']);model=structure.model
    state=torch.load(cp,map_location='cpu',weights_only=True)
    result=model.load_state_dict(state,strict=True)
    model.eval();model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    device=next(model.parameters()).device
    record=json.loads((OUT/'resonance_validation_manifest.json').read_text())['selected'][-1]
    hist,neighbors,gt=payload(record,read_official_file(ROOT/record['path']),device)
    torch.manual_seed(17)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    t=time.monotonic()
    with torch.no_grad():pred=model.implement([hist,neighbors],training=False)[0]
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-t
    target=torch.stack([torch.arange(8)*.4,torch.zeros(8)],-1).to(device)[None]
    anchors=torch.tensor([[1.,2.],[2.,4.]],device=device);v=torch.tensor([[.4,0.],[-.4,0.]],device=device)
    steps=torch.arange(-7,1,dtype=torch.float32,device=device)
    with torch.no_grad():
        a,_=model.rc(target,(anchors[:,None]+steps[None,:,None]*v[:,None])[None])
        b,_=model.rc(target,(anchors[:,None]+steps[None,:,None]*v.flip(0)[:,None])[None])
    # The weight file must remain byte-identical; backward not performed here.
    assert sha(cp)==before
    row=dict(author_repository_commit='42d88ba95e45f441f72318d673c5effd5ef356ec',
        checkpoint=str(cp.relative_to(ROOT)),checkpoint_SHA256=before,author_args=args,
        strict_load=True,missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,
        model_parameter_count=sum(p.numel() for p in model.parameters()),output_shape=list(pred.shape),
        validation_case=dict(sequence=record['sequence'],frame=record['last_observed_frame'],people=len(hist),target_ids=record['target_ids']),
        prediction_finite=bool(torch.isfinite(pred).all()),seconds=elapsed,
        author_weight_pairing_alias_max_difference=float((a-b).abs().max()),
        checkpoint_unchanged=True,source_or_core_changes=False,ADE_FDE_not_computed=True,
        overrides='Annotation/split metadata points only to Stage8 validation adapter; max_agents73 for nontruncating variableN probe, architecture unaffected. All shape-affecting model args match author.',
        earlier_probe='Initial random-init engineering probe used3_7_11 output waypoints, same parameter count but not author exact temporal placement; not a pretrained reproduction',
        validation_scope='Official HOTEL-fold validation with ZARA2 author weights: engineering only, not an out-of-training accuracy claim; no test metric',
        total_seconds=time.monotonic()-start)
    if device.type=='cuda':row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    json_new(OUT/'resonance_author_checkpoint.json',row)
    print(json.dumps(row,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
