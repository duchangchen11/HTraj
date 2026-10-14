"""Stage3 fixed history-only cohorts and measured direct-message diagnostics."""
import torch
from .social_diagnostics import history_groups
from .order_verification import extract_measured_debug,summarize_measured_debug
from .metrics import metric_sums
from ..models.ego_encoder import motion_inputs
from ..models.relation_routing import receiver_candidates
from ..training.smoke import move_batch


def routing_groups(batch):
    groups=history_groups(batch)
    visible=batch['observed_valid'][:,-1]; length=batch['observed_valid'].sum(1)
    encoded=motion_inputs(batch['observed'],batch['observed_valid'],batch['timestamps'])
    mask,_=receiver_candidates(encoded,7)
    neighbor=mask.any(-1); entrant=visible & ~encoded['continuity'][:,-1]
    groups.update(history_full8=visible & (length==8),history_1=visible & (length==1),
        history_2to4=visible & (length>=2) & (length<=4),history_5to7=visible & (length>=5) & (length<=7),
        new_entry_with_neighbors=entrant & neighbor,new_entry_without_neighbors=entrant & ~neighbor,
        one_frame_with_neighbors=visible & (length==1) & neighbor,
        one_frame_without_neighbors=visible & (length==1) & ~neighbor,
        full8_high_density=visible & (length==8) & groups['high_neighbors'],
        full8_low_density=visible & (length==8) & groups['low_neighbors'],
        full8_no_neighbors=visible & (length==8) & groups['no_neighbors'],
        full8_medium_density=visible & (length==8) & groups['medium_neighbors'])
    assert torch.equal(sum(groups[k].long() for k in ('history_full8','history_1','history_2to4','history_5to7')),visible.long())
    return groups


def cache_validation(data,batch_size=8):
    from torch.utils.data import DataLoader
    from ..data.collate import collate_trajectories
    batches=[]
    for batch in DataLoader(data,batch_size=batch_size,collate_fn=collate_trajectories):
        batch['groups']=routing_groups(batch);batches.append(batch)
    return batches


@torch.no_grad()
def evaluate(model,batches,device,full_groups=True,intervention=None,reference=None):
    model.eval(); names=list(batches[0]['groups']) if full_groups else ['all']
    sums={g:dict(ade_sum=0.,ade_count=0,fde_sum=0.,fde_count=0,observed_targets=0,observed_windows=0) for g in names}
    debug_values=[];predictions=[];edges=0;unknown_edges=0;change_sum=0.;change_count=0
    for index,original in enumerate(batches):
        batch=move_batch(original,device)
        options=dict(return_debug=True)
        if intervention:options['intervention']=intervention
        prediction,debug=model(batch['observed'],batch['observed_valid'],batch['timestamps'],**options)
        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite predictions')
        predictions.append(prediction.cpu());values=extract_measured_debug(model,debug)
        for t,frame in enumerate(debug.get('graph_frames',[])):
            edges+=len(frame.source)
            unknown_edges+=int((frame.features[:,-1]==2).sum())
            for key in ('weight',):values.setdefault('relation_alpha',[]).append(getattr(frame,key).detach().cpu())
        for detail in debug.get('relation_details',[]):
            for key in ('edge_state','before'):
                values.setdefault('relation_'+key+'_L2',[]).append(detail[key].norm(dim=-1).detach().cpu())
            values.setdefault('relation_memory_inherited',[]).append(detail['memory_inherited'].float().cpu())
        for k,v in list(values.items()):
            if isinstance(v,list):values[k]=torch.cat(v) if v else torch.tensor([])
        debug_values.append(values)
        for group in names:
            mask=original['groups'][group]; valid=batch['future_valid'] & mask.to(device)[:,None]
            for key,value in metric_sums(prediction,batch['future'],valid).items():sums[group][key]+=value.item()
            sums[group]['observed_targets']+=int(mask.sum());sums[group]['observed_windows']+=int(mask.any(-1).sum())
        if reference is not None:
            delta=(prediction.cpu()-reference[index]).norm(dim=-1);valid=original['observed_valid'][:,-1,None].expand_as(delta)
            change_sum+=float(delta[valid].sum());change_count+=int(valid.sum())
    groups=[dict(group=g,**s,ADE_m=s['ade_sum']/s['ade_count'] if s['ade_count'] else None,
                 FDE_m=s['fde_sum']/s['fde_count'] if s['fde_count'] else None) for g,s in sums.items()]
    diagnostics=summarize_measured_debug(debug_values)
    diagnostics.update(social_messages=edges,unknown_receiver_messages=unknown_edges)
    if reference is not None:diagnostics['prediction_change_L2_m']=change_sum/max(change_count,1)
    return dict(ADE_m=groups[0]['ADE_m'],FDE_m=groups[0]['FDE_m'],groups=groups,diagnostics=diagnostics,predictions=predictions)
