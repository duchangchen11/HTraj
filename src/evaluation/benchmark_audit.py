"""History-only interaction groups and paired per-target-window errors."""
import torch
from ..models.ego_encoder import motion_inputs
from ..models.temporal_graph import select_candidates

GROUPS=['all','no_neighbors','low_density','medium_density','high_density','history_full8','history_partial',
        'history1','history2to4','history5to7','opposing','crossing','co_moving','motion_unclassified',
        'relationship_changing','relationship_stable','relationship_insufficient']


def history_features(batch):
    e=motion_inputs(batch['observed'],batch['observed_valid'],batch['timestamps']);b,T,n,_=batch['observed'].shape
    masks=torch.stack([select_candidates(e['positions'][:,t],e['velocity'][:,t],e['continuity'][:,t])[0] for t in range(T)],1)
    degree=masks.sum((1,3))/e['continuity'].sum(1).clamp_min(1)
    # Relation-only categorical motion at cutoff: multiple categories can
    # coexist for one receiver. These groups deliberately overlap.
    v=e['velocity'][:,-1];speed=v.norm(dim=-1)
    cosine=(v[:,:,None]*v[:,None,:]).sum(-1)/(speed[:,:,None]*speed[:,None,:]).clamp_min(1e-6)
    moving=(speed[:,:,None]>.2)&(speed[:,None,:]>.2)&masks[:,-1]
    opposing=(moving & (cosine<-.7)).any(-1)
    co=(moving & (cosine>.7)).any(-1)
    crossing=(moving & (cosine.abs()<=.7)).any(-1)
    # Only adjacent persistent observed pair velocities; insufficient pairs
    # are separate, not quietly labelled stable.
    relative=e['velocity'][:,:, :,None,:]-e['velocity'][:,:,None,:,:]
    pair_step=masks[:,1:]&masks[:,:-1]
    change=(relative[:,1:]-relative[:,:-1]).norm(dim=-1)
    opportunities=pair_step.sum((1,3));score=(change*pair_step).sum((1,3))/opportunities.clamp_min(1)
    cutoff=e['valid'][:,-1];length=e['valid'].sum(1)
    groups=dict(all=cutoff,no_neighbors=cutoff&(degree==0),low_density=cutoff&(degree>0)&(degree<=1),
        medium_density=cutoff&(degree>1)&(degree<=3),high_density=cutoff&(degree>3),history_full8=cutoff&(length==8),
        history_partial=cutoff&(length<8),history1=cutoff&(length==1),history2to4=cutoff&(length>=2)&(length<=4),
        history5to7=cutoff&(length>=5)&(length<=7),opposing=cutoff&opposing,crossing=cutoff&crossing,co_moving=cutoff&co,
        motion_unclassified=cutoff&~(opposing|crossing|co),relationship_changing=cutoff&(opportunities>0)&(score>=.5),
        relationship_stable=cutoff&(opportunities>0)&(score<.5),relationship_insufficient=cutoff&(opportunities==0))
    assert set(groups)==set(GROUPS)
    return groups,dict(history_frames=length,continuous_motion_frames=e['continuity'].sum(1),mean_neighbor_degree=degree,
        relative_velocity_change_mps=score,relation_change_observations=opportunities)


def target_errors(prediction,target,mask):
    error=torch.where(mask[...,None],prediction-target,torch.zeros_like(prediction)).norm(dim=-1)
    count=mask.sum(1);ade=error.sum(1)/count.clamp_min(1)
    return ade,error[:,-1],count>0,mask[:,-1],count
