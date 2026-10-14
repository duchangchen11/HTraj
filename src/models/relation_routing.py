"""Stage3 direct social routing; never changes archived TOHI modules.

Every edge is source -> target. Unknown receiver velocity is explicitly
tagged (legacy feature[6]=2 versus known=1); its CPA is undefined, so ranking
uses present distance. Source continuity guarantees an observed velocity.
Legacy previous-ego payload timing is deliberately preserved, even when
that previous encoding has a missing-velocity flag. No velocity is invented.
"""
import torch
from torch import nn
from .ego_encoder import motion_inputs
from .temporal_graph import GraphFrame, relation_geometry, target_softmax, aggregate
from .order_verification import OrderVerificationModel


def receiver_candidates(encoded, t, radius=4., max_neighbors=8, random=False):
    pos, vel = encoded['positions'][:, t], encoded['velocity'][:, t]
    visible, known = encoded['valid'][:, t], encoded['continuity'][:, t]
    b,n,_ = pos.shape
    r,w,distance,tau,closest = relation_geometry(pos,vel)
    # A placeholder must never enter CPA as an observed zero velocity.
    pair_known = known[:,:,None] & known[:,None,:]
    w = torch.where(pair_known[...,None],w,torch.zeros_like(w))
    tau = torch.where(pair_known,tau,torch.zeros_like(tau))
    closest = torch.where(pair_known,closest,distance)
    candidate = visible[:,:,None] & known[:,None,:] & ~torch.eye(n,device=pos.device,dtype=torch.bool)[None]
    if not random:
        candidate = candidate & (distance <= radius)
    score = closest+.05*distance
    if random:
        # Stateless history-only hash, no agent-index/padding dependence.
        a = pos[:,:,None,:]; z=pos[:,None,:,:]
        score = torch.frac(torch.sin(a[...,0]*12.9898+a[...,1]*78.233+
                          z[...,0]*39.425+z[...,1]*11.135+(t+1)*.618)*43758.5453).abs()
        real,_ = receiver_candidates(encoded,t,radius,max_neighbors,False)
        budget = real.sum(-1)
        ranked = torch.where(candidate,score,torch.full_like(score,torch.inf)).sort(-1).values
        boundary = ranked.gather(-1,(budget-1).clamp_min(0)[...,None])
        inclusive=candidate & (score<=boundary) & (budget>0)[...,None]
        strict=candidate & (score<boundary) & (budget>0)[...,None]
        # Rare identical-coordinate hash ties preserve equivariance; actual
        # edge budget discrepancies are reported rather than index tie-break.
        candidate=torch.where((inclusive.sum(-1)>budget)[...,None],strict,inclusive)
    elif max_neighbors == 0:
        candidate=torch.zeros_like(candidate)
    elif max_neighbors < n:
        ranked=torch.where(candidate,score,torch.full_like(score,torch.inf))
        boundary=ranked.sort(-1).values[...,max_neighbors-1:max_neighbors]
        inclusive=candidate & (score<=boundary); strict=candidate & (score<boundary)
        candidate=torch.where((inclusive.sum(-1)>max_neighbors)[...,None],strict,inclusive)
    marker=torch.where(known[:,:,None],torch.ones_like(distance),torch.full_like(distance,2))
    features=torch.cat([r,w,tau[...,None],closest[...,None],(marker*candidate)[...,None]],-1)
    return candidate,features


def relation_inputs(encoded,t,features,batch,target,source):
    known=encoded['continuity'][:,t]; vel=encoded['velocity'][:,t]
    coverage=encoded['valid'][:,:t+1].sum(1).to(vel.dtype)/(t+1)
    f=features[batch,target,source]
    return torch.cat([f[:,:2],vel[batch,source],f[:,2:6],
                      known[batch,target,None].to(vel.dtype),known[batch,source,None].to(vel.dtype),
                      coverage[batch,target,None],coverage[batch,source,None]],-1)


class ReceiverGraph(nn.Module):
    def __init__(self,original,hidden_dim=64,mode='legacy',relation_dim=24):
        super().__init__()
        if mode not in ('legacy','history','current','no_update','uniform','random'):
            raise ValueError(mode)
        self.mode,self.radius,self.max_neighbors=mode,original.radius,original.max_neighbors
        self.relation_dim=relation_dim
        if mode in ('legacy','uniform'):
            self.relation_score=original.relation_score
        else:
            if mode=='current':
                self.edge_embedding=nn.Sequential(nn.Linear(12,relation_dim),nn.Tanh())
            else:
                self.edge_gru=nn.GRUCell(12,relation_dim)
            self.relation_score=nn.Sequential(nn.Linear(hidden_dim+relation_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))

    def sequence(self,encoded,intervention=None):
        intervention=intervention or {}
        hidden=encoded['hidden']; b,T,n,d=hidden.shape
        memory=hidden.new_zeros(b*n*n,self.relation_dim)
        previous_mask=torch.zeros(b*n*n,dtype=torch.bool,device=hidden.device)
        frames,details=[],[]
        for t in range(T):
            mask,features=receiver_candidates(encoded,t,self.radius,self.max_neighbors,
                random=(self.mode=='random' or intervention.get('random_graph',False)))
            if intervention.get('remove_messages',False):mask=torch.zeros_like(mask)
            batch,target,source=mask.nonzero(as_tuple=True)
            flat_target,flat_source=batch*n+target,batch*n+source
            pair=(batch*n+target)*n+source
            prev_ego=hidden[:,t-1] if t else torch.zeros_like(hidden[:,0])
            edge_features=features[batch,target,source]
            inputs=relation_inputs(encoded,t,features,batch,target,source)
            before=memory[pair]
            # A pair must persist AND both identities must continue. New-entry,
            # disappearance, graph drop, temporal gap and reappearance reset.
            keep=previous_mask[pair] & encoded['continuity'][batch,t,target] & encoded['continuity'][batch,t,source]
            if self.mode=='no_update' or intervention.get('no_update',False):keep=torch.zeros_like(keep)
            before=before*keep[:,None]
            edge_state=hidden.new_zeros(len(pair),self.relation_dim)
            if self.mode in ('history','no_update','random'):
                edge_state=self.edge_gru(inputs,before)
            elif self.mode=='current':edge_state=self.edge_embedding(inputs)
            if self.mode in ('legacy','uniform'):
                scores=self.relation_score(torch.cat([edge_features,hidden[batch,t,target],prev_ego[batch,source]],-1)).squeeze(-1)
            else:scores=self.relation_score(torch.cat([hidden[batch,t,target],edge_state],-1)).squeeze(-1)
            if self.mode=='uniform' or intervention.get('uniform',False):scores=torch.zeros_like(scores)
            weight=target_softmax(scores,flat_target,b*n)
            totals=hidden.new_zeros(b*n,7).index_add(0,flat_target,edge_features)
            degree=hidden.new_zeros(b*n).index_add(0,flat_target,torch.ones_like(weight))
            frames.append(GraphFrame(flat_source,flat_target,weight,edge_features,(totals/degree[:,None].clamp_min(1)).reshape(b,n,7)))
            memory=hidden.new_zeros(b*n*n,self.relation_dim).index_copy(0,pair,edge_state)
            previous_mask=torch.zeros_like(previous_mask).index_fill(0,pair,True)
            details.append(dict(edge_state=edge_state,before=before,inputs=inputs,memory_inherited=keep,
                                target_motion_known=encoded['continuity'][batch,t,target],pair=pair))
        return frames,details


class RelationRouting(OrderVerificationModel):
    def __init__(self,hidden_dim=64,mode='history',**kwargs):
        super().__init__(hidden_dim=hidden_dim,order=1,**kwargs)
        self.graph=ReceiverGraph(self.graph,hidden_dim,mode)
        self.routing_mode=mode

    def forward(self,observed,valid=None,timestamps=None,return_debug=False,intervention=None):
        encoded=self.encoder(observed,valid,timestamps)
        hidden=encoded['hidden']; b,T,n,d=hidden.shape
        frames,relation_details=self.graph.sequence(encoded,intervention)
        state=hidden.new_zeros(b,n,d); support=torch.zeros(b,n,dtype=torch.bool,device=hidden.device)
        states,supports,parts=[],[],[]
        for t,frame in enumerate(frames):
            continuity,visible=encoded['continuity'][:,t],encoded['valid'][:,t]
            before=state*continuity[...,None]; old_support=support & continuity
            prev_ego=hidden[:,t-1] if t else torch.zeros_like(hidden[:,0])
            message=aggregate(frame,self.order_state.projections[0](prev_ego))
            received=torch.zeros(b*n,device=hidden.device,dtype=torch.long).index_add(0,frame.target,torch.ones_like(frame.target)).reshape(b,n)>0
            gamma=torch.sigmoid(self.order_state.gates[0](torch.cat([hidden[:,t],frame.relation_mean],-1)))
            carry=gamma*before
            support=(old_support|received) & visible
            state=torch.where(support[...,None],carry+message,torch.zeros_like(carry))
            states.append(state[:,:,None]);supports.append(support[:,:,None])
            parts.append(dict(before=before[:,:,None],message=message[:,:,None],carry=carry[:,:,None],source=prev_ego[:,:,None]))
        states,supports=torch.stack(states,1),torch.stack(supports,1)
        context,beta,extra=self.fusion(hidden[:,-1],states[:,-1],supports[:,-1],intervention)
        prediction=self.decoder(context,encoded['positions'][:,-1],encoded['valid'][:,-1])
        if return_debug:
            return prediction,dict(**encoded,graph_frames=frames,relation_details=relation_details,order_states=states,
                order_support=supports,order_weights=beta,propagation_parts={k:torch.stack([p[k] for p in parts],1) for k in parts[0]},**extra)
        return prediction


SPECS={'Receiver-Fix':'legacy','Relation-Routing':'history','Routing-Current':'current',
       'Routing-NoUpdate':'no_update','Routing-Uniform':'uniform','Routing-Random':'random'}


def build_routing_model(name,hidden_dim=64):
    return RelationRouting(hidden_dim,mode=SPECS[name])


class HistoricalCV(nn.Module):
    """Last consecutive observed displacement; otherwise stationary fallback."""
    def forward(self,observed,valid=None,timestamps=None,return_debug=False):
        e=motion_inputs(observed,valid,timestamps)
        v=e['velocity'][:,-1]*e['continuity'][:,-1,:,None]
        dt=torch.arange(1,13,device=observed.device,dtype=observed.dtype)*.4
        prediction=e['positions'][:,-1,None]+dt[None,:,None,None]*v[:,None]
        prediction=prediction*e['valid'][:,-1,None,:,None]
        if return_debug:return prediction,dict(**e,hidden=observed.new_zeros(*observed.shape[:3],64))
        return prediction
