"""Descriptive trajectory diagnostics. Future-derived tags are never model inputs."""
import numpy as np

DT=.4

def _points(x,n):
 x=np.asarray(x,dtype=np.float64)
 if x.shape!=(n,2) or not np.isfinite(x).all():raise ValueError('Finite complete trajectory required')
 return x

def history_features(history):
 h=_points(history,8);v=np.diff(h,axis=0)/DT;s=np.linalg.norm(v,axis=-1)
 net=h[-1]-h[0];length=float(np.linalg.norm(net));path=float(np.linalg.norm(np.diff(h,axis=0),axis=-1).sum())
 reliable=length>=.2 and s.mean()>=.2
 angle=float(np.degrees(np.arctan2(net[1],net[0]))) if reliable else None
 labels=['E','NE','N','NW','W','SW','S','SE']
 heading=labels[int(np.floor((angle+22.5)%360/45))] if reliable else 'unreliable'
 return dict(history_mean_speed_mps=float(s.mean()),history_last_speed_mps=float(s[-1]),
  history_speed_SD_mps=float(s.std()),history_net_displacement_m=length,history_path_m=path,
  history_directness=length/path if path>0 else 0.,history_heading_deg=angle,history_heading_bin=heading,
  history_heading_reliable=bool(reliable))

def diagnose(history,truth,prediction):
 h=_points(history,8);y=_points(truth,12);p=_points(prediction,12)
 f=history_features(h)
 pv=np.diff(np.vstack([h[-1],p]),axis=0)/DT;gv=np.diff(np.vstack([h[-1],y]),axis=0)/DT
 ps=np.linalg.norm(pv,axis=-1);gs=np.linalg.norm(gv,axis=-1)
 err=p-y;dist=np.linalg.norm(err,axis=-1);pn=p[-1]-h[-1];gn=y[-1]-h[-1]
 pl=float(np.linalg.norm(pn));gl=float(np.linalg.norm(gn))
 angle=None
 if pl>=.2 and gl>=.2:angle=float(np.degrees(np.arccos(np.clip(pn.dot(gn)/(pl*gl),-1,1))))
 sustained=gs.mean()>=.4 and gs[-3:].mean()>=.75*max(gs[:3].mean(),1e-12)
 decel=bool(sustained and ps[:3].mean()>=.2 and ps[-3:].mean()<.7*ps[:3].mean() and ps[-3:].mean()<.7*gs[-3:].mean())
 along=cross=None
 along_step=[None]*12;cross_step=[None]*12
 if f['history_heading_reliable']:
  u=(h[-1]-h[0])/f['history_net_displacement_m'];w=np.array([-u[1],u[0]])
  along_step=(err@u).tolist();cross_step=(err@w).tolist();along=along_step[-1];cross=cross_step[-1]
 out=dict(**f,ADE_m=float(dist.mean()),FDE_m=float(dist[-1]),
  predicted_mean_speed_mps=float(ps.mean()),truth_mean_speed_mps=float(gs.mean()),speed_bias_mps=float((ps-gs).mean()),
  predicted_early_speed_mps=float(ps[:3].mean()),predicted_late_speed_mps=float(ps[-3:].mean()),
  truth_early_speed_mps=float(gs[:3].mean()),truth_late_speed_mps=float(gs[-3:].mean()),
  predicted_net_displacement_m=pl,truth_net_displacement_m=gl,
  endpoint_direction_error_deg=angle,endpoint_along_error_m=along,endpoint_cross_error_m=cross,
  tag_sustained_future=bool(sustained),tag_sustained_but_predicted_deceleration=decel,
  tag_history_stationary_predicted_moving=bool(f['history_mean_speed_mps']<.05 and ps.mean()>=.2),
  tag_future_stationary_predicted_moving=bool(gs.mean()<.05 and ps.mean()>=.2),
  tag_direction_error=bool(angle is not None and angle>=30),
  tag_displacement_overshoot=bool(gl>=.5 and pl>2*gl),
  tag_displacement_undershoot=bool(gl>=.5 and pl<.5*gl))
 steps=[dict(step=k+1,horizon_seconds=DT*(k+1),error_m=float(dist[k]),predicted_speed_mps=float(ps[k]),
  truth_speed_mps=float(gs[k]),speed_bias_mps=float(ps[k]-gs[k]),velocity_vector_error_mps=float(np.linalg.norm(pv[k]-gv[k])),
  along_error_m=along_step[k],cross_error_m=cross_step[k]) for k in range(12)]
 return out,steps
