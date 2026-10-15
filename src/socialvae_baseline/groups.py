"""Predeclared groups from eight observed frames only, no future argument."""
import numpy as np

GROUPS = ['all','speed_stationary','speed_low','speed_normal','speed_fast',
    'turning','not_turning','turn_heading_unreliable','direction_change','direction_stable','direction_unreliable',
    'legal_neighbor_absent','legal_neighbor_present','last_neighbors_0','last_neighbors_1_2','last_neighbors_3_5','last_neighbors_6plus',
    'density_0','density_0_1','density_1_3','density_3plus','new_neighbor_present','new_neighbor_absent',
    'unknown_neighbor_velocity_present','unknown_neighbor_velocity_absent',
    'relative_same','relative_opposing','relative_crossing','relative_heading_unavailable',
    'relative_approaching','relative_departing','relative_radially_stable','relative_radial_unavailable']


def history_features(context,index):
    c=context;others=np.arange(len(c.ids))!=index
    vel=c.state[1:,index,2:4].astype(float)/.4
    speeds=np.linalg.norm(vel,axis=1);speed=float(speeds.mean())
    groups=['all', 'speed_'+('stationary' if speed<.05 else 'low' if speed<.4 else 'normal' if speed<1.8 else 'fast')]
    first,last=vel[:3].mean(0),vel[-3:].mean(0)
    turn=None
    if min(np.linalg.norm(first),np.linalg.norm(last))>=.2:
        turn=float(np.degrees(np.arccos(np.clip(first@last/(np.linalg.norm(first)*np.linalg.norm(last)),-1,1))))
    groups.append('turn_heading_unreliable' if turn is None else 'turning' if turn>=30 else 'not_turning')
    reliable=(speeds[:-1]>=.2)&(speeds[1:]>=.2)
    changes=[]
    for j in np.flatnonzero(reliable):
        changes.append(float(np.degrees(np.arccos(np.clip(vel[j]@vel[j+1]/(speeds[j]*speeds[j+1]),-1,1)))))
    max_change=max(changes) if changes else None
    groups.append('direction_unreliable' if max_change is None else 'direction_change' if max_change>=45 else 'direction_stable')
    dp=c.state[:,others,:2].astype(float)-c.state[:,index:index+1,:2]
    near=c.position_valid[:,others]&(np.linalg.norm(dp,axis=-1)<=2)
    vv=c.velocity_valid[:,others];legal=near[1:]&vv[1:]
    available=bool(near[0].any() or legal.any())
    groups.append('legal_neighbor_present' if available else 'legal_neighbor_absent')
    count=int(legal[-1].sum());density=float(legal.sum()/7)
    groups.append('last_neighbors_'+('0' if count==0 else '1_2' if count<=2 else '3_5' if count<=5 else '6plus'))
    groups.append('density_'+('0' if density==0 else '0_1' if density<=1 else '1_3' if density<=3 else '3plus'))
    first_seen=c.position_valid[:,others].argmax(0)
    entering=bool(((first_seen>0)&near.any(0)).any())
    unknown=bool((near[1:]&~vv[1:]).any())
    groups.extend(['new_neighbor_'+('present' if entering else 'absent'),
                   'unknown_neighbor_velocity_'+('present' if unknown else 'absent')])
    relative=set();radial=set()
    for j in np.flatnonzero(legal[-1]):
        source_v=c.state[-1,others,2:4][j].astype(float)/.4
        sn=float(np.linalg.norm(source_v));tn=speeds[-1]
        if min(sn,tn)>=.2:
            cosine=float(source_v@vel[-1]/(sn*tn))
            relative.add('relative_'+('same' if cosine>=.7 else 'opposing' if cosine<=-.7 else 'crossing'))
        distance=float(np.linalg.norm(dp[-1,j]))
        if distance>1e-8:
            rate=float(dp[-1,j]@(source_v-vel[-1])/distance)
            radial.add('relative_'+('approaching' if rate<-.2 else 'departing' if rate>.2 else 'radially_stable'))
    groups+=sorted(relative or {'relative_heading_unavailable'})
    groups+=sorted(radial or {'relative_radial_unavailable'})
    assert set(groups)<=set(GROUPS)
    return dict(mean_speed_mps=speed,last_speed_mps=float(speeds[-1]),speed_std_mps=float(speeds.std()),
        turn_angle_deg=turn,max_adjacent_heading_change_deg=max_change,last_legal_neighbors=count,
        mean_legal_neighbors=density,new_near_neighbor=entering,unknown_near_velocity=unknown,
        legal_neighbor_available=available,groups='|'.join(groups))
