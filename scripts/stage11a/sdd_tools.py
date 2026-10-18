"""Stage11A independent annotation/coordinate audit, no prediction loader."""
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS=['track_id','xmin','ymin','xmax','ymax','frame','lost','occluded','generated','label']
LABELS={'Pedestrian','Biker','Skater','Cart','Car','Bus'}

def parse_annotation(path):
    # Real raw files have no header. The official YNet loader uses header=0;
    # this independent audit intentionally preserves the first annotation.
    df=pd.read_csv(path,sep=r'\s+',header=None,names=COLUMNS,quotechar='"')
    if df.empty or df.isnull().any().any():raise ValueError('Empty/incomplete annotation rows')
    for key in COLUMNS[:-1]:
        values=df[key].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values,np.round(values)).all():raise ValueError('Noninteger '+key)
        df[key]=values.astype(np.int64)
    if not set(df['label'])<=LABELS:raise ValueError('Unknown label')
    if not all(set(df[k])<={0,1} for k in ('lost','occluded','generated')):raise ValueError('Invalid flag')
    if (df.xmin>df.xmax).any() or (df.ymin>df.ymax).any():raise ValueError('Inverted bounding box')
    if df.duplicated(['track_id','frame']).any():raise ValueError('Duplicate track/frame')
    df['x']=(df.xmin+df.xmax)/2.
    df['y']=(df.ymin+df.ymax)/2.
    return df

def resize_transform(width,height,new_width,new_height):
    if min(width,height,new_width,new_height)<=0:raise ValueError('Invalid dimensions')
    # Continuous annotation coordinate convention: image origin at top-left
    # edge; box center scales by geometric width/height ratio, no homography.
    return np.diag([new_width/width,new_height/height,1.])

def transform_xy(points,matrix):
    points=np.asarray(points,dtype=np.float64)
    h=np.concatenate([points,np.ones((*points.shape[:-1],1))],-1)
    out=h@np.asarray(matrix,dtype=np.float64).T
    if np.any(np.abs(out[...,2])<1e-12):raise ValueError('Point at infinity')
    return out[...,:2]/out[...,2:]

def sample_history(df,step=12,count=8,min_movement_px=20.):
    # A deterministic actual pedestrian track, with eight true rows separated
    # by 12 raw frames. No fabricated/interpolated points added by this audit.
    for actor,g in df[(df.label=='Pedestrian')&(df.lost==0)].groupby('track_id',sort=True):
        g=g.sort_values('frame').drop_duplicates('frame').set_index('frame')
        frame_set=set(g.index)
        for start in g.index[::step]:
            frames=int(start)+np.arange(count)*step
            if set(frames)<=frame_set:
                xy=g.loc[frames,['x','y']].to_numpy(dtype=np.float64)
                if np.linalg.norm(xy[-1]-xy[0])>=min_movement_px:
                    return int(actor),frames,xy
    raise ValueError('No complete eight-point pedestrian history in downloaded prefix')

def semantic_violations(points,class_map,obstacle_classes):
    """Metric utility specification; no real semantic score without label map.

    Outside image is reported separately, never silently clipped onto a valid
    border. Exact boundary pixels are not an assertion of physical safety.
    """
    points=np.asarray(points,dtype=np.float64)
    h,w=class_map.shape
    inbounds=(points[...,0]>=0)&(points[...,0]<w)&(points[...,1]>=0)&(points[...,1]<h)
    values=np.full(inbounds.shape,-1,dtype=int)
    idx=np.floor(points[inbounds]).astype(int)
    values[inbounds]=class_map[idx[:,1],idx[:,0]]
    return dict(outside=~inbounds,obstacle=inbounds&np.isin(values,list(obstacle_classes)))
