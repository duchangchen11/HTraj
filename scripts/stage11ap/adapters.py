"""MIT-derived author data adapters only. Frozen model forward/loss untouched.

Source: Human-Path-Prediction/ynet/utils at a345ada1557f53dc2acb4c07f71d340d366980da.
Explicit per-scene arrays replace implicit NumPy ragged arrays. Pandas concat
replaces removed append without changing row/index/augmentation order.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SceneDatasetAdapter(Dataset):
    def __init__(self,data,resize,total_len):
        self.trajectories=[];self.meta=[];self.scene_list=[]
        for scene,df in data.groupby('sceneId',sort=True):
            if len(df)%total_len:raise ValueError('Scene rows not divisible by sequence length')
            counts=df.groupby('metaId',sort=False).size()
            if not (counts==total_len).all():raise ValueError('Incomplete trajectory')
            self.trajectories.append(df[['x','y']].to_numpy().astype('float32').reshape(-1,total_len,2)*resize)
            self.meta.append(df);self.scene_list.append(scene)
    def __len__(self):return len(self.scene_list)
    def __getitem__(self,i):return self.trajectories[i],self.meta[i],self.scene_list[i]

def scene_collate_adapter(batch):
    if len(batch)!=1:raise ValueError('Author scene batching requires batch_size=1')
    return torch.as_tensor(batch[0][0]),[batch[0][1]],batch[0][2]

def augment_data_adapter(data,image_path,original_preprocessing,image_file='reference.jpg',seg_mask=False):
    """Same author transforms/IDs and loop ordering; concat is the only numerical change."""
    import cv2
    from pathlib import Path
    images={}
    for scene in data.sceneId.unique():
        image=cv2.imread(str(Path(image_path)/scene/image_file),0 if seg_mask else 1)
        if image is None:raise FileNotFoundError(scene)
        images[scene]=image
    data_=data.copy(); suffix={1:'_rot90',2:'_rot180',3:'_rot270'}
    for k in [1,2,3]:
        maximum=data.metaId.max()
        for scene in data_.sceneId.unique():
            image=cv2.imread(str(Path(image_path)/scene/image_file),0 if seg_mask else 1)
            rotated,image=original_preprocessing.rot(data_[data_.sceneId==scene],image,k)
            images[scene+suffix[k]]=image
            rotated['sceneId']=scene+suffix[k];rotated['metaId']+=maximum+1
            data=pd.concat([data,rotated],axis=0)
    maximum=data.metaId.max()
    for scene in data.sceneId.unique():
        flipped,image=original_preprocessing.fliplr(data[data.sceneId==scene],images[scene])
        flipped['sceneId']=flipped['sceneId']+'_fliplr';flipped['metaId']+=maximum+1
        data=pd.concat([data,flipped],axis=0);images[scene+'_fliplr']=image
    return data,images

def iter_all_scenes(loader):
    """Explicit full scene traversal for a future HTraj runner; NOT author epoch semantics.

    Does not call train(), loss, backward or optimizer.step(). The official 26-scene
    cap stays frozen; any future full-epoch training schedule must be named separately.
    """
    yield from loader
