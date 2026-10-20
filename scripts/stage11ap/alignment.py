"""Six author TRAIN examples; GT is offline alignment only, never predictor input."""
from common import *
from safe_data import load_data_pickle
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
RES=CACHE/'author_resources/ynet_additional_files'

def main():
    df,_=load_data_pickle(RES/'data/SDD/train_trajnet.pkl')
    val=json.loads((OUT/'split_identity_audit.json').read_text())['candidate_val_scenes']
    scenes=[sorted(s for s in df.sceneId.unique() if s.rsplit('_',1)[0]==site and s not in val)[0] for site in sorted({s.rsplit('_',1)[0] for s in df.sceneId.unique()})]
    records=[];points=[];folder=OUT/'alignment_figures';folder.mkdir(exist_ok=True)
    cmap=ListedColormap(['#cccccc','#ffa947','#6fa8dc','#b366d9','#94bf5c','#c8665f'])
    for scene in scenes:
        g=next(iter(df[df.sceneId==scene].groupby('metaId',sort=True)))[1]
        rp=RES/'data/SDD/train'/scene/'reference.jpg'
        mp=RES/'data/SDD_semantic_maps/train_masks'/f'{scene}_mask.png'
        ref=np.asarray(Image.open(rp));mask=np.asarray(Image.open(mp))
        assert ref.shape[:2]==mask.shape
        xy=g[['x','y']].to_numpy();h,w=mask.shape
        hist=xy[:8];future=xy[8:]
        fig,axs=plt.subplots(1,3,figsize=(15,6),layout='constrained')
        axs[0].imshow(ref);axs[0].set_title('Author reference RGB + observed history')
        axs[1].imshow(mask,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,6.5),6));axs[1].set_title('Author label IDs (name mapping unverified)')
        axs[2].imshow(ref);axs[2].imshow(mask,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,6.5),6),alpha=.35);axs[2].set_title('Overlay: observed + TRAIN GT (alignment only)')
        for ax in axs:
            ax.plot(hist[:,0],hist[:,1],'o-',color='#00a6ff',ms=3,lw=1.5,label='8 observations')
            ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_aspect('equal')
            ax.set_xlabel('raw reference x [pixel]');ax.set_ylabel('raw reference y [pixel]')
        axs[1].plot(future[:,0],future[:,1],'o--',color='#ff2250',ms=2,lw=1,label='12 train GT')
        axs[2].plot(np.r_[hist[-1,0],future[:,0]],np.r_[hist[-1,1],future[:,1]],'o--',color='#ff2250',ms=2,lw=1,label='12 train GT')
        axs[2].legend(loc='upper left',fontsize=7)
        track=int(g.trackId.iloc[0]);frames=g.frame.to_numpy(dtype=int)
        fig.suptitle(f'{scene} | track={track} meta={int(g.metaId.iloc[0])} | frames={frames[0]}..{frames[-1]} step12 | {w}x{h}\n'
                     'Coordinate convention: raw pixel identity (RGB/mask); inference x0.25 + bottom/right pad; no world calibration',fontsize=10)
        p=folder/f'{scene}.png';assert not p.exists();fig.savefig(p,dpi=130);plt.close(fig)
        records.append(dict(scene=scene,video=scene.rsplit('_',1)[1],track_id=track,meta_id=int(g.metaId.iloc[0]),frames=frames.tolist(),
                            reference_SHA=sha(rp),semantic_SHA=sha(mp),figure=str(p.relative_to(ROOT)),figure_SHA=sha(p),
                            image_size=[w,h],mask_size=[w,h],transform_RGB_to_semantic=[[1,0,0],[0,1,0],[0,0,1]],
                            inference_nominal_transform=[[.25,0,0],[0,.25,0],[0,0,1]],GT_used_only_for_alignment=True,
                            precise_original_video_registration='UNVERIFIED: original video and reference frame ID not included'))
        for i,(f,(x,y)) in enumerate(zip(frames,xy)):
            points.append(dict(scene=scene,person_id=track,frame_id=int(f),phase='history' if i<8 else 'train_GT_alignment_only',x=float(x),y=float(y)))
    cnew(OUT/'alignment_points.csv',points)
    jnew(OUT/'alignment_cases.json',dict(real_cases=len(records),test_cases=0,cases=records,semantic_names_verified=False,
                                       all_reference_mask_dimensions_match=True,raw_video_registration_proven=False))
    (folder/'LICENSE_CC_BY_NC_SA_3.0.md').write_text('SDD原始参考图及作者语义图的派生对齐图，按SDD官网CC BY-NC-SA3.0标注。\n来源：Stanford Drone Dataset (Robicquet et al., ECCV2016), https://cvgl.stanford.edu/projects/uav_data/；Y-Net作者补充包，Drive ID1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa。\n本轮修改：叠加作者训练池轨迹和标签ID着色；不是预测性能图，不将此许可扩展到全仓库源码。\n')
    print('Real train RGB/mask/history/future alignment figures:',len(records))
if __name__=='__main__':main()
