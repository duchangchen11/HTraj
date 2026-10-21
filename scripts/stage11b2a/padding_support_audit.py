"""EXPLORATORY geometric support check, no model or segmentation forward."""
from .common import *
import cv2,numpy as np,pandas as pd,yaml
from scripts.stage11b0.common import RES
def main():
    cpu_guard();cfg=yaml.safe_load((RES/'config/sdd_trajnet.yaml').read_text());assert cfg['resize']==.25
    division=2**len(cfg['encoder_channels']);assert division==32
    points=pd.read_csv(OUT/'geometric_out_of_image_points.csv',float_precision='round_trip');rows=[];sizes={}
    for scene in points.scene.unique():
        path=RES/'data/SDD/train'/scene/'reference.jpg';im=cv2.imread(str(path));assert im is not None
        # Exact author cv2.resize geometry; only its resulting shape is needed.
        resized=cv2.resize(im,(0,0),fx=.25,fy=.25,interpolation=cv2.INTER_AREA);h,w=resized.shape[:2]
        sizes[scene]=(h,w,int(np.ceil(h/32)*32),int(np.ceil(w/32)*32))
    for r in points.itertuples():
        h,w,hp,wp=sizes[r.scene];x,y=r.x*.25,r.y*.25
        rows.append(dict(sample_id=r.sample_id,scene=r.scene,head=r.head,future_step=r.future_step,x=r.x,y=r.y,
            resized_height=h,resized_width=w,padded_height=hp,padded_width=wp,
            within_decoder_canvas=bool(0<=x<=wp-1 and 0<=y<=hp-1),
            beyond_resized_pixel_centers=bool(x>w-1 or y>h-1),
            side=';'.join(k for k,z in [('left',r.x<0),('right',r.x>=r.width),('top',r.y<0),('bottom',r.y>=r.height)] if z)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'EXPLORATORY_padding_support.csv',index=False)
    summary=dict(analysis_class='EXPLORATORY_PADDING_SUPPORT_AUDIT',new_model_calls=0,GPU_seconds=0,points=len(rows),
        points_inside_padded_decoder_canvas=int(result.within_decoder_canvas.sum()),
        points_beyond_resized_original_pixel_centers=int(result.beyond_resized_pixel_centers.sum()),
        source_image_utils_SHA=sha(ROOT/'experiments/stage11b0/cache/official/ynet/utils/image_utils.py'),
        source_adapter_SHA=sha(ROOT/'scripts/stage11b0/training_adapter.py'),
        interpretation='Spatial support is compatible with right/bottom padding. This does not causally prove padding caused predictions or exclude semantic/model effects; no output clipping or metric changes.')
    writej(OUT/'EXPLORATORY_padding_support.json',summary)
    text=f'''\n## EXPLORATORY：参考边界与解码画布支持\n\n额外静态/CPU几何核查：作者image_utils.py的resize使用cv2.resize(fx=fy=.25)，pad补到32的倍数；路径softargmax解码在padding后的完整画布内。51个原图越界点中，{summary['points_inside_padded_decoder_canvas']}点仍处于原网络合法padding画布范围，{summary['points_beyond_resized_original_pixel_centers']}点超出缩放后原图像素中心范围。逐点证据见EXPLORATORY_padding_support.csv。此现象与padding提供的额外坐标支持相容，不能据此证明padding是成因，也不能称NaN/数值爆炸或语义碰撞。本轮没有裁剪、重算预测或改动指标。该解释性扩展明确标为EXPLORATORY，未改变冻结分类阈值。\n'''
    with (REPORT/'06_FAILURE_TAXONOMY.md').open('a') as f:f.write(text)
    cpu_guard();print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
