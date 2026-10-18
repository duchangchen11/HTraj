"""Six real audit overlays from explicitly unverified mirror; no semantics GT."""
from common import *
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sdd_tools import parse_annotation,resize_transform,transform_xy,sample_history

def main():
    download=json.loads((OUT/'sdd_sample_download.json').read_text())
    rows=[];points=[]
    for scene in ['bookstore','coupa','deathCircle','gates','hyang','little']:
        assets={r['kind']:r for r in download['rows'] if r['scene']==scene and 'path' in r}
        if len(assets)!=2:
            rows.append(dict(scene=scene,error='Missing annotation/reference pair'));continue
        df=parse_annotation(ROOT/assets['annotations.txt']['path'])
        image=Image.open(ROOT/assets['reference.jpg']['path']).convert('RGB')
        w,h=image.size
        visible=df[df.lost==0]
        inside=(visible.x>=0)&(visible.x<w)&(visible.y>=0)&(visible.y<h)
        actor,frames,xy=sample_history(df)
        nw,nh=max(1,round(w*.4)),max(1,round(h*.4))
        matrix=resize_transform(w,h,nw,nh)
        small=image.resize((nw,nh),resample=Image.Resampling.BOX)
        reduced=transform_xy(xy,matrix)
        recovered=transform_xy(reduced,np.linalg.inv(matrix))
        max_error=float(np.abs(recovered-xy).max())
        assert max_error<1e-9
        fig,ax=plt.subplots(1,3,figsize=(15,6))
        for a,im,p in [(ax[0],image,xy),(ax[1],small,reduced),(ax[2],image,xy)]:
            a.imshow(im);a.plot(p[:,0],p[:,1],'-o',c='magenta',lw=2,ms=4)
            a.scatter(p[0,0],p[0,1],c='cyan',s=35,edgecolors='black',zorder=4)
            a.scatter(p[-1,0],p[-1,1],c='yellow',s=35,edgecolors='black',zorder=4)
            a.set_xlabel('x [pixel]');a.set_ylabel('y [pixel]')
        ax[0].set_title(f'Original {w}x{h}; identity annotation -> reference')
        ax[1].set_title(f'Resized {nw}x{nh}; diag({nw/w:.6f}, {nh/h:.6f}, 1)')
        ax[2].set_xlim(max(0,xy[:,0].min()-90),min(w,xy[:,0].max()+90))
        ax[2].set_ylim(min(h,xy[:,1].max()+90),max(0,xy[:,1].min()-90))
        ax[2].set_title('Raw-pixel crop; cyan=start, yellow=end')
        fig.suptitle(f'{scene}/video0 | track_id={actor} | frames={frames[0]}..{frames[-1]}\nPixel coordinates only; MIRROR PROVENANCE UNVERIFIED; no semantic mask')
        fig.text(.02,.01,'SDD: Robicquet et al., ECCV 2016; CC BY-NC-SA 3.0. Mirror: flclain/StanfordDroneDataset.',fontsize=7)
        fig.tight_layout(rect=[0,.03,1,.9])
        name=OUT/'alignment_figures'/(scene+'_video0.png');name.parent.mkdir(exist_ok=True)
        fig.savefig(name,dpi=130);plt.close(fig)
        for frame,p in zip(frames,xy):points.append(dict(scene=scene,video='video0',frame=int(frame),track_id=actor,x_px=float(p[0]),y_px=float(p[1]),coordinate_system='raw_reference_pixel',source_annotation_SHA=assets['annotations.txt']['sha256']))
        positive_frames=np.diff(np.sort(df.frame.unique()))
        rows.append(dict(scene=scene,video='video0',image_width=w,image_height=h,annotation_rows=len(df),
            annotation_complete_file=assets['annotations.txt']['complete_file'],
            classes=sorted(df.label.unique().tolist()),pedestrian_rows=int((df.label=='Pedestrian').sum()),
            lost_rows=int(df.lost.sum()),occluded_rows=int(df.occluded.sum()),generated_rows=int(df.generated.sum()),
            visible_centers=len(visible),visible_centers_outside=int((~inside).sum()),
            near_boundary_visible_centers=int(((visible.x<10)|(visible.x>=w-10)|(visible.y<10)|(visible.y>=h-10)).sum()),
            raw_frame_gap_min=int(positive_frames.min()),raw_frame_gap_max=int(positive_frames.max()),
            track_id=actor,first_history_frame=int(frames[0]),last_history_frame=int(frames[-1]),
            resize_matrix=matrix.tolist(),inverse_roundtrip_max_px=max_error,
            overlay_path=str(name.relative_to(ROOT)),overlay_SHA=sha(name),
            reference_SHA=assets['reference.jpg']['sha256'],annotation_SHA=assets['annotations.txt']['sha256'],
            original_video_available=False,reference_frame_id_known=False,semantic_alignment_verified=False,
            official_archive_identity_verified=False,geometric_audit_status='PARTIAL'))
    jnew(OUT/'alignment_results.json',dict(cases=rows,real_cases=sum('overlay_path' in r for r in rows),
        exact_original_video_registration_proven=False,semantic_maps_available=False,
        note='Bounds and resize identity are necessary checks, not proof of correct semantic or temporal registration; final visual fixture uses first complete history with >=20px net movement for legibility, not a model group'))
    if points:cnew(OUT/'alignment_history_points.csv',points)
    print('Real audit overlays',sum('overlay_path' in r for r in rows),flush=True)

if __name__=='__main__':main()
