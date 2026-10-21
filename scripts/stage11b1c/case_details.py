"""CPU-only enlarged views of all nine frozen cases, retaining all trajectories."""
from .common import *
import numpy as np,cv2,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    assert not torch.cuda.is_initialized()
    meta=json.loads((OUT/'qualitative_cases.json').read_text())
    coordinates=list(csv.DictReader((OUT/'qualitative_case_coordinates.csv').open()))
    paired={r['sample_id']:r for r in csv.DictReader((OUT/'paired_epoch030.csv').open())}
    view=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    for c in meta['cases']:
        rows=[r for r in coordinates if r['case']==c['case']];m=paired[c['sample_id']]
        def array(kind,head=None):
            r=[r for r in rows if r['kind']==kind and (head is None or int(r['head'])==head)]
            return np.array([[float(z['x']),float(z['y'])] for z in sorted(r,key=lambda r:int(r['step']))])
        allxy=np.array([[float(r['x']),float(r['y'])] for r in rows]);lo=allxy.min(0);hi=allxy.max(0)
        padding=np.maximum((hi-lo)*.1,20);lo-=padding;hi+=padding
        rgb=cv2.cvtColor(cv2.imread(str(RES/'data/SDD/train'/c['scene']/'reference.jpg')),cv2.COLOR_BGR2RGB)
        fig,ax=plt.subplots(figsize=(8,6));ax.imshow(rgb)
        for k in range(20):
            p=array('prediction',k);ax.plot(*p.T,color='#8B9EB7',alpha=.65,lw=1,label='All20 predictions' if k==0 else None)
        for kind,color,label in [('history','#222222','History8'),('GT_future','#D17B61','GT future12')]:
            ax.plot(*array(kind).T,'-o',ms=3,lw=2,color=color,label=label)
        ax.plot(*array('prediction',int(m['best_ADE_head'])).T,color='#267E7C',lw=2.5,label='ADE oracle path')
        ax.scatter(*array('goal').T,marker='x',s=30,color='#6B4C9A',label='All20 sampled goals')
        ax.scatter(*array('GT_future')[-1],marker='*',s=130,color='#F2BB54',edgecolor='black',label='GT endpoint',zorder=10)
        ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]),xlabel='Reference x (pixels)',ylabel='Reference y (pixels)',
            title=f"{c['case']} | {c['sample_id']}\nFixed spatial detail; every path and goal retained")
        ax.legend(fontsize=7,loc='best')
        fig.text(.01,.01,'Y-Net a345ada | HTraj-YNet-SDD-Development-v1 | seed17 | epoch30 | K20 | raw pixels\n'+
            f"ADE1/FDE1={float(m['ADE1']):.3f}/{float(m['FDE1']):.3f} | minADE20/minFDE20={float(m['minADE20']):.3f}/{float(m['minFDE20']):.3f}\n"+
            f"GoalFDE1/minGoalFDE20={float(m['GoalFDE1']):.3f}/{float(m['minGoalFDE20']):.3f} | checkpoint: {c['checkpoint_SHA']}",fontsize=6)
        fig.tight_layout(rect=(0,.11,1,1))
        for ext in ['png','svg']:fig.savefig(FIG/(c['case']+'_detail.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
        view.append(dict(case=c['case'],x_min=float(lo[0]),x_max=float(hi[0]),y_min=float(lo[1]),y_max=float(hi[1]),
            paths=20,goals=20,all_coordinates_inside_view=True,coordinate_changes=False,image_adjustments=False))
    jnew(OUT/'case_detail_viewports.json',dict(rule='Bounding box of all existing history, GT,20 paths and20 goals, plus max(10% span,20px) per axis',cases=view))
    assert not torch.cuda.is_initialized()
    print('All9 fixed cases enlarged without omitting any predicted coordinates; CPU only')

if __name__=='__main__':main()
