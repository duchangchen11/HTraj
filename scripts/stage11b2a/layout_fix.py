"""Re-render fixed case IDs after visual QA; no new selection or statistics."""
from .visualize import *
def main():
    cpu_guard();pack,h,y,_=frozen();ids={s:i for i,s in enumerate(pack['sample_ids'])}
    f=pd.read_csv(OUT/'motion_features.csv').set_index('sample_id',drop=False)
    cases=pd.read_csv(OUT/'failure_case_manifest.csv');coords=pd.read_csv(OUT/'failure_case_coordinates.csv',float_precision='round_trip')
    before=sha(OUT/'failure_case_coordinates.csv');records=[]
    backup=ROOT/'experiments/stage11b2a/cache/visual_QA';backup.mkdir(parents=True,exist_ok=True)
    import shutil
    example=FIG/'case_goal_path_mismatch_2_detail.png'
    shutil.copy2(example,backup/'before_caption_overlap.png')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    for c in cases.itertuples():
        r=f.loc[c.sample_id];i=ids[c.sample_id];p=pack['paths'][:,i].numpy();g=pack['goals'][:,i].numpy()
        stored=coords[(coords['case']==c.case)&(coords.kind=='prediction')].sort_values(['head','step'])
        assert np.array_equal(stored[['x','y']].to_numpy(),p.reshape(-1,2))
        image=cv2.cvtColor(cv2.imread(str(RES/'data/SDD/train'/r.scene/'reference.jpg')),cv2.COLOR_BGR2RGB)
        record=(h[i],y[i],p,g,baseline_path(h[i],'CV-Last2'),baseline_path(h[i],'OLS8'),r,image)
        for view in ['full','detail']:
            name='case_'+c.case+'_'+view;oldsha=sha(FIG/(name+'.png'))
            fig,a=plt.subplots(figsize=(8,6.8));display_case(a,record,detail=view=='detail')
            fig.text(.01,.085,f"{c.case}\nFDE1/minFDE20={r.FDE1:.2f}/{r.minFDE20:.2f} px; GoalFDE1/minGoal={r.GoalFDE1:.2f}/{r.minGoalFDE20:.2f} px\nClosest-goal head ADE/FDE={r.closest_goal_ADE:.2f}/{r.closest_goal_FDE:.2f} px (offline GT)",fontsize=7)
            save(fig,name,bottom=.2)
            records.append(dict(figure=name,original_PNG_SHA=oldsha,final_PNG_SHA=sha(FIG/(name+'.png')),
                reason='Reserve footer space; show corresponding closest-goal path explicitly; same fixed IDs and all20 path coordinates'))
    assert sha(OUT/'failure_case_coordinates.csv')==before
    cpu_guard();writej(OUT/'case_layout_QA.json',dict(passed=True,initial_issue='Case caption overlapped x-axis label in first render',
        original_example_retained=str((backup/'before_caption_overlap.png').relative_to(ROOT)),case_coordinate_SHA_unchanged=before,
        sample_selection_unchanged=True,all_cached_paths_exactly_match_CSV=True,figures=records))
    print('Fixed42 case layouts; same21 fixed IDs and exact cached coordinates; no model or metric changes')
if __name__=='__main__':main()
