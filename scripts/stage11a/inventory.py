"""Separate possession, provenance, geometry and semantic readiness."""
from common import *

def main():
    samples=json.loads((OUT/'sdd_sample_download.json').read_text())
    jnew(OUT/'resource_inventory.json',dict(
        local_search_roots=[str(ROOT),'/home/lrj/datasets','/home/lrj/data0109','/home/lrj/Downloads',
            '/home/lrj/trajectory_dataset_audit','/home/lrj/trajectory_research_audit'],
        local_search_method='rg --files --hidden matching SDD/sdd/reference.jpg/annotations.txt/scales, exclude Git and Python environments',
        limitation='Not a claim of exhaustive scan of every mounted volume. No old-project custom model or prediction reused.',
        preexisting_verified_SDD_assets_found=False,
        false_positive_audit='/home/lrj/trajectory_dataset_audit/evidence/sdd_false_positive_audit.json; gate/quad code names are not SDD assets',
        official_original=dict(url='https://cvgl.stanford.edu/projects/uav_data/',locations=8,videos=60,
            scene_counts=dict(gates=9,little=4,nexus=12,coupa=4,bookstore=7,deathCircle=5,quad=4,hyang=15),
            license='CC BY-NC-SA 3.0',archive_advertised_size='approximately69GB',downloaded=False),
        resources=[
            dict(kind='original video',status='UNAVAILABLE',count=0,provenance='not acquired'),
            dict(kind='raw-format annotations',status='PARTIAL',count=6,complete_files=1,prefix_files=5,
                provenance='UNVERIFIED against Stanford archive; mirror commit and available full-file blobs verified',
                training_ready=False),
            dict(kind='reference RGB image',status='PARTIAL',count=6,provenance='UNVERIFIED against Stanford archive; all six mirror Git blobs match',training_ready=False),
            dict(kind='processed official YNet trajectory pool/split/config',status='UNAVAILABLE',count=0,
                provenance='author Google Drive page visible; package bytes/manifest not obtained'),
            dict(kind='PECNet serialized trajectory pools',status='UNVERIFIED',
                provenance='official Git tree advertises binary pickle files; not materialized/unpickled or accepted as raw labelled corpus; test objects not inspected'),
            dict(kind='manual semantic class maps',status='UNAVAILABLE',count=0,provenance='paper describes maps; actual mask files/palette not obtained'),
            dict(kind='semantic network pretrained weights',status='UNAVAILABLE',count=0,provenance='external author bundle not obtained'),
            dict(kind='YNet trained prediction weights',status='UNAVAILABLE',count=0,provenance='external author bundle not obtained'),
            dict(kind='coordinate/world scale or homography',status='UNAVAILABLE',count=0,provenance='no verified per-video world calibration or estimated_scales.yaml'),
            dict(kind='YNet/PECNet official small code',status='AVAILABLE',commit='a345ada1557f53dc2acb4c07f71d340d366980da',
                provenance='26 small code/notebook blobs plus MIT LICENSE verified to exact Git blob SHA1; no modified vendor code')],
        bounded_sample_download_bytes=samples['total_saved_bytes'],sample_budget_bytes=samples['max_download_budget_bytes'],
        large_video_or_author_bundle_downloaded=False,automatic_pseudo_labels=False,
        source_clone_caveat='Initial partial Git metadata/size traversal populated about157MiB object cache including large published binary blobs; switched to no-size tree and bounded raw code fetch. No binary data pool unpickled or predictor weight loaded.',
        Semantic_Drone_Dataset_used=False,ETH_UCY_test_metrics_used=False))
    # Per-case numerical coordinate evidence only, not registration accuracy.
    import numpy as np
    cases=json.loads((OUT/'alignment_results.json').read_text())['cases']
    points=cread(OUT/'alignment_history_points.csv')
    rows=[]
    for c in cases:
        p=np.array([[float(r['x_px']),float(r['y_px'])] for r in points if r['scene']==c['scene']])
        m=np.array(c['resize_matrix'])
        diff=np.abs(p*(np.array([m[0,0],m[1,1]])-.4)).max()
        rows.append(dict(scene=c['scene'],video=c['video'],visual_resize_x=m[0,0],visual_resize_y=m[1,1],
            scalar_author_style_resize=.4,max_difference_vs_exact_rounded_dimension_ratio_px=float(diff),
            note='Pixel-edge geometric convention vs scalar resize; subpixel rounding must be documented, not proof of original registration'))
    cnew(OUT/'resize_convention_audit.csv',rows)

if __name__=='__main__':main()
