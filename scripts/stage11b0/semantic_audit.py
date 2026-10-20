from .common import *
import requests,zipfile,io,re
def main():
    url='https://www.openaccess.thecvf.com/content/ICCV2021/supplemental/Mangalam_From_Goals_Waypoints_ICCV_2021_supplemental.zip'
    log=dict(primary_page='https://openaccess.thecvf.com/content/ICCV2021/html/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.html',
             supplemental_link_verified_in_author_CVF_page=True,
             prior_canonical_ZIP_attempt='SSLError UNEXPECTED_EOF_WHILE_READING; no verified PDF saved',attempts=[])
    try:
        r=requests.get(url,stream=True,timeout=20);entry=dict(url=url,status=r.status_code,content_type=r.headers.get('content-type'),content_length=r.headers.get('content-length'));log['attempts'].append(entry)
        r.raise_for_status();buf=bytearray()
        for b in r.iter_content(1<<20):
            buf.extend(b)
            if len(buf)>32<<20:raise ValueError('Supplement exceeds bounded32MiB paper-access scope')
        p=CACHE/'ynet_supplement.zip';p.write_bytes(buf)
        entry.update(bytes=len(buf),sha256=sha(p),real_zip=zipfile.is_zipfile(p))
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                entry['members']=[dict(name=x.filename,bytes=x.file_size) for x in z.infolist()]
                for x in z.infolist():
                    if x.filename.lower().endswith('.pdf') and x.file_size<16<<20:
                        q=CACHE/('supplement_'+Path(x.filename).name);q.write_bytes(z.read(x))
                        assert q.read_bytes().startswith(b'%PDF')
                        entry['extracted_PDF']=dict(path=str(q.relative_to(ROOT)),sha256=sha(q),bytes=q.stat().st_size)
    except Exception as e:
        log['error_type']=type(e).__name__;log['error']=str(e)
    paper=ROOT/'experiments/stage11a/cache/papers/ynet.txt'
    masks=json.loads((ROOT/'results/tables/stage11ap/semantic_map_audit.json').read_text())
    log.update(paper_SHA=sha(ROOT/'experiments/stage11a/cache/papers/ynet.pdf'),paper_read_scope='Sections4.1/Implementation details: five named stuff, inD same class set, dataset-finetuned pretrained segmentation; ID/ignore/train-label membership not specified in these sections',
        masks_checked=len(masks['maps']),IDs_present=sorted({v for x in masks['maps'] for v in x['unique_values']}),
        config_channels=6,SDD_specific_mapping='SEMANTIC_CLASS_MAPPING_UNVERIFIED',segmentation_pretraining='SEGMENTATION_EXPOSURE_UNVERIFIED',
        inD_map_not_assumed_SDD_authority=True,class0_not_named_background=True,tree_terrain_not_assumed_obstacles=True,
        scope='Train-only engineering with frozen author feature extractor allowed; no unseen-scene generalization claim')
    jnew(OUT/'semantic_audit.json',log)
    print('Semantic uncertainties retained; supplemental access logged')
if __name__=='__main__':main()
