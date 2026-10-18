"""Bounded audit samples, not a dataset download or a train/test split.

Third-party SDD mirror provenance remains UNVERIFIED against Stanford archive.
At most 4MiB annotation prefix + 3MiB reference per video, six videos only.
"""
from common import *
import concurrent.futures
import requests

SCENES=['bookstore','coupa','deathCircle','gates','hyang','little']

def main():
    repo=CACHE/'SDD_annotation_mirror'
    version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tree=subprocess.check_output(['git','ls-tree','-r','HEAD'],cwd=repo,text=True)
    (OUT/'sdd_mirror_tree.txt').write_text(tree)
    expected={line.split('\t')[1]:line.split()[2] for line in tree.splitlines()}
    tasks=[(scene,kind) for scene in SCENES for kind in ('annotations.txt','reference.jpg')]
    def fetch(task):
        scene,kind=task
        name=scene+'/video0/'+kind
        url='https://raw.githubusercontent.com/flclain/StanfordDroneDataset/'+version+'/'+name
        cap=(4 if kind=='annotations.txt' else 3)*1024*1024
        try:
            r=requests.get(url,stream=True,timeout=25);r.raise_for_status()
            chunks=[];size=0;complete=True
            for chunk in r.iter_content(65536):
                if size+len(chunk)>cap:
                    complete=False;break
                chunks.append(chunk);size+=len(chunk)
            r.close();raw=b''.join(chunks)
            if not complete and kind=='reference.jpg':raise ValueError('Image exceeds small-file budget')
            if not complete:raw=raw[:raw.rfind(b'\n')+1]
            p=CACHE/'sdd_samples'/scene/'video0'/(kind if complete else 'annotations_prefix.txt')
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(raw)
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() if complete else None
            if complete:assert blob==expected[name]
            return dict(scene=scene,video='video0',kind=kind,path=str(p.relative_to(ROOT)),url=url,
                bytes=len(raw),sha256=sha(p),complete_file=complete,expected_mirror_blob=expected[name],
                mirror_blob_verified=bool(complete and blob==expected[name]),
                acquisition_status='PARTIAL',provenance_status='UNVERIFIED',
                provenance_limit='Third-party mirror fixed Git blob; Stanford archive byte identity not verified',
                data_license='Stanford original CC-BY-NC-SA-3.0; do not replace by mirror software license')
        except Exception as e:return dict(scene=scene,video='video0',kind=kind,url=url,error=str(e),acquisition_status='UNAVAILABLE',provenance_status='UNVERIFIED')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,tasks))
    jnew(OUT/'sdd_sample_download.json',dict(mirror_commit=version,max_download_budget_bytes=42*1024*1024,
        video_downloaded=False,rows=rows,total_saved_bytes=sum(r.get('bytes',0) for r in rows),
        split='audit-only; YNet manifest not available, no training/evaluation use',official_equivalence_verified=False))
    print('Audit small sample files',sum('path' in r for r in rows),'of',len(rows),flush=True)

if __name__=='__main__':main()
