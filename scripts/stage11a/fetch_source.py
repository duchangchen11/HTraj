"""Official small code blobs only, verified against frozen Git blob IDs."""
from common import *
import concurrent.futures
import requests

def main():
    repo=CACHE/'Human-Path-Prediction'
    version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    lines=(OUT/'source_tree_no_sizes.txt').read_text().splitlines()
    entries=[]
    for line in lines:
        info,name=line.split('\t')
        if (name.startswith('ynet/') or name.startswith('PECNet/') or name=='LICENSE') and name.endswith(('.py','.txt','.md','.yaml','.ipynb')):
            entries.append((name,info.split()[-1]))
    def fetch(item):
        name,blob=item
        url='https://raw.githubusercontent.com/HarshayuGirase/Human-Path-Prediction/'+version+'/'+name
        try:
            r=requests.get(url,timeout=30);r.raise_for_status()
            assert len(r.content)<3*1024*1024,'Not a small code blob'
            check=hashlib.sha1(b'blob '+str(len(r.content)).encode()+b'\0'+r.content).hexdigest()
            assert check==blob,'Git blob SHA mismatch'
            p=CACHE/'official'/name;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(r.content)
            return dict(path=name,url=url,git_blob_SHA1=blob,sha256=sha(p),bytes=len(r.content),verified=True)
        except Exception as exc:return dict(path=name,url=url,error=str(exc),verified=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,entries))
    jnew(OUT/'source_download.json',dict(repository='https://github.com/HarshayuGirase/Human-Path-Prediction',commit=version,files=rows))
    print('Small official code blobs',len(rows),'verified',sum(r['verified'] for r in rows),flush=True)

if __name__=='__main__':main()
