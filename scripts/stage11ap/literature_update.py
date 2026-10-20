"""Bounded T1 metadata, lawful OA and small author-code audit; no paper scraping loops."""
from common import *
import requests

PAPERS=[('CINet2025','10.1109/TMM.2025.3542967'),('McTrans2026','10.1016/j.jvcir.2026.104723'),('GDS_TGTF2026','10.1016/j.eswa.2026.134731')]
def main():
    log=[]; folder=CACHE/'literature';folder.mkdir(exist_ok=True)
    for name,doi in PAPERS:
        for service,url in [('crossref','https://api.crossref.org/works/'+doi),
                            ('openalex','https://api.openalex.org/works/https://doi.org/'+doi),
                            ('semantic_scholar','https://api.semanticscholar.org/graph/v1/paper/DOI:'+doi+'?fields=title,year,openAccessPdf,url,externalIds')]:
            row=dict(paper=name,doi=doi,service=service,url=url)
            try:
                p=folder/f'{name}_{service}.json'
                if '--cached' in sys.argv and p.exists():
                    class Cached:
                        content=p.read_bytes(); status_code=None
                        def json(self): return json.loads(self.content)
                    r=Cached()
                else:
                    r=requests.get(url,timeout=20);p.write_bytes(r.content)
                row.update(status=r.status_code,bytes=len(r.content),response_sha256=sha(p))
                if r.status_code==200 or r.status_code is None:
                    d=r.json()
                    if service=='crossref':row['full_text_links']=d.get('message',{}).get('link',[])
                    if service=='openalex':row['locations']=d.get('locations',[])
                    if service=='semantic_scholar':row['openAccessPdf']=d.get('openAccessPdf')
            except Exception as e:row.update(error_type=type(e).__name__,error=str(e))
            log.append(row)
    repo='Jayzhou3219/CINet-main'
    head=subprocess.check_output(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],text=True).split()[0]
    clone=folder/'CINet_git'
    if not clone.exists():
        cp=subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/'+repo+'.git',str(clone)],capture_output=True,text=True)
        (folder/'CINet_clone.txt').write_text(cp.stdout+cp.stderr);assert cp.returncode==0
    tree=[]
    for line in subprocess.check_output(['git','ls-tree','-r',head],cwd=clone,text=True).splitlines():
        meta,path=line.split('\t');mode,kind,blob=meta.split()
        tree.append(dict(path=path,type=kind,sha=blob))
    if not (OUT/'CINet_source_tree.json').exists():
        jnew(OUT/'CINet_source_tree.json',dict(repo=repo,commit=head,tree=tree,api_failure='HTTP403 GitHub rate limit; git tree used'))
    records=[]
    for e in tree:
        if e['type']!='blob' or e.get('size',0)>262144: continue
        if e['path'].split('/')[0] in {'data','output','weights','pretrained_models'}:continue
        if not e['path'].endswith(('.py','.md','.txt','.yaml','.yml')):continue
        url=f'https://raw.githubusercontent.com/{repo}/{head}/'+e['path']
        s=requests.get(url,timeout=20);s.raise_for_status();assert len(s.content)<262144
        p=folder/'CINet'/e['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(s.content)
        blob=hashlib.sha1(b'blob '+str(len(s.content)).encode()+b'\0'+s.content).hexdigest()
        assert blob==e['sha']
        records.append(dict(path=e['path'],source_url=url,bytes=len(s.content),sha256=sha(p),git_blob_sha=blob))
    jnew(OUT/'CINet_source_manifest.json',dict(repo=repo,commit=head,executed=False,license='No root license found; audit only, source not redistributed',files=records))
    jnew(OUT/'literature_access_update.json',dict(workflow='multi-source-search',academic_MCP_available=False,primary_metadata_then_OA=True,attempts=log))
    print('Metadata requests:',len(log),'CINet sources read only:',len(records))
if __name__=='__main__':main()
