"""Pin two author repositories and retrieve source blobs only, no datasets/scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,urllib.request,hashlib,subprocess,time,concurrent.futures
from src.socialvae_baseline.artifacts import json_new

ROOT=Path('results/tables/stage7/sources');CACHE=Path('experiments/stage7/cache');ROOT.mkdir(parents=True,exist_ok=True)


def request(url):
 start=time.perf_counter()
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'HTraj-stage7-source-audit'}),timeout=35) as r:data=r.read();status=r.status
  return data,dict(url=url,status=status,bytes=len(data),seconds=time.perf_counter()-start)
 except Exception as e:return None,dict(url=url,error=repr(e),seconds=time.perf_counter()-start)


def main():
 for name,repo in [('AgentFormer','Khrylx/AgentFormer'),('PECNet','HarshayuGirase/PECNet')]:
  remote=subprocess.check_output(['git','ls-remote',f'https://github.com/{repo}.git','HEAD'],text=True).strip();commit=remote.split()[0]
  bare=CACHE/(name+'.git')
  if not bare.exists():
   clone=subprocess.run(['git','clone','--bare','--filter=blob:none','--depth=1',f'https://github.com/{repo}.git',str(bare)],capture_output=True,text=True,check=True)
   clone_text=clone.stdout+clone.stderr
  else:clone_text='Reuse current Stage7 partial clone; no previous-stage source modified'
  assert subprocess.check_output(['git','-C',str(bare),'rev-parse','HEAD'],text=True).strip()==commit
  tree=subprocess.check_output(['git','-C',str(bare),'ls-tree','-r',commit],text=True)
  with (ROOT/f'{name}_tree.txt').open('x') as f:f.write(tree)
  entries=[]
  for line in tree.splitlines():
   meta,p=line.split('\t',1);parts=meta.split();low=p.lower()
   if parts[1]!='blob':continue
   keep=(Path(p).suffix in ['.py','.yml','.yaml','.json','.sh'] or Path(p).name in ['README.md','requirements.txt','LICENSE'])
   keep=keep and not any(s in low for s in ['saved_models/','results/','social_pool_data/','datasets/'])
   if name=='PECNet':keep=keep and (p.startswith('PECNet/') or p in ['README.md','LICENSE'])
   if keep:entries.append((p,parts[2]))
  folder=CACHE/name/commit;folder.mkdir(parents=True)
  def fetch(entry):
   path,blob=entry;data,log=request(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}')
   log.update(path=path,git_blob=blob)
   if data is not None:
    assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==blob
    out=folder/path;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f:f.write(data)
    log.update(local_path=str(out),sha256=hashlib.sha256(data).hexdigest(),git_blob_verified=True)
   return log
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:files=list(pool.map(fetch,entries))
  json_new(ROOT/f'{name}.json',dict(repository=repo,commit=commit,remote_HEAD=remote,clone_log=clone_text,
   files=files,source_directory=str(folder),success=all('error' not in f for f in files),ETH_evaluated=False,
   pretrained_weights_downloaded=False,datasets_downloaded=False))
  print(name,commit,len(files),'files',sum('error' in f for f in files),'failed',flush=True)


if __name__=='__main__':main()
