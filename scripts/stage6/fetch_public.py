"""Freeze public source trees; fetch only two nonETH SGCN state dictionaries."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,time,urllib.request,concurrent.futures
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

ROOT=Path('results/tables/stage6/public');ROOT.mkdir(parents=True,exist_ok=True)
CACHE=Path('experiments/stage6/cache/public');CACHE.mkdir(parents=True,exist_ok=True)

def request(url):
 start=time.perf_counter()
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'HTraj-stage6-research-audit'})
  with urllib.request.urlopen(req,timeout=30) as response: data=response.read();status=response.status
  return data,dict(url=url,status=status,bytes=len(data),seconds=time.perf_counter()-start)
 except Exception as e:return None,dict(url=url,error=str(e),seconds=time.perf_counter()-start)

def main():
 for name,repo,revision in [('SGCN','shuaishiliu/SGCN','5153702d8ae32a2b0586b093d4e094aee08b0f9b'),('SingularTrajectory','InhwanBae/SingularTrajectory','main')]:
  logs=[];metadata={}
  for kind,url in [('commit',f'https://api.github.com/repos/{repo}/commits/{revision}'),('repository',f'https://api.github.com/repos/{repo}')]:
   data,log=request(url);logs.append(log)
   if data is not None:metadata[kind]=json.loads(data)
  if 'commit' not in metadata:
   json_new(ROOT/f'{name}_fetch.json',dict(repository=repo,logs=logs,success=False));continue
  commit=metadata['commit']['sha'];data,log=request(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1');logs.append(log)
  if data is None:
   json_new(ROOT/f'{name}_fetch.json',dict(repository=repo,commit=commit,logs=logs,success=False));continue
  tree=json.loads(data);json_new(ROOT/f'{name}_tree.json',tree)
  folder=CACHE/name/commit;folder.mkdir(parents=True)
  selected=[]
  for r in tree['tree']:
   p=r['path'];low=p.lower()
   if r['type']!='blob':continue
   if name=='SGCN':
    keep=p in ['model.py','utils.py','metrics.py','train.py','test.py','train.sh','README.md','LICENSE','requirements.txt','checkpoints/hotel/val_best.pth','checkpoints/zara2/val_best.pth']
   else:
    keep=(Path(p).suffix in ['.py','.json','.yaml','.yml','.sh','.txt','.md'] or Path(p).name.startswith('LICENSE')) and not any(s in low for s in ['dataset/','datasets/','checkpoint/','checkpoint.', 'result/','results/','weight/','weights/']) and r.get('size',0)<500000
   if keep:selected.append(r)
  def get_file(r):
   p=r['path'];data,log=request(f'https://raw.githubusercontent.com/{repo}/{commit}/{p}');log.update(path=p,git_blob=r['sha'])
   if data is not None:
    out=folder/p;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f:f.write(data)
    log.update(local_path=str(out),sha256=sha256(out))
   return log
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:files=list(pool.map(get_file,selected))
  json_new(ROOT/f'{name}_fetch.json',dict(repository=repo,commit=commit,requested_revision=revision,repository_default_branch=metadata.get('repository',{}).get('default_branch'),
   commit_date=metadata['commit']['commit']['committer']['date'],commit_message=metadata['commit']['commit']['message'],logs=logs,files=files,success=all('error' not in r for r in files),
   ETH_scores_fetched=False,weights='only SGCN HOTEL/ZARA2 val_best state dictionaries; no score/args pickle',source_directory=str(folder)))
  print(name,commit,len(files),'files; failed',sum('error' in f for f in files),flush=True)

if __name__=='__main__':main()
