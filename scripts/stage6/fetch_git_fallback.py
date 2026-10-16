"""Bounded fallback after GitHub REST rate limit; raw blobs pinned by Git trees."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,subprocess,concurrent.futures
from scripts.stage6.fetch_public import request
from src.data.preprocessing import sha256
from src.socialvae_baseline.artifacts import json_new

def main():
 root=Path('results/tables/stage6/public');cache=Path('experiments/stage6/cache/public')
 for name,repo,commit,local in [
  ('SGCN','shuaishiliu/SGCN','5153702d8ae32a2b0586b093d4e094aee08b0f9b',Path('experiments/stage5/repos/sgcn.git')),
  ('SingularTrajectory','InhwanBae/SingularTrajectory','f7855881976276787bc16d951a9259d1cd50c101',cache/'SingularTrajectory.git')]:
  remote=subprocess.run(['git','ls-remote',f'https://github.com/{repo}.git','HEAD'],capture_output=True,text=True,check=True).stdout
  clone_log=None
  if name=='SingularTrajectory':
   result=subprocess.run(['git','clone','--bare','--filter=blob:none','--depth=1',f'https://github.com/{repo}.git',str(local)],capture_output=True,text=True,check=True)
   clone_log=result.stdout+result.stderr
   assert subprocess.check_output(['git','-C',str(local),'rev-parse','HEAD'],text=True).strip()==commit
  tree=subprocess.check_output(['git','-C',str(local),'ls-tree','-r','--long',commit],text=True)
  with (root/f'{name}_git_tree.txt').open('x') as f:f.write(tree)
  folder=cache/name/commit;folder.mkdir(parents=True)
  entries=[]
  for line in tree.splitlines():
   meta,p=line.split('\t',1);parts=meta.split();low=p.lower()
   if parts[1]!='blob':continue
   if name=='SGCN':keep=p in ['model.py','utils.py','metrics.py','train.py','test.py','train.sh','README.md','LICENSE','checkpoints/hotel/val_best.pth','checkpoints/zara2/val_best.pth']
   else:keep=(Path(p).suffix in ['.py','.json','.yaml','.yml','.sh','.txt','.md'] or Path(p).name.startswith('LICENSE')) and not any(s in low for s in ['dataset/','datasets/','checkpoint/','checkpoint.', 'result/','results/','weight/','weights/']) and int(parts[3])<500000
   if keep:entries.append((p,parts[2]))
  def fetch(entry):
   p,blob=entry;data,log=request(f'https://raw.githubusercontent.com/{repo}/{commit}/{p}');log.update(path=p,git_blob=blob)
   if data is not None:
    import hashlib
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blob
    out=folder/p;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f:f.write(data)
    log.update(local_path=str(out),sha256=sha256(out),git_blob_verified=True)
   return log
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:files=list(pool.map(fetch,entries))
  json_new(root/f'{name}_git_fallback.json',dict(repository=repo,commit=commit,remote_HEAD=remote.strip(),clone_log=clone_log,source_directory=str(folder),files=files,
   success=all('error' not in f for f in files),fallback_reason='GitHub REST HTTP403 rate limit; Git metadata plus raw fixed-commit blobs',ETH_scores_read=False))
  print(name,commit,len(files),'files',sum('error' in r for r in files),'failed',flush=True)

if __name__=='__main__':main()
