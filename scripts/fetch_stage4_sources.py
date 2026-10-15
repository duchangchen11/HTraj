"""Freeze official code; retrieve only explicitly allowed non-ETH assets.

Git tree names are metadata. Do not use ls-tree --long on partial clones:
blob size inspection can trigger unnecessary lazy downloads. No ETH files or
historical official metrics.pkl/constant_metrics.pkl are checked out/read.
"""
import hashlib,json,subprocess,urllib.request
from pathlib import Path

ROOT=Path('data/external/stage4')
TABLE=Path('results/tables/stage4')
CONF=Path('configs/stage4')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args])


def extract(repo,commit,file,out):
    if out.exists():
        expected=git(repo,'rev-parse',commit+':'+file).decode().strip()
        actual=subprocess.check_output(['git','hash-object',str(out)]).decode().strip()
        if actual!=expected:raise RuntimeError('Existing official snapshot differs: '+str(out))
        return dict(official_path=file,local=str(out),sha256=sha(out),bytes=out.stat().st_size)
    out.parent.mkdir(parents=True,exist_ok=True)
    expected=git(repo,'rev-parse',commit+':'+file).decode().strip()
    try:payload=git(repo,'show',commit+':'+file)
    except subprocess.CalledProcessError:
        remote=git(repo,'remote','get-url','origin').decode().strip().removesuffix('.git')
        owner_repo=remote.split('github.com/')[-1]
        url=f'https://raw.githubusercontent.com/{owner_repo}/{commit}/{file}'
        request=urllib.request.Request(url,headers={'User-Agent':'HTraj-public-baseline-audit'})
        with urllib.request.urlopen(request,timeout=40) as response:payload=response.read()
    actual=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    if actual!=expected:raise RuntimeError('Official Git blob mismatch '+file)
    out.write_bytes(payload)
    return dict(official_path=file,local=str(out),sha256=sha(out),bytes=out.stat().st_size)


def main():
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert base=='f12146143514d2b6eb6cf14877cb5b4cae454543'
    old=[p for p in subprocess.check_output(['git','ls-files','-z']).decode().split('\0') if p]
    weights=[str(p) for p in Path('experiments').rglob('*.pt') if not str(p).startswith('experiments/stage4/')]
    protected=sorted(set(old+weights))
    protection=dict(base_commit=base,tracked_files=len(old),old_checkpoints=len(weights),files=[dict(path=p,sha256=sha(p)) for p in protected])
    protection_path=CONF/'protected_artifacts.json'
    if protection_path.exists():assert json.loads(protection_path.read_text())==protection
    else:protection_path.write_text(json.dumps(protection,sort_keys=True,indent=2)+'\n')
    sources=[]
    for name,url in [('social_stgcnn','https://github.com/abduallahmohamed/Social-STGCNN'),('eqmotion','https://github.com/MediaBrain-SJTU/EqMotion')]:
        repo=ROOT/'repos'/name;commit=git(repo,'rev-parse','HEAD').decode().strip()
        paths=git(repo,'ls-tree','-r','--name-only',commit).decode().splitlines()
        # Metadata only inventory explicitly marks sealed paths.
        (TABLE/(name+'_tree_inventory.json')).write_text(json.dumps(dict(commit=commit,paths=paths,blob_contents_read_for_inventory=False),indent=2)+'\n')
        files=['LICENSE','requirements.txt','model.py','utils.py','metrics.py','train.py','test.py'] if name=='social_stgcnn' else ['LICENSE']+[p for p in paths if p.endswith('.py') and ('/data/' not in p) and ('__pycache__' not in p)]
        # EqMotion's other domain source is metadata/audit only; avoid
        # unrelated numeric datasets, checkpoints, caches and result logs.
        if name=='eqmotion':files=['LICENSE']+[p for p in files if p.startswith('eth_ucy/') or ('/' not in p and p.endswith('.py'))]
        records=[]
        for file in files:
            if file in paths:records.append(extract(repo,commit,file,Path('third_party/stage4')/name/file))
        readme='README.MD' if name=='social_stgcnn' else 'README.md'
        # Keep upstream readme local. Only metadata/setup lines are examined;
        # no official ETH metric table is parsed into research reports.
        records.append(extract(repo,commit,readme,ROOT/'readmes'/name/readme))
        assets=[]
        if name=='social_stgcnn':
            allow=[p for p in paths if any(p.startswith('datasets/'+fold+'/test/') for fold in ('hotel','univ','zara1','zara2'))]
            allow += [f'checkpoint/social-stgcnn-{fold}/{file}' for fold in ('hotel','univ','zara1','zara2') for file in ('args.pkl','val_best.pth')]
            for file in allow:
                assert 'biwi_eth' not in file and '/social-stgcnn-eth/' not in file and not file.endswith(('metrics.pkl','constant_metrics.pkl'))
                assets.append(extract(repo,commit,file,ROOT/'assets'/name/file))
        sources.append(dict(name=name,url=url,commit=commit,files=records,assets=assets,ETH_trajectories_checked_out=False,ETH_metrics_read=False))
    (CONF/'official_sources.json').write_text(json.dumps(dict(base_commit=base,sources=sources,ETH_evaluated=False),sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(protected_artifacts=len(protected),old_checkpoints=len(weights),repositories=[dict(name=s['name'],commit=s['commit'],code_files=len(s['files']),nonETH_assets=len(s['assets'])) for s in sources])))


if __name__=='__main__':main()
