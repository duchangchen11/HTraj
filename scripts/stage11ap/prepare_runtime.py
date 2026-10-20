from common import *
import shutil, importlib, platform
def main():
    old=ROOT/'experiments/stage11a/cache/official/ynet';new=CACHE/'official/ynet'
    new.parent.mkdir(exist_ok=True)
    if not new.exists():shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__'))
    records=[]
    old_manifest=json.loads((ROOT/'results/tables/stage11a/source_download.json').read_text())
    for r in old_manifest['files']:
        if not r['path'].startswith('ynet/'):continue
        p=CACHE/'official'/r['path'];assert sha(p)==r['sha256']
        records.append(dict(path=r['path'],sha256=sha(p),bytes=p.stat().st_size))
    if not (OUT/'frozen_source_manifest.json').exists():
        jnew(OUT/'frozen_source_manifest.json',dict(commit=MODEL_COMMIT,repository='https://github.com/HarshayuGirase/Human-Path-Prediction',files=records,source_modified=False,license='MIT'))
    versions={}
    for name in ['torch','torchvision','numpy','pandas','yaml','cv2','scipy','segmentation_models_pytorch','timm','pretrainedmodels','efficientnet_pytorch']:
        try:
            m=importlib.import_module(name)
            versions[name]=dict(version=getattr(m,'__version__','not declared'),file=m.__file__,import_passed=True)
        except ImportError as e:
            versions[name]=dict(import_passed=False,error=str(e),chosen_ResNet_branch_uses_this_package=False)
    jnew(OUT/'runtime_environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,
         isolation='New venv packages; read-only .pth for existing Torch/numerics in stage4; no prior packages installed/modified',
         model_dependencies_install_log='experiments/stage11ap/cache/model_dependencies_install.txt',
         dependency_compatibility='SMP0.1.0 matches pinned official requirement; Torch remains2.5.1. Historical cross-version numerical equivalence not proven'))
    print('Official files copied and verified:',len(records))
if __name__=='__main__':main()
