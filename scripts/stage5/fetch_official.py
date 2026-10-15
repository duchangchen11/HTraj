"""Fetch only pinned code and an explicit non-ETH functional asset allowlist.

No official train/test entrypoint is invoked. Re-running verifies cached bytes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
from scripts.stage5.common import ROOT, CACHE, OUT, COMMITS, export_blob, check_non_eth_asset

REPOS = {'eqmotion':'https://github.com/MediaBrain-SJTU/EqMotion.git',
         'socialvae':'https://github.com/xupei0610/SocialVAE.git',
         'sgcn':'https://github.com/shuaishiliu/SGCN.git'}
FILES = {
    'eqmotion':['eth_ucy/model_t.py','eth_ucy/gcl_t.py','eth_ucy/preprocessor.py',
                'eth_ucy/dataloader_diverse.py','eth_ucy/process_eth_data_diverse.py','main_eth_diverse.py','LICENSE'],
    'socialvae':['social_vae.py','data.py','utils.py','main.py','requirements.txt','LICENSE',
                 'config/hotel.py','config/zara01.py','config/zara02.py'],
    'sgcn':['model.py','utils.py','metrics.py','train.py','test.py','train.sh','LICENSE'],
}


def main():
    records = []
    for name, url in REPOS.items():
        repo = ROOT/'experiments/stage5/repos'/(name+'.git')
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git','clone','--bare','--filter=blob:none',url,str(repo)],check=True)
        for file in FILES[name]:
            records.append(export_blob(name, COMMITS[name], file, CACHE/name/COMMITS[name]/file))
    for file in ['models/hotel/ckpt-best','models/zara02/ckpt-best','data/hotel/test/biwi_hotel.txt',
                 'data/zara02/test/crowds_zara02.txt','data/zara02/train/crowds_zara01.txt']:
        check_non_eth_asset(file)
        records.append(export_blob('socialvae',COMMITS['socialvae'],file,ROOT/'experiments/stage5/assets/socialvae'/file))
    print(json.dumps(dict(fetched_or_verified=records, no_ETH_assets=True),indent=2))


if __name__ == '__main__':
    main()
