"""Whitelist two non-ETH EqMotion assets for bounded functional tests."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fetch_stage4_sources import extract,git,ROOT


def main():
    path=Path('configs/stage4/official_sources.json');p=json.loads(path.read_text());eq=next(s for s in p['sources'] if s['name']=='eqmotion')
    repo=ROOT/'repos'/'eqmotion';commit=eq['commit'];assert git(repo,'rev-parse','HEAD').decode().strip()==commit
    allow=['eth_ucy/saved_models/hotel_ckpt.pth.tar','eth_ucy/saved_models/zara1_ckpt.pth.tar',
           'eth_ucy/data/hotel/biwi_hotel.txt','eth_ucy/data/zara1/crowds_zara01.txt']
    records=[]
    for file in allow:
        assert 'eth_ckpt' not in file and 'biwi_eth' not in file
        records.append(extract(repo,commit,file,ROOT/'assets'/'eqmotion'/file))
    eq['assets']=records
    path.write_text(json.dumps(p,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(model='EqMotion',assets=[dict(path=r['official_path'],bytes=r['bytes']) for r in records],ETH_read=False)))


if __name__=='__main__':main()
