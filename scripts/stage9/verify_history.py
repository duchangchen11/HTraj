"""Opaque hash validation only; old test errors/predictions are never parsed."""
from common import *

def run():
    failures=[];count=0
    def check(path,expected):
        nonlocal count
        count+=1
        p=ROOT/path
        if not p.is_file() or sha(p)!=expected:failures.append(path)
    with gzip.open(ROOT/'configs/stage9/protected_files.csv.gz','rt') as f:
        for r in csv.DictReader(f):check(r['path'],r['sha256'])
    probe=json.loads((ROOT/'results/tables/stage8/resonance_probe.json').read_text())
    check(probe['checkpoint_restore']['path'],probe['checkpoint_restore']['sha256'])
    old=json.loads((ROOT/'configs/stage8/protocol.json').read_text())
    for index in old['prediction_SHA_indices']:
        check(index['path'],index['sha256'])
        with gzip.open(ROOT/index['path'],'rt') as f:
            for r in csv.DictReader(f):check(r['path'],r.get('sha256',r.get('SHA256')))
    json_new(OUT/('historical_preservation_final.json' if '--final' in sys.argv else 'historical_preservation.json'),dict(passed=not failures,checked_files=count,failures=failures,test_prediction_bytes_hashed_only=True,test_errors_parsed=False))
    assert not failures,failures[:10]
    print('Historical byte SHA verified',count)

if __name__=='__main__':run()
