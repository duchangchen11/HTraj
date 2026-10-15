"""Seal all prior research before independent public baseline inspection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import subprocess,csv,json
from src.data.preprocessing import sha256,write_json_new


def main():
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert base=='e6e9700eba015140d00e989a2aaaf7d7a88bd00d'
    paths=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))-{''}
    paths|={r['path'] for r in json.loads(Path('configs/stage4p2_final/protected_artifacts.json').read_text())['files']}
    paths|={r['path'] for r in csv.DictReader(Path('results/tables/stage4p2_final/checkpoint_inventory.csv').open())}
    write_json_new('configs/stage5/protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
    write_json_new('configs/stage5/scope.json',dict(base_commit=base,branch='feat/stage5-baseline-selection',
        static_candidates=['EqMotion','HighGraph','SGCN','SocialVAE'],max_functional_candidates=2,
        EqMotion_stop_rule='inspect all official history touchingETH/UCY network; bounded official-configuration grid, strict keys+shapes only; never synthesize missing layers or strictFalse',
        functional_policy='history-only deterministic subset selected before metrics, nonETH only; atmost2 candidates; no architecture changes or official training entrypoints with test selection',
        no_new_model=True,no_formal_training=True,no_ETH_metrics=True,no_push=True))
    print('Protected files:',len(paths))


if __name__=='__main__':main()
