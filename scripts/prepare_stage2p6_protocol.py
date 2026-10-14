"""Freeze a new experiment contract and protect all old code/experiment bytes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import write_json_new


def main():
    root=Path('configs/stage2p6'); root.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('configs/stage2p5/protocol.json').read_text())
    keys=['held_out_scene','test_evaluation','data_manifest','raw_root','epochs','batch_size','learning_rate',
          'gradient_clip','cpu_threads','hidden_dim','prediction_steps','max_order','max_neighbors','radius',
          'initial_residual_gate','loss','metrics','best_epoch_rule','convergence_rule']
    protocol={k:old[k] for k in keys}
    protocol.update(schema_version=1,stage='2.6',base_commit='fb8373997134c476f98a1912cf82fbceb536ac39',
        subset_manifest=old['large_subset'],seeds=[17,42,2026],frozen_models=['PedBase','Dynamic-GNN-RN'],
        new_models=['TOHI-H1','TOHI-H2','TOHI-H3','TOHI-NoOrder','TOHI-H1-Cap','TOHI-H2-Cap'],
        time_control='strict old lower state vs old lower state + current received lower message; q1 and carry unchanged',
        capacity_rule='same d64; independently widen only residual delta intermediate to395/230, near H3 params',
        initialization='full H3 A-R template, truncate modules without RNG draws; common parameters identical per seed',
        groups='original Stage2.5 groups plus full8/partial intersections; no future-dependent grouping',
        path_statistics='history-only timed walks at cutoff; H2 3-distinct nodes and unique intermediaries per endpoint pair; H3 backtracks/end returns',
        negative_controls='fixed-checkpoint inference only; local max-change, scene random/max-change, in/out degree swaps, indirect cut and matched random deletion',
        negative_seed=1726,scale_control_trained=False,additional_loss=False,augmentation=False,decoder_changes=False,
        relation_intervention_legality='all: continuous same-scene no-self no-duplicate history endpoints; nonlocal controls explicitly relax4m radius',
        interpretation='paired three seed development differences, no significance/generalization claim; independent training vs inference separated')
    write_json_new(root/'protocol.json',protocol)
    protected={}
    paths=subprocess.check_output(['git','ls-files','src','configs/stage1_design.yaml','configs/data_manifests',
        'configs/stage2_smoke.json','configs/smoke_subsets','configs/stage2p5','reports/stage1','reports/stage2',
        'reports/stage2p5','results/tables'],text=True).splitlines()
    paths += [str(p) for p in Path('experiments').rglob('*.pt')]
    # New untracked code is absent from git ls-files at this pre-freeze point.
    for path in sorted(set(paths)):
        protected[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    write_json_new(root/'protected_artifacts.json',protected)
    print('Fixed Stage2.6 protocol; protected previous artifacts:',len(protected))


if __name__=='__main__':
    main()
