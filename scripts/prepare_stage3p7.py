"""Protect the full archive and preregister the finite coordinate experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import subprocess
from src.data.preprocessing import sha256, write_json_new


def main():
    conf = Path('configs/stage3p7'); conf.mkdir(exist_ok=False)
    root = Path('results/tables/stage3p7'); root.mkdir(exist_ok=False)
    tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    checkpoints = sorted(p for p in Path('experiments').rglob('*.pt'))
    paths = sorted(set(p for p in tracked if p) | {str(p) for p in checkpoints})
    write_json_new(conf/'protected_artifacts.json', dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        tracked_files=sum(bool(x) for x in tracked), old_checkpoints=len(checkpoints), files=[dict(path=p, sha256=sha256(p)) for p in paths]))
    old = json.loads(Path('configs/stage3p6/protocol.json').read_text())
    protocol = dict(stage='3.7', protocol_name=old['protocol_name'], raw_root=old['raw_root'], source_files=old['source_files'], folds=old['folds'],
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), seed=17, epochs=30, batch_size=8, hidden_dim=64, prediction_steps=12,
        sampling_seconds=.4, learning_rate=.001, optimizer='Adam default betas/eps, weight_decay=0', gradient_clip=1., cpu_threads=1,
        schedule='constant; no LR factor tested in this stage', loss='unchanged masked per-target mean Euclidean distance',
        neural_models=['M0', 'M2', 'M3'], M1='exact alias of M0; translation centering cannot change derivative-only inputs mathematically; no duplicate training',
        models=dict(M0='original PedBase, newly trained seed17', M2='same GRU/head in per-target last-origin history-heading frame',
                    M3='M2 with historical last-velocity CV base; correction head unchanged'),
        heading_threshold_mps=.2,
        heading_rule='time-weighted mean velocity in current contiguous suffix; speed>=.2m/s is reliable; else identity rotation, explicitly not globally rotation-equivariant in fallback group',
        fallback_rule='unknown/low speed: unit=(1,0), retain existing velocity/acceleration missing flags; never invent observed zero velocity',
        CV='last consecutive observed velocity, fixed .4s future steps; stationary fallback on missing velocity',
        normalization='translation and SO(2) rotation only, no scaling, reflect/augmentation/learned parameters',
        best_rule='minimum own-fold validation ADE; earliest exact tie; paired FDE; retain last30; no early stopping/extension',
        endpoints=['best', 'last'],
        supplemental=dict(uses_no_new_training=True, used_for_model_selection=False, threshold_sensitivity_mps=[.1, .4],
            rotations_degrees=[90, 180, -90],
            purpose='frozen checkpoint geometry/threshold sensitivity, not independent retrained performance proof; endpoint train/val errors distinguish fit and generalization but cannot uniquely rule out optimization effects'),
        compatible_reference='Stage3.6 CV-Residual constant seed17, identical original source/data/evaluator/loss/init/optimizer/batch/budget; frozen reference only; SHA checked; M3 minus this isolates frame computation in training+inference, not architecture capacity',
        speed_groups='history contiguous-suffix mean speed: low<.2, medium .2<=v<1, high>=1 m/s; unknown separate; plus all Stage3.6 history groups',
        candidate_rule='recommend (never auto-run) seed42/2026 only if M2 or M3 reduces M0 ADE by max(.005m,.5% M0 ADE) in BOTH recordings with paired FDE no worse; CV dominance and low/high speed harm always reported',
        limitations=['recordings chosen using historical results; not independent confirmation', 'one seed only, overlapping windows correlated',
            'M2/M3 use whole observed history to define frame; hidden states are not prefix-causal social payloads without prefix recomputation',
            'identity heading fallback breaks unconditional rotation equivariance; motion threshold sensitivity uses frozen weights only'],
        ETH_evaluated=False, B_run=False, extra_seeds_automatically_run=False, auto_push=False,
        endpoint='two fixed recordings; analytical CV; six fresh 30-epoch neural runs M0/M2/M3 seed17; M1 alias; frozen old unnormalized CV-residual reference; no additional training')
    write_json_new(conf/'protocol.json', protocol)
    print(json.dumps(dict(protected_files=len(paths), old_checkpoints=len(checkpoints), new_runs=6, epochs=180)))


if __name__ == '__main__': main()
