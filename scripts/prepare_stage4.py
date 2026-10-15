"""Fixed non-ETH public-baseline scope, no architecture/metric searching."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,platform,subprocess
import torch,numpy,scipy,networkx
from src.data.preprocessing import sha256,write_json_new


def main():
    source=json.loads(Path('configs/stage4/official_sources.json').read_text())
    protocol=dict(stage='4',base_commit=source['base_commit'],social_folds=['hotel','univ','zara1','zara2'],monte_carlo_seeds=[17,42,2026],K=20,
        social_scope='complete official test partitions for four nonETH benchmark scenes, five recording files; official supplied val_best per fold; no HTraj data/weights substituted',
        social_metrics=['K1_first_sample from the same20 bank','K1_distribution_mean deterministic Gaussian mean','K20_independent_min per-person separate ADE/FDE minima'],
        social_execution='AST extract and execute upstream test() unchanged; avoid upstream global driver reading saved metrics/ETH, aggregate4 not5; numerical equality check against independent error calculation',
        social_compatibility='nx.from_numpy_matrix alias to nx.from_numpy_array, no_grad, num_workers0; upstream source byte-identical, no model/loss change',
        eqmotion_folds=['hotel','zara1'],eqmotion_max_target_samples_per_fold=128,eqmotion_batch_size=16,
        eqmotion_neighbor_budget=dict(hotel=1,zara1=3),eqmotion_channels=dict(hotel=64,zara1=128),eqmotion_neighbor_radius_m=5.,
        eqmotion_scope='bounded official pretrained functional validation; chronological first128 target-centered examples per fold; original preprocessor complete8/12; metadata and strict key checks; not full published performance reproduction',
        eqmotion_metrics=['K1_head0 predeclared deterministic head','K20_independent_min over20 learned output heads, target0 only as official evaluator'],
        eqmotion_execution='do not run main() (--test falls through into training); isolated test() only; preprocess selected test files, no official train pool or ETH files loaded; absolute path redirected by wrapper',
        eqmotion_compatibility='np.int=int API alias in wrapper; preprocessor path as explicit argument; checkpoints strict=True rather than silently ignoring missing keys',
        eqmotion_geometry_checks='first batch, fixed .731rad rotation, translation[3,-2], reflection, agent permutation; output equivariance and category invariance; not new model/selection',
        obs_len=8,pred_len=12,nominal_dt=.4,cpu_threads=1,device='cuda',training=False,ETH_evaluated=False,ETH_results_read=False,
        independent_confirmation=False,protocol_comparison='official full-recording/complete20 context vs HTraj originalval/history-complete context; official pools include extra recordings and ETH-trained pretrained models; not fair K1-vsK20 or same-data model ranking',
        pretrained_ETH_training_exposure='nonETH official pretrained checkpoints may have seen ETH training splits in their upstream training; unknown upstream selection history; this run never loads ETH trajectory/evaluation metrics; not ETH-free training or future external test independence claim',
        forbidden=['new training','ETH data/evaluation scores','loss/architecture change','sample-count/head/seed/data choices based on resulting metrics','read stored official metrics.pkl'],
        endpoint='full four-scene Social checkpoint evaluation with3 fixed MonteCarlo seeds; at most128 target cases each two EqMotion folds; source/protocol/novelty audit; local commit only',auto_push=False)
    write_json_new('configs/stage4/protocol.json',protocol)
    environment=dict(python=sys.version,prefix=sys.prefix,torch=str(torch.__version__),numpy=numpy.__version__,scipy=scipy.__version__,networkx=networkx.__version__,
        cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),platform=platform.platform(),
        isolated_environment=True,environment_method='conda offline clone of installed dependency environment into independent experiments/stage4/venv; no old-project model/results imported')
    write_json_new('results/tables/stage4/environment.json',environment)
    versions=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True)
    Path('results/tables/stage4/package_versions.json').write_text(versions)
    print(json.dumps(dict(environment=environment['prefix'],official_sources_sha256=sha256('configs/stage4/official_sources.json'),ETH_evaluated=False)))


if __name__=='__main__':main()
