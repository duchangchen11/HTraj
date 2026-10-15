"""Freeze exactly two independently initialized 100-full-epoch runs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import subprocess
from src.data.preprocessing import sha256,write_json_new
from src.socialvae_causal.official import SOURCE,EXPECTED


def main():
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert base=='b7bcd4a5bfb47cc5c5d4f19d3d3f05dafe8ad301'
    paths=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
    paths|={r['path'] for r in json.load(open('configs/stage5p1/protected_artifacts.json'))['files']}
    import csv
    paths|={r['path'] for r in csv.DictReader(open('results/tables/stage5p1/checkpoint_inventory.csv'))}
    write_json_new('configs/stage5p2/protected_artifacts.json',dict(base_commit=base,files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
    plan=dict(stage='5.2',base_commit=base,branch='feat/stage5p2-socialvae-baseline',
        protocol='HTraj-SocialVAE-Causal8x12-v1',official_commit='eb23cbae562bac2c2dcf0d81529e6ab2ac280962',
        official_source_sha256={k:sha256(SOURCE/k) for k in EXPECTED},
        manifests={r:dict(path=f'configs/stage5p1/manifests/{r}/B.json',sha256=sha256(f'configs/stage5p1/manifests/{r}/B.json')) for r in ['crowds_zara02','biwi_hotel']},
        runs=[dict(name='Run-A',validation_recording='crowds_zara02'),dict(name='Run-B',validation_recording='biwi_hotel')],
        seed=17,epochs=100,batch_size=128,hidden_dim=256,ob_radius_m=2.,optimizer='Adam',learning_rate=.0005,
        weight_decay=0.,gradient_clip=None,scheduler=None,augmentation=False,
        training_order='numpy default_rng(17+epoch).permutation(all train target indices), one complete pass, retain final partial batch',
        initialization='seed17 fresh original model per run, no checkpoint import or smoke warmstart; matching initial seeds allowed, no trained weight sharing',
        loss='unchanged official mean reconstruction + Monte Carlo latent KL mean; not exact marginal NLL',
        model_selection='minimum validation target-window-weighted K1 latent-prior ADE; exact tie earlier epoch; FDE from same checkpoint',
        checkpoints='save immutable every epoch; best/last refer to those files; strict restore at1,10,20,...100 and selected best',
        evaluation=dict(prior='original deterministic latent-prior-mean trajectory, not expectation over predicted trajectories',
            samples=20,FPC=False,seed_namespace='stage5p2-prior-noise-v1',posterior_seed_namespace='stage5p2-posterior-noise-v1',
            stable_key='protocol|seed17|recording|start_frame|target_id; SHA256 first8 bytes modulo2^63',
            method='external evaluation-only Gaussian standard-noise provider through temporary distribution hooks; original loc/scale, forward, parameters and loss unchanged; remove hooks before training',
            K20='independent ADE and FDE minima per target; first sample is candidate0',
            every_epoch='full validation all metrics plus original supervised posterior objective; same per-target noise across epochs',
            train_K1_epochs=[1,5,10,20,30,40,50,60,70,80,90,100],selected_best_full_train_K1=True),
        history_groups=dict(speed_mps=[.05,.4,1.8],heading_reliable_speed_mps=.2,turn_angle_deg=30.,adjacent_direction_change_deg=45.,
            relative_heading_cosine=.7,radial_speed_threshold_mps=.2,last_neighbor_count_bins=[0,2,5],
            neighbor_density_bins=[0.,1.,3.],radius_m=2.,
            support='only raw eight-frame positions and masks, never future error; keep all/empty groups'),
        CV=dict(models=['CV-Last2','CV-OLS4','CV-OLS8'],primary='CV-Last2',
            source='recompute and verify Stage5.1 target IDs/history/future SHA and existing CV errors',
            oracle_label='解析基线上界参考：事后逐录像最佳CV，非可部署的新模型；ADE/FDE若取各自最低需分别标明'),
        aggregation=['target-window weighted','equal recording-person ID after within-person averaging','two-recording macro separately for each weighting'],
        failure_policy='first nonfinite stops affected run, save model/optimizer/RNG/batch and traceback; no retry or hyperparameter search; independent other preregistered run still attempted',
        embargo='exact Stage5.1 train/dev support; no held-recording train and no promotion of old validation',
        data_root='data/raw/eth_ucy',ETH_evaluated=False,independent_confirmation=False,new_network=False,
        no_extra_runs_or_extension=True,automatic_push=False)
    write_json_new('configs/stage5p2/protocol.json',plan)
    print('Protected files:',len(paths))


if __name__=='__main__':main()
