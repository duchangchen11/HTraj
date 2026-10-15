"""Seal legacy artifacts and freeze a bounded causal-data feasibility study."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import subprocess
from src.data.preprocessing import sha256, write_json_new


def main():
    base = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert base == 'd25da2fa743034338c337bee5d726885b91a2ef1'
    paths = set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
    paths |= {r['path'] for r in json.load(open('configs/stage5/protected_artifacts.json'))['files']}
    paths |= {r['local_path'] for r in json.load(open('results/tables/stage5/fetch_verification.json'))['fetched_or_verified']}
    paths.add('experiments/stage5/assets/socialvae/smoke_seed17_one_step.pth')
    write_json_new('configs/stage5p1/protected_artifacts.json',dict(base_commit=base,
        files=[dict(path=p,sha256=sha256(p)) for p in sorted(paths)]))
    plan = dict(stage='5.1',base_commit=base,branch='feat/stage5p1-socialvae-causal',
        official_commit='eb23cbae562bac2c2dcf0d81529e6ab2ac280962',
        official_source='experiments/stage5/cache/socialvae/eb23cbae562bac2c2dcf0d81529e6ab2ac280962',
        protocols=['SocialVAE-Official-Offline','HTraj-SocialVAE-Causal8x12-v1'],
        observation_frames=8,prediction_frames=12,seconds_per_sample=.4,frame_step=10,coordinate_unit='m',
        data_root='data/raw/eth_ucy',recordings=['biwi_hotel','students001','students003','crowds_zara01','crowds_zara02'],
        development_recordings=['biwi_hotel','crowds_zara02'],
        split_sources={r:dict(path=f'configs/stage3p5/folds/val_{r}.json',sha256=sha256(f'configs/stage3p5/folds/val_{r}.json')) for r in ['biwi_hotel','crowds_zara02']},
        split_policy='Stage3.5 exact train/validation window support; train only old train regions on other recordings; no old validation promoted to train; held-recording original train excluded',
        boundary_policy='backward differences within 8 raw frames; unknown neighbor velocity excludes motion packet at steps1..7; initial position-only social init allowed at step0; ego first acceleration is an explicit zero-acceleration boundary prior with validity=False',
        units='official neural interface: displacements m/step and differences m/step^2; physical derivatives divide by dt and dt^2 only for diagnostics/CV',
        posterior_policy='unchanged official VAE and loss; y used only by supervised posterior/loss; neighbor future slots all padding constructed from history, never future-neighbor GT',
        coverage_stop=dict(blocked_near_motion_fraction_gt=.25,all_history_social_context_lost_target_fraction_gt=.25),
        resource_benchmark=dict(batch_sizes=[16,32,64,128],seed=17,warmup_updates=2,timed_updates=10,
            initialization='fresh random per batch size',optimizer='Adam',learning_rate=.0005,
            selection='fixed seed17 random training target permutation; includes naturally occurring padded context sizes, plus maximum-neighbor batch forward/backward check',
            record_each_step=True,stop_on_nonfinite=True),
        smoke=dict(validation_recording='crowds_zara02',seed=17,epochs=5,batch_size=128,
            batch_fallback='largest successful predeclared batch among128/64/32/16; single fixed choice after resource-only checks',
            epoch_definition='one shuffled pass over every supervised training target; no resampling cap',
            initialization='fresh random; never pretrained',optimizer='Adam',learning_rate=.0005,
            gradient_clip=None,scheduler=None,weight_decay=0.,data_augmentation=False,
            loss='unchanged official reconstruction mean + Monte Carlo latent KL mean',
            best_rule='minimum validation K1 latent-prior-mean ADE; exact tie earliest epoch; paired FDE; last5 separate',
            evaluation='entire frozen validation target set each epoch; no FPC; K20 sampling seed171000+batch_index; diagnostic posterior seed172000+batch_index',
            stop='first nonfinite loss/gradient/output stops run; no LR search or automatic retry; no extension'),
        CV_models=['CV-Last2','CV-OLS4','CV-OLS8'],FPC=False,ETH_evaluated=False,
        formal_baseline_training=False,new_network=False,independent_confirmation=False,automatic_push=False)
    write_json_new('configs/stage5p1/protocol.json',plan)
    print('Protected legacy files:',len(paths))


if __name__=='__main__':main()
