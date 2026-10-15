"""Freeze all five manifests, formulas, thresholds and historical references."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import subprocess
from src.data.preprocessing import sha256, write_json_new
from src.data.benchmark_audit import assert_recording_isolation


def candidate(path, checkpoint=None, kind='PedBase'):
    s = json.loads(Path(path).read_text())
    checkpoint = checkpoint or s['best_checkpoint_path']
    assert sha256(checkpoint) == s['best_checkpoint_sha256']
    return dict(summary=path, summary_sha256=sha256(path), checkpoint=checkpoint,
        checkpoint_sha256=s['best_checkpoint_sha256'], kind=kind, seed=17,
        historical_best_ADE_m=s['best_ADE_m'], historical_paired_FDE_m=s['best_FDE_m'],
        source_commit=s['source_commit'], manifest_sha256=s['manifest_sha256'])


def main():
    conf = Path('configs/stage3p8'); conf.mkdir(exist_ok=False)
    root = Path('results/tables/stage3p8'); root.mkdir(exist_ok=False)
    base = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    tracked = [x for x in subprocess.check_output(['git','ls-files','-z']).decode().split('\0') if x]
    weights = [str(x) for x in Path('experiments').rglob('*.pt')]
    files = sorted(set(tracked+weights))
    write_json_new(conf/'protected_artifacts.json',dict(base_commit=base,tracked_files=len(tracked),old_checkpoints=len(weights),files=[dict(path=x,sha256=sha256(x)) for x in files]))
    folds=[]; sources={}; refs=[]; coverage=[]
    for rec in ('biwi_hotel','students001','students003','crowds_zara01','crowds_zara02'):
        path=Path('configs/stage3p5/folds')/('val_'+rec+'.json'); m=json.loads(path.read_text()); assert_recording_isolation(m)
        assert m['seconds_per_sample']==.4 and m['observation_frames']==8 and m['prediction_frames']==12
        assert {w['recording'] for w in m['splits']['validation']}=={rec}
        folds.append(dict(recording=rec,manifest=str(path),manifest_sha256=sha256(path),validation_windows=len(m['splits']['validation'])))
        for name, info in m['source_files'].items():
            assert info['scene']!='ETH'; assert sha256(Path('data/raw/eth_ucy')/info['file'])==info['sha256']; sources[name]=info
        if rec in ('biwi_hotel','crowds_zara02'):
            m0=candidate(f'results/tables/stage3p7/{rec}/M0/seed_17/summary.json')
            ped=[m0]+[candidate(f'results/tables/stage3p6/{rec}/ego/PedBase__{schedule}/seed_17/summary.json') for schedule in ('constant','step10_20')]
            cvrs=[candidate(f'results/tables/stage3p6/{rec}/ego/CV-Residual__{schedule}/seed_17/summary.json',kind='CVResidual') for schedule in ('constant','step10_20')]
            cvrs.append(candidate(f'results/tables/stage3p7/{rec}/M3/seed_17/summary.json',kind='HeadingCVResidual'))
            for label, candidates in [('PedBase-M0',[m0]),('PedBase-BestCompatible',ped),('CVResidual-BestCompatible',cvrs)]:
                selected=min(candidates,key=lambda x:x['historical_best_ADE_m'])
                refs.append(dict(recording=rec,label=label,selected=selected,candidates=candidates,selection='minimum previously published validation ADE, fixed before Stage3.8; historical exposure, no fresh checkpoint selection'))
                coverage.append(dict(recording=rec,model=label,status='existing_compatible_checkpoint',reason='same LORO manifest/protocol, train recording exclusion and old source hashes verified at evaluation'))
        elif rec=='students003':
            path='results/tables/stage3p5/pilot/A_complete8x12_history_context/PedBase/summary.json'
            c=candidate(path,'experiments/stage3p5/pilot/A_complete8x12_history_context/PedBase/checkpoints/best.pt')
            for label in ('PedBase-M0','PedBase-BestCompatible'):
                refs.append(dict(recording=rec,label=label,selected=c,candidates=[c],selection='only compatible historical PedBase run; no new training'))
                coverage.append(dict(recording=rec,model=label,status='existing_compatible_checkpoint',reason='Stage3.5 all effective A train2058 / val70; original architecture and protocol'))
            coverage.append(dict(recording=rec,model='CVResidual-BestCompatible',status='not_available',reason='no same-recording-isolated CVResidual checkpoint; no training authorized'))
        else:
            for label in ('PedBase-M0','PedBase-BestCompatible','CVResidual-BestCompatible'):
                coverage.append(dict(recording=rec,model=label,status='not_available',reason='no corresponding LORO checkpoint; other-fold models may have trained on this recording and cannot replace it'))
    p=dict(stage='3.8',base_commit=base,protocol_name='A_complete8x12_history_context',raw_root='data/raw/eth_ucy',folds=folds,source_files=sources,
        models=['CV-Last2','CV-OLS4','CV-OLS8','CV-Median'],trainable_parameters=0,new_neural_training=False,ETH_evaluated=False,B_run=False,
        history_frames=8,prediction_frames=12,sampling_seconds=.4,batch_size=8,cpu_threads=1,analytic_dtype='float64 from unchanged float32 loader coordinates',neural_dtype='float32 frozen historical weights',
        prediction='last observed position + estimated velocity * (1..12)*0.4; OLS never replaces last position with fitted intercept',
        formulas=dict(Last2='(x7-x6)/(t7-t6)',OLS='sum((t-tbar)*(x-xbar))/sum((t-tbar)^2), last4 or all8; time centered at t7 and space at x7',
            Median='componentwise median of seven observed adjacent velocities; average two central values for even count; axis-dependent, not generally rotation-equivariant'),
        missing_history='current contiguous suffix only; reset at absent frame or interval>0.6s; OLS up toK available points, min2; median available observed velocities; <2 -> explicit stationary fallback',
        groups=dict(speed_vector_mean_low_lt=.2,speed_vector_mean_high_ge=1.,quasi_stationary='max speed<=.1m/s AND path<=.2m AND net<=.15m, history proxy only',
            out_and_back='path>=.5m AND net/path<=.5',direction_changing='angular span among any two historical velocities with speed>=.2m/s >=60deg OR known last-mean angle>=60deg',
            terminal_unstable='norm(last_velocity - time_weighted_mean_velocity)>=.5m/s',variance_bins_m2ps2=[.01,.25],linear_fit_RMS_bins_m=[.02,.1],
            direction_consistent='net/path>=.9 and known last-mean angle<30deg',angle_known='both last and vector-mean speeds>=.2m/s',
            stationary_ratio='zero-length path ratio numerical1 with defined=False; out/back requires path>=.5m',observable_length='all supervised A targets must have8 observations; partial/unknown groups retained at N0'),
        timing=dict(device='CPU',warmup_passes=3,repeat_passes=20,measure='time perf_counter for each full forward incl motion preprocessing, excluding I/O/groups/GT/error computation; same cached observation batches, single thread'),
        noise_diagnostic='historical OLS8 in-sample RMS vs one-step historical last2 backtest RMS over targets t2..t7; diagnostic only, not future evidence of true velocity noise',
        analysis='all groups; per-recording metrics; equal-recording macro vs target-window weighted separate; per-person mean paired differences and person-macro metrics; Pearson/Spearman without iid-window p-values',
        candidate_rule='ADE reduction>=max(.005m, .5%Last2ADE) in >=3/5 recordings with pairedFDE no worse than +max(.005m,.01*Last2FDE) in those; five-recording macro ADE improves and macroFDE does not worsen; remaining recording failures always reported; recommend prior only, not learning or submission claim',
        stronger_reference_rule='compare candidates with M0, best-compatible PedBase and historical best compatible CVResidual on exact same targets where present; no five-recording neural-superiority claim with missing coverage; matched coverage aggregates only',
        references=refs,reference_coverage=coverage,independent_confirmation=False,extra_estimators_allowed=False,learned_gate=False,auto_push=False,
        endpoint='evaluate four fixed analytic models on all five validation recordings; frozen compatible neural inference only; no estimator/hyperparameter search, clipping, extra training or ETH')
    write_json_new(conf/'protocol.json',p)
    print(json.dumps(dict(protected_files=len(files),old_checkpoints=len(weights),folds=len(folds),references=len(refs),new_neural_training=False)))


if __name__=='__main__':main()
