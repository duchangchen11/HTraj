"""Freeze two historical-exposure LORO folds and all decision rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.benchmark_audit import BenchmarkDataset, assert_recording_isolation
from src.data.collate import collate_trajectories
from src.data.preprocessing import sha256, write_json_new
from src.evaluation.frozen_ego import history_groups
from src.training.social_diagnostics import csv_new

ROOT = Path('results/tables/stage3p6')
CONF = Path('configs/stage3p6')
VIEW = 'A_complete8x12_history_context'


def main():
    torch.set_num_threads(1)
    folds, counts, motion, source_hashes = [], [], [], {}
    old = json.loads(Path('configs/data_manifests/stage2_v1/fold_ETH.json').read_text())
    original_train = {w['sample_id'] for w in old['splits']['train']}
    original_val = {w['sample_id'] for w in old['splits']['validation']}
    for rec in ('crowds_zara02', 'biwi_hotel'):
        path = Path('configs/stage3p5/folds/val_'+rec+'.json')
        m = json.loads(path.read_text()); assert_recording_isolation(m)
        assert {w['sample_id'] for w in m['splits']['train']} <= original_train
        assert not {w['sample_id'] for w in m['splits']['train']} & original_val
        assert {w['recording'] for w in m['splits']['validation']} == {rec}
        fold = dict(recording=rec, manifest=str(path), manifest_sha256=sha256(path), indices={})
        for split in ('train', 'validation'):
            data = BenchmarkDataset('data/raw/eth_ucy', path, split, VIEW)
            indices, supervised, groups, fields = [], 0, {}, {}
            for bi, b in enumerate(DataLoader(data, batch_size=8, collate_fn=collate_trajectories)):
                gs, fs = history_groups(b); target = b['future_valid'].any(1)
                assert torch.equal(target, b['future_valid'].all(1))
                indices.extend(bi*8+i for i in range(len(b['sample_id'])) if target[i].any())
                supervised += int(target.sum())
                for name, mask in gs.items():
                    groups[name] = groups.get(name, 0)+int((target & mask).sum())
                for name in ('cutoff_speed_mps', 'cutoff_velocity_x_mps', 'cutoff_velocity_y_mps', 'history_velocity_change_mps', 'mean_neighbor_degree', 'relative_velocity_change_mps'):
                    fields.setdefault(name, []).extend(fs[name][target].tolist())
            fold['indices'][split] = indices
            counts.append(dict(recording=rec, split=split, manifest_sha256=sha256(path), total_windows=len(data), effective_windows=len(indices), targets=supervised, future_points=supervised*12, **groups))
            for name, values in fields.items():
                motion.append(dict(recording=rec, split=split, history_measure=name, target_windows=len(values), mean=float(np.mean(values)), p05=float(np.quantile(values, .05)), median=float(np.median(values)), p95=float(np.quantile(values, .95))))
        for name, info in m['source_files'].items():
            source = Path('data/raw/eth_ucy')/info['file']
            assert info['scene'] != 'ETH' and sha256(source) == info['sha256']
            source_hashes[name] = dict(info, locally_verified_sha256=sha256(source))
        folds.append(fold)
    protocol = dict(stage='3.6', protocol_name=VIEW, folds=folds, raw_root='data/raw/eth_ucy', source_files=source_hashes,
        seed17=17, conditional_seeds=[42, 2026], epochs=30, extension_epochs=0, batch_size=8, hidden_dim=64, prediction_steps=12,
        sampling_seconds=.4, optimizer='Adam default betas/eps; weight_decay0', learning_rate=.001, gradient_clip=1., cpu_threads=1,
        schedules=['constant', 'step10_20'], schedule_rule='step10_20: epochs1-10=.001,11-20=.0003,21-30=.00009; no other tuning',
        ego_models=['CV', 'PedBase', 'CV-Residual'], neural_ego_runs=['PedBase/constant', 'PedBase/step10_20', 'CV-Residual/constant', 'CV-Residual/step10_20'],
        residual_models=['Ego-Only', 'Ego-Residual', 'Social-Previous', 'Social-Current', 'Social-Random'],
        residual_schedule='constant', branch_initialization='same seed, last correction layer zero, one scalar gate logit=-2',
        loss='unchanged masked per-track mean Euclidean L2; complete targets have12 points',
        best_rule='minimum own-recording validation ADE; earliest exact tie; paired FDE; last30 also retained',
        anchor_rule='per recording choose lowest best-ADE neural candidate, deterministic order neural_ego_runs breaks ties; if analytical CV ADE<=best neural ADE, use CV prediction with that neural encoder/head frozen; otherwise use selected neural output; same checkpoint/output rule for every residual control',
        base_stability_rule='all4 neural runs per recording complete30 epochs with finite loss/gradients/metrics and all structural tests pass; nonfinite failure stops residual training; analytical CV remains eligible anchor when it beats neural candidates',
        graph=dict(radius_m=4., max_neighbors=8, CPA_seconds=4.8, eligibility='both endpoints two consecutive historical observations', weights='uniform incoming neighbors, uniform mean over frames with messages; target gap resets pooled history'),
        random_rule='retrain branch with coordinate/time-hash ranked same-frame/same-batch/radius4/continuous/no-self candidates; match real incoming budget except entire hash ties; report actual changed fraction and distance/budget shift; weak swapping cannot establish no relational value',
        round3_rule='a true social variant must beat both its own frozen Ego-Only and matched Ego-Residual by at least max(.005m,.005*reference_ADE) in BOTH recordings and not worsen pairedFDE against either; only then run42/2026, freeze per-recording anchor model/schedule/output family selected at17 and train it from scratch for each seed, then all4 residual controls with same seed-specific anchor; no seed replacement',
        conditional_candidate_priority=['Social-Current', 'Social-Previous', 'Social-Random'],
        history_groups=dict(CV_like_mean_adjacent_velocity_change_mps_lt=.2, changing_ge=.5, relative_velocity_change_ge=.5, motion_cosine=.7, moving_speed_min=.2),
        CV_large_error_group='after completed training only: stored analytical per-target CV ADE>=1.0m; analysis only, never input/gate/checkpoint selection',
        independent_confirmation=False, ETH_evaluated=False, B_run=False, old_validation_promoted_to_train=False,
        endpoint='seed17 two recordings:8 ego neural runs +4 residual trained branches per recording if finite stability passes; conditional stability only under frozen round3 rule; no extension or architecture/LR search')
    write_json_new(CONF/'protocol.json', protocol)
    csv_new(ROOT/'split_counts.csv', counts); csv_new(ROOT/'historical_motion_distribution.csv', motion)
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
