"""Source histories and identity/label hashes, no prediction errors computed."""
from common import *
from collections import Counter
import math
import time
import numpy as np
from src.socialvae_causal.data import CausalDataset, array_sha


def main():
    start = time.perf_counter()
    baseline = json.loads((ROOT / 'configs/stage5p1/protocol.json').read_text())
    old = json.loads((ROOT / 'configs/data_manifests/stage2_v1/fold_ETH.json').read_text())
    allowed_train = {(w['recording'], w['start_frame']) for w in old['splits']['train']}
    old_validation = {(w['recording'], w['start_frame']) for w in old['splits']['validation']}
    nearest, histories, seen_windows = [], [], set()
    summary, metadata, contexts, manifests = [], [], [], []
    for fold in ('biwi_hotel', 'crowds_zara02'):
        path = ROOT / 'configs/stage5p1/manifests' / fold / 'B.json'
        manifests.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        for split in ('train', 'validation'):
            ds = CausalDataset(ROOT / baseline['data_root'], path, split)
            train_sources = {w['recording'] for w in ds.manifest['splits']['train']}
            assert fold not in train_sources and 'biwi_eth' not in ds.records
            for w in ds.windows:
                key = (w['recording'], w['start_frame'])
                assert key in (allowed_train if split == 'train' else old_validation)
                assert split != 'train' or key not in old_validation
            old_rows = read_csv(ROOT / 'results/tables/stage5p1/manifests' / fold / ('B_' + split + '_targets.csv.gz'))
            expected = {(r['recording'], int(r['start_frame']), int(r['target_id'])): r for r in old_rows}
            rows, nlist, missing = [], [], 0
            for wi, c in enumerate(ds.contexts):
                w = ds.windows[wi]
                n = len(c.ids)
                nlist.append(n)
                missing += int((c.position_valid & ~c.velocity_valid).sum())
                contexts.append(dict(fold=fold, split=split, sample_id=w['sample_id'], source=w['recording'],
                    frame_id=w['start_frame'], context_people=n, complete8=int(c.position_valid.all(0).sum()),
                    partial_history=int((c.position_valid.any(0) & ~c.position_valid.all(0)).sum()),
                    unknown_velocity=int((c.position_valid & ~c.velocity_valid).sum())))
                key = (w['recording'], w['start_frame'])
                if split == 'train' and key not in seen_windows:
                    seen_windows.add(key)
                    p = c.state[..., :2]
                    r = p[:, :, None] - p[:, None, :]
                    d = np.linalg.norm(r, axis=-1)
                    legal = c.position_valid[:, :, None] & c.position_valid[:, None, :] & ~np.eye(n, dtype=bool)
                    near = np.min(np.where(legal, d, np.inf), axis=-1)
                    nearest.extend(near[np.isfinite(near)].tolist())
                    histories.append((d, legal))
            for i, (wi, target_index) in enumerate(ds.indices):
                e, c, w = ds[i], ds.contexts[wi], ds.windows[wi]
                key = (e['recording'], e['start_frame'], e['actor'])
                ref = expected.pop(key)
                hs, gs = array_sha(e['x'][:, :2]), array_sha(e['y'])
                assert hs == ref['history_sha256'] and gs == ref['future_sha256']
                rows.append(dict(sample_id=w['sample_id'] + ':' + str(e['actor']), source=e['recording'],
                    frame_id=e['start_frame'], last_observed_frame=e['start_frame'] + 70,
                    person_id=e['actor'], history_SHA=hs, GT_SHA=gs,
                    context_history_SHA=array_sha(c.state[..., :2]), context_mask_SHA=array_sha(c.position_valid),
                    context_ID_SHA=array_sha(c.ids), history_frames=8, future_frames=12))
            assert not expected
            write_csv(OUT / (fold + '_' + split + '_targets.csv.gz'), rows)
            counts = Counter((r['source'], r['person_id']) for r in rows)
            summary.append(dict(fold=fold, split=split, windows=len(ds.windows), supervised_windows=len({x['frame_id'] for x in rows}) if split == 'validation' else len({(x['source'], x['frame_id']) for x in rows}),
                targets=len(rows), source_person_IDs=len(counts), max_windows_per_person=max(counts.values()),
                context_N_quantiles=np.quantile(nlist, [0,.5,.95,1]).tolist(), unknown_velocity_positions=missing,
                train_validation_recordings_disjoint=True, old_validation_promoted=False,
                target_identity_and_GT_equal_stage5p1=True))
            for name, r in ds.records.items():
                p = ROOT / baseline['data_root'] / (name + '.txt')
                if not any(x['path'] == str(p.relative_to(ROOT)) for x in metadata):
                    metadata.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size))
    # Freeze deterministic, coarse scales from pooled unique SOURCE TRAIN histories.
    # Neither target futures nor validation trajectories determine these values.
    q = np.quantile(nearest, [.25,.5,.75,.95,.99]).tolist()
    radius = max(2., min(8., math.ceil(q[3] * 2) / 2))
    degree = np.concatenate([(mask & (dist <= radius)).sum(-1).ravel() for dist,mask in histories])
    k = int(max(1, min(8, math.ceil(float(np.quantile(degree, .75))))))
    scale = max(.5, min(2., round(q[1] * 2) / 2))
    config = dict(radius=radius, max_neighbors=k, horizon=4.8, distance_scale=scale, dt=.4, speed_epsilon=1e-4)
    write_json(CONF / 'graph.json', config)
    write_csv(OUT / 'context_counts.csv.gz', contexts)
    write_json(OUT / 'data_audit.json', dict(summary=summary, source_files=metadata, manifests=manifests,
        threshold_basis=dict(source='pooled unique SOURCE TRAIN histories only', unique_windows=len(seen_windows),
            nearest_distance_quantiles=q, quantile_levels=[.25,.5,.75,.95,.99], radius_rule='ceil(2*q95)/2 clipped [2,8]m',
            k_rule='ceil(q75 visible source degrees within frozen radius) clipped[1,8]',
            scale_rule='nearest distance median rounded to .5m clipped[.5,2]',
            degree_quantiles=np.quantile(degree, [0,.5,.75,.95,1]).tolist()),
        graph_config=config, graph_config_SHA=sha(CONF / 'graph.json'), seconds=time.perf_counter()-start,
        no_prediction_error_computed=True, ETH_opened=False, execution_commit=commit()))
    print(json.dumps(summary, indent=2))
    print('Frozen graph configuration', config, flush=True)


if __name__ == '__main__':
    main()
