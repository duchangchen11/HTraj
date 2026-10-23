"""CPU-only data identity, canvas shape, all-zero input and order audit.

Only the physically separated train/validation artifacts are loaded. The sealed
test artifact has no entry point. Author image resize/padding arithmetic is
reproduced for canvas shape only; neither segmentation nor GPU is executed.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import time
import cv2
import numpy as np
import torch

from scripts.stage11b0.common import RES, sha, jnew, cnew
from scripts.stage11b0.data_audit import canonical_sha
from scripts.stage11b1c.data import load, manifest, scene_arrays

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / 'configs/stage11c1'
OUT = ROOT / 'results/tables/stage11c1'


def order_sha(values):
    """Same no-trailing-newline SHA definition as original visit_order_SHA."""
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def tensor_sha(value):
    h = hashlib.sha256()
    value = value.detach().contiguous().cpu()
    h.update(b'tensor')
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    started = time.perf_counter()
    previous = json.loads((ROOT / 'results/tables/stage11b1c/data_audit.json').read_text())
    source_config = ROOT / 'configs/stage11b1c/train.json'
    train_config = json.loads(source_config.read_text())
    assert train_config['resize'] == .25 and train_config['semantic_classes'] == 6
    all_frames = {}
    all_rows = {}
    counts = {}
    scenes = {}
    references = {}
    all_outside = []
    manifests = {}
    for split, expected, expected_scenes in [('train', 7496, 25), ('validation', 998, 5)]:
        frame = load(split)
        rows = manifest(split)
        old_path = ROOT / f'results/tables/stage11b1c/{split}_target_manifest.csv'
        inherited_path = ROOT / f'results/tables/stage11b1/{split}_target_manifest.csv'
        assert sha(old_path) == sha(inherited_path) == previous['artifacts'][split]['manifest_SHA']
        assert len(rows) == frame.metaId.nunique() == expected
        assert len(frame) == expected * 20
        assert len(set(row['sample_id'] for row in rows)) == expected
        groups = {int(key): group for key, group in frame.groupby('metaId', sort=False)}
        for row in rows:
            group = groups[int(row['meta_id'])]
            xy = group[['x', 'y']].to_numpy(dtype=float)
            assert len(group) == 20 and group.sceneId.nunique() == group.trackId.nunique() == 1
            assert np.array_equal(np.diff(group.frame.to_numpy()), np.full(19, 12))
            assert group.sceneId.iloc[0] == row['scene']
            assert int(group.trackId.iloc[0]) == int(row['person_id'])
            assert int(group.frame.iloc[0]) == int(row['frame_id'])
            assert canonical_sha(xy[:8]) == row['history_SHA']
            assert canonical_sha(xy[8:]) == row['GT_SHA']
            scene = row['scene']
            if scene not in references:
                path = RES / 'data/SDD/train' / scene / 'reference.jpg'
                image = cv2.imread(str(path))
                assert image is not None
                resized = cv2.resize(image, (0, 0), fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
                h, w = resized.shape[:2]
                height, width = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
                padded = cv2.copyMakeBorder(resized, 0, height-h, 0, width-w, cv2.BORDER_CONSTANT)
                assert padded.shape[:2] == (height, width)
                zero = torch.zeros((1, 6, height, width), dtype=torch.float32)
                assert torch.count_nonzero(zero) == 0 and torch.isfinite(zero).all()
                cached = ROOT / f'experiments/stage11b2c/cache/semantic_{scene}.pt'
                cache_info = None
                if cached.is_file():
                    original_semantic = torch.load(cached, map_location='cpu', weights_only=True)
                    assert tuple(original_semantic.shape) == tuple(zero.shape)
                    assert original_semantic.dtype == torch.float32
                    cache_info = dict(path=str(cached.relative_to(ROOT)), sha256=sha(cached),
                                      shape=list(original_semantic.shape), dtype=str(original_semantic.dtype))
                references[scene] = dict(scene=scene, split=split, RGB_path=str(path.relative_to(ROOT)),
                    RGB_SHA=sha(path), original_height=int(image.shape[0]), original_width=int(image.shape[1]),
                    valid_height=h, valid_width=w, height=height, width=width,
                    semantic_shape=list(zero.shape), semantic_dtype=str(zero.dtype),
                    semantic_zero_SHA=tensor_sha(zero), all_zero_including_padding=True,
                    original_cached_semantic=cache_info)
            info = references[scene]
            assert info['RGB_SHA'] == row['reference_SHA']
            for step, (x, y) in enumerate(xy):
                if not (0 <= x < info['original_width'] and 0 <= y < info['original_height']):
                    all_outside.append(dict(sample_id=row['sample_id'], split=split, step=step, x=x, y=y))
        counts[split] = expected
        scenes[split] = sorted(str(x) for x in frame.sceneId.unique())
        assert len(scenes[split]) == expected_scenes
        all_frames[split], all_rows[split] = frame, rows
        manifests[split] = dict(path=str(old_path.relative_to(ROOT)), sha256=sha(old_path),
                                artifact=previous['artifacts'][split])
    assert not set(scenes['train']) & set(scenes['validation'])
    assert scenes['validation'] == ['bookstore_2', 'deathCircle_3', 'gates_7', 'hyang_7', 'nexus_4']
    assert len(all_outside) == 2 and all(r['split'] == 'train' for r in all_outside)

    # Frozen prediction generation reads this physically history-only artifact.
    hp = ROOT / 'experiments/stage11b2c/cache/validation_history_only.pt'
    assert sha(hp) == 'b484f5b320908e39ad56f91baf677f000afc8328984f31e72ab98f3a0e53c495'
    hist = torch.load(hp, map_location='cpu', weights_only=True)
    assert set(hist) == {'sample_ids', 'scenes', 'history'}
    assert hist['history'].shape == (998, 8, 2)
    assert len(hist['sample_ids']) == len(set(hist['sample_ids'])) == 998
    val_groups = {int(k): g for k, g in all_frames['validation'].groupby('metaId', sort=False)}
    val_index = {r['sample_id']: r for r in all_rows['validation']}
    assert set(hist['sample_ids']) == set(val_index)
    for i, sid in enumerate(hist['sample_ids']):
        row = val_index[sid]
        assert hist['scenes'][i] == row['scene']
        source = val_groups[int(row['meta_id'])][['x', 'y']].iloc[:8].to_numpy(dtype='float32')
        assert torch.equal(hist['history'][i], torch.from_numpy(source))
    history_info = dict(path=str(hp.relative_to(ROOT)), sha256=sha(hp),
                        shape=list(hist['history'].shape), keys=sorted(hist),
                        all998_history_matches_source=True, future_fields_present=False)

    # This audit verifies every exact original epoch order before any new run.
    identity = {int(r['meta_id']): r['sample_id'] for r in all_rows['train']}
    source_epochs = list(csv.DictReader((ROOT / 'results/tables/stage11b1c/epochs.csv').open()))
    assert len(source_epochs) == 30
    ordered_ids = {}
    for scene in scenes['train']:
        _, ids = scene_arrays(all_frames['train'], scene)
        ordered_ids[scene] = [identity[mid] for mid in ids]
    expected_orders = []
    for epoch in range(1, 31):
        order = [scenes['train'][i] for i in torch.randperm(25, generator=torch.Generator().manual_seed(17+epoch-1)).tolist()]
        visits = [sid for scene in order for sid in ordered_ids[scene]]
        old = source_epochs[epoch-1]
        assert int(old['epoch']) == epoch and old['scene_order'] == '|'.join(order)
        assert order_sha(visits) == old['visit_order_SHA']
        assert len(visits) == len(set(visits)) == 7496
        updates = sum((len(ordered_ids[scene])+3)//4 for scene in order)
        assert updates == 1882
        expected_orders.append(dict(epoch=epoch, scene_order='|'.join(order),
            scene_order_SHA=order_sha(order), visit_order_SHA=order_sha(visits),
            targets=7496, unique_targets=7496, optimizer_updates=updates, matches_semantic_baseline=True))

    CONF.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cnew(OUT / 'expected_orders.csv', expected_orders)
    jnew(CONF / 'scene_shapes.json', dict(
        passed=True, source='Actual reference.jpg cv2 INTER_AREA resize .25, author right/bottom division32 padding',
        resize=.25, division_factor=32, scene_count=30, scenes=references))
    jnew(OUT / 'semantic_zero_audit.json', dict(
        passed=True, channels=6, input_channels=14, history_channels=8, dtype='torch.float32',
        shape_source='configs/stage11c1/scene_shapes.json', shape_source_SHA=sha(CONF/'scene_shapes.json'),
        all30_zero_canvases_verified=True, zero_including_padding=True,
        cached_original_shapes_verified=sum(v['original_cached_semantic'] is not None for v in references.values()),
        segmentation_executed=False, RGB_values_used_in_predictor=False,
        zero_tensor_digest_definition="state_digest({'tensor': zero_float32_NCHW})",
        scene_zero_SHA={scene: info['semantic_zero_SHA'] for scene, info in references.items()},
        coordinate_and_canvas_scene_information_retained=True,
        effective_first_layer_semantic_connections_inactive=True,
        equal_nominal_capacity_does_not_imply_equal_effective_capacity=True))
    jnew(OUT / 'data_audit.json', dict(
        passed=True, protocol='HTraj-YNet-SDD-Development-v1', status='PROTOCOL_DEVELOPMENT_ONLY',
        source_manifest_SHA=previous['source_manifest_SHA'], manifests=manifests,
        counts=dict(train=7496, validation=998, sealed_test_not_loaded=2829),
        scenes=scenes, all8494_history_GT_SHA_verified=True,
        recording_overlap=False, target_exclusion=False, out_of_image_points_kept=all_outside,
        history_only_validation_cache=history_info,
        baseline_epochs_path='results/tables/stage11b1c/epochs.csv',
        baseline_epochs_SHA=sha(ROOT/'results/tables/stage11b1c/epochs.csv'),
        expected_orders_path='results/tables/stage11c1/expected_orders.csv',
        expected_orders_SHA=sha(OUT/'expected_orders.csv'), all30_epoch_orders_match=True,
        source_train_config_SHA=sha(source_config),
        independent_confirmation=False, overlapping_windows_independent_samples=False,
        test_loaded=False, GPU_initialized=torch.cuda.is_initialized(),
        limitations=previous['limits'], wall_seconds=time.perf_counter()-started))
    print('PASS: all 8494 target history/GT identities; 30 scene canvases; 30 exact epoch orders; test sealed', flush=True)


if __name__ == '__main__':
    main()
