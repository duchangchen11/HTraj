from pathlib import Path
import json
import numpy as np
import pytest
import torch
from src.data.preprocessing import Recording, SCENE_FILES, build_manifests
from src.data.dataset import TrajectoryDataset
from src.data.collate import collate_trajectories


def make_recording(path, length=240):
    rows = []
    for t in range(length):
        rows.append([t*10, 1, t*.04, 0])
        if t < 10:
            rows.append([t*10, 2, t*.04, 1])
        if t < 3:
            rows.append([t*10, 3, t*.04, 2])
        if t >= 8:
            rows.append([t*10, 99, t*.04, 3])
    np.savetxt(path, rows)


def test_history_agent_set_and_future_supervision_separate(tmp_path):
    path = tmp_path/'toy.txt'
    make_recording(path)
    rec = Recording.read(path, 'ETH')
    sample = rec.sample(0)
    assert sample['agent_ids'].tolist() == [1, 2, 3]
    assert sample['observed'].shape == (8, 3, 2)
    assert sample['future_valid'][:, 1].sum() == 2  # leaving target
    assert not sample['future_valid'][:, 2].any()  # history-only context
    assert np.allclose(np.diff(sample['timestamps']), .4)
    for frame in range(80, 200, 10):
        rec.by_frame[frame] = {1: np.array([999., 999.], np.float32), 88: np.array([1., 1.], np.float32)}
    changed = rec.sample(0)
    for name in ('observed', 'observed_valid', 'agent_ids', 'timestamps'):
        assert np.array_equal(sample[name], changed[name])
    observation_only = rec.sample(0, include_supervision=False)
    assert not observation_only['future_valid'].any()


def test_collation_padding_and_ids(tmp_path):
    path = tmp_path/'toy.txt'
    make_recording(path)
    rec = Recording.read(path, 'ETH')
    samples = [rec.sample(0), rec.sample(300)]
    for sample in samples:
        for key, value in list(sample.items()):
            if isinstance(value, np.ndarray):
                sample[key] = torch.from_numpy(value)
    batch = collate_trajectories(samples)
    assert batch['observed'].shape == (2, 8, 3, 2)
    assert not batch['observed_valid'][1, :, 2].any()
    assert not batch['future_valid'][1, :, 2].any()
    assert batch['agent_ids'][1, 2] == -1
    assert batch['recording'] == ['toy', 'toy']


def test_reject_malformed_duplicate_and_nonfinite_rows(tmp_path):
    path = tmp_path/'bad.txt'
    for rows in [ [[0, 1, 2]], [[0, 1, 2, 3], [0, 1, 3, 4]], [[0, 1, np.nan, 3]] ]:
        np.savetxt(path, rows)
        with pytest.raises(ValueError):
            Recording.read(path, 'ETH')


def test_windows_do_not_bridge_global_empty_gap(tmp_path):
    path = tmp_path/'gap.txt'
    np.savetxt(path, [[t*10, 1, t*.1, 0] for t in list(range(12))+list(range(20, 32))])
    rec = Recording.read(path, 'ETH')
    assert list(rec.windows()) == []


def test_five_fold_time_isolation_and_hash_guard(tmp_path):
    raw, output = tmp_path/'raw', tmp_path/'manifests'
    raw.mkdir()
    for files in SCENE_FILES.values():
        for name in files:
            make_recording(raw/name)
    build_manifests(raw, output)
    for path in output.glob('fold_*.json'):
        manifest = json.loads(path.read_text())
        held = manifest['held_out_scene']
        assert {x['scene'] for x in manifest['splits']['test']} == {held}
        assert held not in {x['scene'] for split in ('train', 'validation') for x in manifest['splits'][split]}
        def support(split):
            return {(w['recording'], f) for w in manifest['splits'][split]
                    for f in range(w['start_frame'], w['end_frame']+1, 10)}
        assert not (support('train') & support('validation'))
    dataset = TrajectoryDataset(raw, output/'fold_ETH.json', 'train')
    assert 'biwi_eth' not in dataset.recordings
    assert len(dataset[0]['agent_ids']) > 0
    with (raw/'biwi_hotel.txt').open('a') as stream:
        stream.write('2400 1 0 0\n')
    with pytest.raises(ValueError, match='hash changed'):
        TrajectoryDataset(raw, output/'fold_ETH.json', 'train')
    with pytest.raises(FileExistsError):
        build_manifests(raw, output)
