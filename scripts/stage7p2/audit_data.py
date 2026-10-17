"""Audit all 74 official files and every official split without scoring models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import io
import json
from collections import Counter
import numpy as np
from src.agentformer_benchmark.official import (install_compatibility, configuration,
    read_official_file, enumerate_windows, DATA, SOURCE)
from src.socialvae_baseline.artifacts import json_new
from src.socialvae_causal.io import csv_new
from src.data.preprocessing import sha256

OUT = Path('results/tables/stage7p2')


def main():
    install_compatibility()
    from data.ethucy_split import get_ethucy_split
    from data.dataloader import data_generator
    acquired = json.load(open(OUT/'official_data_acquisition.json'))
    assert acquired['all_files_verified']
    file_data, file_windows, rows, summaries = {}, {}, [], []
    for r in acquired['files']:
        path = Path(r['path'])
        assert sha256(path) == r['sha256']
        a = read_official_file(path)
        windows = enumerate_windows(a)
        key = (path.parent.name, path.stem)
        file_data[key], file_windows[key] = a, windows
        ids, lengths = np.unique(a[:, 1], return_counts=True)
        gaps = np.diff(np.unique(a[:, 0]))
        summaries.append(dict(fold=key[0], sequence=key[1], path=str(path), sha256=r['sha256'],
            git_blob=r['git_blob'], rows=len(a), actor_ids=len(ids),
            frame_min=int(a[:, 0].min()), frame_max=int(a[:, 0].max()),
            coordinate_min=a[:, 2:].min(0).tolist(), coordinate_max=a[:, 2:].max(0).tolist(),
            frame_gaps=np.unique(gaps).tolist(), trajectory_observation_lengths=dict(Counter(map(int, lengths))),
            valid_windows=len(windows), target_windows=sum(w['people'] for w in windows),
            max_people=max(w['people'] for w in windows),
            context_histogram=dict(Counter(w['people'] for w in windows))))
        for w in windows:
            rows.append(dict(w, fold=key[0], sequence=key[1], target_ids=json.dumps(w['target_ids'])))
    split_records, loader_checks, overlaps = {}, [], []
    for fold in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        cfg = configuration(fold)
        splits = dict(zip(['train', 'val', 'test'], get_ethucy_split(fold)))
        held = set(splits['test'])
        assert not held & {s.removesuffix('_train') for s in splits['train']}
        assert not held & {s.removesuffix('_val') for s in splits['val']}
        split_records[fold] = {}
        for split, seqs in splits.items():
            source = [s for s in summaries if s['fold'] == fold and s['sequence'] in seqs]
            assert len(source) == len(seqs)
            split_records[fold][split] = dict(sequences=seqs, files=len(source),
                valid_windows=sum(s['valid_windows'] for s in source),
                targets=sum(s['target_windows'] for s in source),
                max_people=max(s['max_people'] for s in source))
            # Instantiate the original whole split, but only inspect two
            # predetermined nonempty positions in each sequence, never score.
            log = io.StringIO()
            generator = data_generator(cfg, log, split=split, phase='testing')
            assert generator.sequence_to_load == seqs
            for sequence in generator.sequence:
                available = file_windows[(fold, sequence.seq_name)]
                choices = [available[0], max(available, key=lambda w: w['people'])]
                for w in choices:
                    packet = sequence(w['last_observed_frame'])
                    assert list(map(int, packet['valid_id'])) == [int(v) for v in sequence.GetID(sequence.PreData(w['last_observed_frame'])[0]) if int(v) in w['target_ids']]
                    assert sorted(map(int, packet['valid_id'])) == w['target_ids']
                    assert all(v.eq(1).all() for v in packet['pre_motion_mask']+packet['fut_motion_mask'])
                    loader_checks.append(dict(fold=fold, split=split, sequence=sequence.seq_name,
                        frame=w['last_observed_frame'], people=w['people'], exact_IDs_and_complete_masks=True))
            with (OUT/f'loader_{fold}_{split}.txt').open('x') as f:
                f.write(log.getvalue())
        for train in splits['train']:
            name = train.removesuffix('_train')
            a = file_data[(fold, train)]
            b = file_data[(fold, name+'_val')]
            af, bf = set(a[:, 0].astype(int)), set(b[:, 0].astype(int))
            common = af & bf
            # Frame support disjointness is stronger than merely actor IDs.
            overlaps.append(dict(fold=fold, recording=name, shared_frame_count=len(common),
                train_last_frame=int(a[:, 0].max()), val_first_frame=int(b[:, 0].min()),
                separation_ticks=int(b[:, 0].min()-a[:, 0].max())))
    assert not any(r['shared_frame_count'] for r in overlaps)
    cases = []
    for fold in ['hotel', 'zara2']:
        train = split_records[fold]['train']['sequences']
        candidates = [r for r in rows if r['fold'] == fold and r['sequence'] in train]
        picks = [next(r for r in candidates if r['people'] == 1),
            next(r for r in candidates if r['people'] >= 2), max(candidates, key=lambda r: r['people'])]
        seen = set()
        for r in picks:
            key = (r['sequence'], r['last_observed_frame'])
            if key not in seen:
                cases.append(r)
                seen.add(key)
    csv_new(OUT/'official_window_inventory.csv.gz', rows)
    json_new(OUT/'official_data_audit.json', dict(official_files=summaries, splits=split_records,
        train_val_frame_overlap_checks=overlaps, whole_loader_checks=loader_checks, smoke_cases=cases,
        all_expected_recordings_present=True, all_74_file_Git_blobs_verified=True,
        ETH_test_scores_computed=False, source_SHA256=sha256(__file__),
        protocol_source_SHA256={str(SOURCE/p): sha256(SOURCE/p) for p in ['data/ethucy_split.py','data/preprocessor.py','data/dataloader.py','data/convert_ethucy.py']}))
    print(json.dumps(dict(splits=split_records, cases=cases), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
