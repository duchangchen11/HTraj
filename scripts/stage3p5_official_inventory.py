"""Read pinned official dataset tree metadata; do not read trajectory blobs."""
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    repo = Path('data/external/stage3p5/social_stgcnn_metadata')
    commit = 'ebd57aaf34d84763825d05cf9d4eff738d8c96bb'
    command = ['git', '--git-dir=' + str(repo), 'ls-tree', '-r', commit, 'datasets']
    tree = subprocess.check_output(command, text=True)
    entries = []
    for line in tree.splitlines():
        metadata, path = line.split('\t', 1)
        mode, kind, oid = metadata.split()
        entries.append(dict(path=path, mode=mode, type=kind, git_blob_oid=oid))
    folds = {}
    for fold in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
        folds[fold] = {
            split: [e['path'] for e in entries if e['path'].startswith(f'datasets/{fold}/{split}/')]
            for split in ('train', 'val', 'test')
        }
        train_recordings = {Path(p).stem[:-6] for p in folds[fold]['train']}
        val_recordings = {Path(p).stem[:-4] for p in folds[fold]['val']}
        test_recordings = {Path(p).stem for p in folds[fold]['test']}
        assert train_recordings == val_recordings
        assert not train_recordings & test_recordings
    result = dict(
        official_repository='https://github.com/abduallahmohamed/Social-STGCNN',
        source_commit=commit,
        inventory_tree_sha256=hashlib.sha256(tree.encode()).hexdigest(),
        entries=entries,
        folds=folds,
        status='official pinned file inventory verified; content and split boundaries not evaluated',
        initial_api_attempt='HTTP403 rate limit; subsequently read public filtered bare Git tree',
        trajectory_blobs_read=False,
        ETH_trajectory_or_metrics_read=False,
        external_model_replicated=False,
        HTraj_nonETH_pool=['biwi_hotel', 'crowds_zara01', 'crowds_zara02', 'students001', 'students003'],
        additional_official_ETH_fold_train_val_recordings=['crowds_zara03', 'uni_examples'],
        implication='Official training population is larger than HTraj; not an official protocol reproduction or fair paper metric comparison.',
    )
    out = Path('results/tables/stage3p5/external_official_split_inventory.json')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(status=result['status'], folds={k: {s: len(v) for s, v in d.items()} for k, d in folds.items()}, ETH_evaluated=False)))


if __name__ == '__main__':
    main()
