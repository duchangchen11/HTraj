"""Verify frozen cohorts and enumerate causal context coverage without models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
from collections import Counter
from src.agentformer_admission.causal import build_context, labels_for_context
from src.data.preprocessing import Recording, sha256
from src.socialvae_causal.data import array_sha
from src.socialvae_causal.io import csv_new, read_csv
from src.socialvae_baseline.artifacts import json_new

OUT = Path('results/tables/stage7p1')


def main():
    protocol = json.load(open('configs/stage7p1/admission_protocol.json'))
    summaries, selected, all_rows, data_sha = [], [], [], {}
    records = {}
    for fold, source in protocol['manifests'].items():
        assert sha256(source['path']) == source['sha256']
        m = json.load(open(source['path']))
        assert {w['recording'] for w in m['splits']['validation']} == {fold}
        assert fold not in {w['recording'] for w in m['splits']['train']}
        assert not m['original_validation_promoted_to_train'] and not m['ETH_evaluated']
        for name, f in m['source_files'].items():
            assert name != 'biwi_eth'
            p = Path('data/raw/eth_ucy')/f['file']
            assert sha256(p) == f['sha256']
            data_sha[str(p)] = f['sha256']
            if name not in records:
                records[name] = Recording.read(p, f['scene'])
        for split in ['train', 'validation']:
            targets_path = f'results/tables/stage5p1/manifests/{fold}/B_{split}_targets.csv.gz'
            expected = read_csv(targets_path)
            expected_map = {(r['recording'], int(r['start_frame']), int(r['target_id'])): r for r in expected}
            actual, rows = set(), []
            for w in m['splits'][split]:
                r = records[w['recording']]
                c = build_context(r, w['start_frame'])
                y, valid = labels_for_context(c, r)
                supervised = c.ids[valid.all(1)]
                for i, actor in enumerate(c.ids):
                    if not valid[i].all():
                        continue
                    key = (r.name, c.start, int(actor))
                    assert key in expected_map, key
                    ref = expected_map[key]
                    assert ref['history_sha256'] == array_sha(c.history[i])
                    assert ref['future_sha256'] == array_sha(y[i])
                    actual.add(key)
                rows.append(dict(fold=fold, split=split, recording=r.name, start_frame=c.start,
                    people=len(c.ids), targets=len(supervised), partial_history_excluded=len(c.partial_ids),
                    incomplete_future_nodes=int((~valid.all(1)).sum()),
                    no_future_nodes=int((valid.sum(1) == 0).sum()),
                    ids_json=json.dumps(c.ids.tolist()), target_ids_json=json.dumps(supervised.tolist()),
                    partial_ids_json=json.dumps(c.partial_ids.tolist()),
                    future_lengths_json=json.dumps(valid.sum(1).tolist())))
            assert actual == set(expected_map)
            counts = Counter(r['people'] for r in rows)
            summaries.append(dict(fold=fold, split=split, windows=len(rows),
                windows_with_targets=sum(r['targets'] > 0 for r in rows),
                targets=len(actual), exact_ids_history_and_GT_SHA=True,
                target_manifest=targets_path, target_manifest_sha256=sha256(targets_path),
                max_people=max(counts), person_count_histogram=dict(sorted(counts.items())),
                total_full8_node_windows=sum(r['people'] for r in rows),
                total_partial_node_windows_excluded=sum(r['partial_history_excluded'] for r in rows),
                mixed_label_windows=sum(r['targets'] > 0 and r['incomplete_future_nodes'] > 0 for r in rows),
                source_manifest=source, valid_supports_unchanged=True))
            if split == 'validation':
                eligible = [r for r in rows if r['targets']]
                selectors = dict(single=lambda r: r['people'] == 1,
                    two=lambda r: r['people'] == 2,
                    multi_all_labeled=lambda r: r['people'] >= 3 and not r['incomplete_future_nodes'],
                    mixed_future=lambda r: r['incomplete_future_nodes'] > 0,
                    partial_history=lambda r: r['partial_history_excluded'] > 0)
                picks = {}
                for reason, test in selectors.items():
                    match = next((r for r in eligible if test(r)), None)
                    if match is not None:
                        picks.setdefault((match['recording'], match['start_frame']), dict(**match, reasons=[]))['reasons'].append(reason)
                maximum = max(eligible, key=lambda r: r['people'])
                picks.setdefault((maximum['recording'], maximum['start_frame']), dict(**maximum, reasons=[]))['reasons'].append('maximum_validation_context')
                selected.extend(picks.values())
            all_rows.extend(rows)
    csv_new(OUT/'context_windows.csv.gz', all_rows)
    maximum = max(all_rows, key=lambda r: r['people'])
    json_new(OUT/'data_audit.json', dict(protocol=protocol, summaries=summaries,
        raw_data_sha256=data_sha, cases=selected, maximum_source_context=maximum,
        original_stage7_max_N=57, ETH_evaluated=False, source_SHA256=sha256(__file__)))
    print(json.dumps(dict(summaries=summaries, cases=selected, maximum=maximum), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
