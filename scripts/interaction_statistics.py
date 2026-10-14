import argparse
from pathlib import Path
import sys
import csv
import json
import time
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import Recording, sha256
from src.models.ego_encoder import motion_inputs
from src.models.temporal_graph import select_candidates
from src.evaluation.interaction_coverage import count_ordered_walks


def bucket_name(value):
    if value == 0:
        return '0'
    if value <= 1:
        return '(0,1]'
    if value <= 3:
        return '(1,3]'
    return '>3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw/eth_ucy')
    parser.add_argument('--manifest', default='configs/data_manifests/stage2_v1/observation_windows.json')
    parser.add_argument('--output', default='results/tables/stage2_interaction_statistics.csv')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Do not overwrite statistics: '+str(output))
    torch.set_num_threads(1)
    manifest = json.loads(Path(args.manifest).read_text())
    records = {}
    for name, info in manifest['source_files'].items():
        path = Path(args.raw)/info['file']
        if sha256(path) != info['sha256']:
            raise ValueError('Hash mismatch')
        records[name] = Recording.read(path, info['scene'])
    # Fixed before reading results; descriptive sensitivity only, no tuning.
    specs = [(2., 8), (4., 8), (8., 8), (4., 4), (4., 12)]
    groups = defaultdict(lambda: dict(people=[], degrees=[], count=np.zeros(3, np.int64),
                                      returns=np.zeros(3, np.int64), covered=np.zeros(3, np.int64),
                                      final_covered=np.zeros(3, np.int64), final_people=0,
                                      nonreturn2=0, comparisons=0, changes=0, jaccards=[],
                                      only_first=0, windows=0, observed_steps=0, eligible_steps=0,
                                      edges=0, individuals=0))
    start = time.perf_counter()
    for index, entry in enumerate(manifest['windows']):
        rec = records[entry['recording']]
        # Does not look up future coordinates or future validity.
        sample = rec.sample(entry['start_frame'], include_supervision=False)
        x = torch.from_numpy(sample['observed'])[None]
        valid = torch.from_numpy(sample['observed_valid'])[None]
        encoded = motion_inputs(x, valid, torch.from_numpy(sample['timestamps'])[None])
        people = x.shape[2]
        active, cont = sample['observed_valid'], encoded['continuity'][0].numpy()
        for radius, neighbors in specs:
            masks = np.stack([select_candidates(encoded['positions'][:, t], encoded['velocity'][:, t],
                                encoded['continuity'][:, t], radius, neighbors)[0][0].numpy() for t in range(8)])
            walk = count_ordered_walks(masks, cont)
            new, returns, final = walk['new_by_agent'], walk['return_by_agent'], walk['final_counts']
            degrees = masks.sum((0, 2))/np.maximum(cont.sum(0), 1)
            all_group = groups[(entry['scene'], radius, neighbors, 'all')]
            all_group['people'].append(people)
            all_group['windows'] += 1
            all_group['observed_steps'] += int(active.sum())
            all_group['eligible_steps'] += int(cont.sum())
            all_group['edges'] += int(masks.sum())
            all_group['only_first'] += int(new[0].sum() > 0 and new[1].sum() == 0)
            for t in range(2, len(masks)):
                previous, current = masks[t-1], masks[t]
                union = int((previous | current).sum())
                if union:
                    all_group['comparisons'] += 1
                    all_group['changes'] += int(not np.array_equal(previous, current))
                    all_group['jaccards'].append(float((previous & current).sum()/union))
            for person in range(people):
                for bucket in ('all', bucket_name(degrees[person])):
                    group = groups[(entry['scene'], radius, neighbors, bucket)]
                    group['individuals'] += 1
                    group['degrees'].append(float(degrees[person]))
                    group['count'] += new[:, person]
                    group['returns'] += returns[:, person]
                    group['covered'] += (new[:, person] > 0)
                    group['nonreturn2'] += int(new[1, person]-returns[1, person] > 0)
                    if active[-1, person]:
                        group['final_people'] += 1
                        group['final_covered'] += (final[:, person] > 0)
        if (index+1) % 500 == 0:
            print(f'coverage {index+1}/{len(manifest["windows"])} windows; {time.perf_counter()-start:.1f}s', flush=True)
    rows = []
    for (scene, radius, neighbors, bucket), group in sorted(groups.items()):
        total = group['individuals']
        rows.append(dict(scene=scene, radius_m=radius, max_neighbors=neighbors, neighbor_bin=bucket,
            windows=group['windows'] if bucket == 'all' else '', person_window_occurrences=total,
            historical_people_mean=float(np.mean(group['people'])) if group['people'] else '',
            historical_people_p50=float(np.median(group['people'])) if group['people'] else '',
            historical_people_p95=float(np.quantile(group['people'], .95)) if group['people'] else '',
            historical_people_max=max(group['people']) if group['people'] else '',
            average_in_neighbors=float(np.mean(group['degrees'])),
            edge_count=int(group['count'][0]), second_order_walk_count=int(group['count'][1]),
            third_order_walk_count=int(group['count'][2]),
            second_order_person_fraction=float(group['covered'][1]/total),
            third_order_person_fraction=float(group['covered'][2]/total),
            second_order_nonreturn_person_fraction=group['nonreturn2']/total,
            second_order_return_fraction=float(group['returns'][1]/max(group['count'][1], 1)),
            third_order_closed_walk_fraction=float(group['returns'][2]/max(group['count'][2], 1)),
            cutoff_target_occurrences=group['final_people'],
            cutoff_second_order_fraction=float(group['final_covered'][1]/max(group['final_people'], 1)),
            cutoff_third_order_fraction=float(group['final_covered'][2]/max(group['final_people'], 1)),
            adjacent_edge_change_fraction=group['changes']/group['comparisons'] if group['comparisons'] else '',
            adjacent_edge_jaccard=float(np.mean(group['jaccards'])) if group['jaccards'] else '',
            only_first_order_window_fraction=group['only_first']/group['windows'] if group['windows'] else '',
            manifest_sha256=sha256(args.manifest), future_gt_used=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows, {len(manifest["windows"])} windows, {time.perf_counter()-start:.1f}s')


if __name__ == '__main__':
    main()
