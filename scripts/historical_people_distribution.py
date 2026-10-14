"""Full history-union/cutoff headcount histograms; no future labels read."""
from pathlib import Path
from collections import Counter
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import Recording, sha256

manifest_path = Path('configs/data_manifests/stage2_v1/observation_windows.json')
manifest = json.loads(manifest_path.read_text())
records = {name: Recording.read(Path('data/raw/eth_ucy')/info['file'], info['scene'])
           for name, info in manifest['source_files'].items()}
counts = Counter()
for entry in manifest['windows']:
    rec = records[entry['recording']]
    history = [rec.by_frame.get(entry['start_frame']+t*rec.frame_step, {}) for t in range(8)]
    counts[(entry['scene'], 'history_union', len(set().union(*(x.keys() for x in history))))] += 1
    counts[(entry['scene'], 'observation_cutoff', len(history[-1]))] += 1
with Path('results/tables/stage2_historical_people_distribution.csv').open('x', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['scene', 'definition', 'people', 'window_count', 'manifest_sha256'])
    for (scene, definition, people), count in sorted(counts.items()):
        writer.writerow([scene, definition, people, count, sha256(manifest_path)])
print('Wrote headcount histogram bins:', len(counts))
