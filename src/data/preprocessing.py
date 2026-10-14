"""Parse original four-column tracks. No old loader/cache dependency."""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

SCENE_FILES = {
    'ETH': ['biwi_eth.txt'], 'HOTEL': ['biwi_hotel.txt'],
    'UNIV': ['students001.txt', 'students003.txt'],
    'ZARA1': ['crowds_zara01.txt'], 'ZARA2': ['crowds_zara02.txt'],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


@dataclass
class Recording:
    name: str
    scene: str
    frames: np.ndarray
    by_frame: dict
    frame_step: int = 10
    dt: float = 0.4

    @classmethod
    def read(cls, path, scene, frame_step=10, dt=0.4):
        rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
        if rows.shape[1] != 4 or not np.isfinite(rows).all():
            raise ValueError('Expected finite frame_id pedestrian_id x y: ' + str(path))
        if not np.equal(rows[:, :2], np.round(rows[:, :2])).all():
            raise ValueError('Noninteger frame/pedestrian identifier')
        keys = rows[:, :2].astype(np.int64)
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError('Duplicate frame/pedestrian pair: ' + str(path))
        frames = np.unique(keys[:, 0])
        if len(frames) < 2 or frame_step <= 0 or dt <= 0:
            raise ValueError('Invalid time calibration')
        gaps = np.diff(frames)
        if np.any(gaps % frame_step) or np.min(gaps) != frame_step:
            raise ValueError('Frame step differs from explicitly calibrated step')
        by_frame = {}
        for (frame, person), xy in zip(keys, rows[:, 2:]):
            by_frame.setdefault(int(frame), {})[int(person)] = xy.astype(np.float32)
        return cls(Path(path).stem, scene, frames, by_frame, frame_step, dt)

    def audit(self, path):
        counts = np.asarray([len(self.by_frame[int(f)]) for f in self.frames])
        positions = np.stack([xy for frame in self.by_frame.values() for xy in frame.values()])
        speeds = []
        for frame in self.frames[1:]:
            current, previous = self.by_frame[int(frame)], self.by_frame.get(int(frame)-self.frame_step, {})
            for person in current.keys() & previous.keys():
                speeds.append(float(np.linalg.norm(current[person]-previous[person])/self.dt))
        return {
            'file': Path(path).name, 'recording': self.name, 'scene': self.scene,
            'sha256': sha256(path), 'bytes': Path(path).stat().st_size,
            'rows': int(counts.sum()), 'pedestrians': len({p for v in self.by_frame.values() for p in v}),
            'columns': ['frame_id', 'pedestrian_id', 'world_x', 'world_y'],
            'coordinate_unit': 'm', 'unit_basis': 'upstream distribution + publication; no new camera calibration',
            'coordinate_min': positions.min(0).tolist(), 'coordinate_max': positions.max(0).tolist(),
            'frame_min': int(self.frames[0]), 'frame_max': int(self.frames[-1]),
            'frame_gap_counts': {str(int(k)): v for k, v in Counter(np.diff(self.frames)).items()},
            'frame_step': self.frame_step, 'seconds_per_sample': self.dt,
            'seconds_per_frame_id': self.dt/self.frame_step,
            'time_basis': 'observed 10-frame grid + documented 2.5Hz convention, not recoverable from numbers alone',
            'speed_m_per_s_quantiles': np.quantile(speeds, [0.5, 0.95, 1.0]).tolist(),
            'duplicate_pairs': 0, 'finite_coordinates': True,
        }

    def windows(self, observation=8, prediction=12, stride=1):
        # Never join recordings or compress empty intervals into adjacent steps.
        segments = np.split(self.frames, np.where(np.diff(self.frames) != self.frame_step)[0]+1)
        length = observation + prediction
        for segment in segments:
            for index in range(0, len(segment)-length+1, stride):
                selected = segment[index:index+length]
                history_ids = set().union(*(self.by_frame[int(f)].keys() for f in selected[:observation]))
                # Eligibility depends only on history, never future existence.
                if not history_ids or not self.by_frame[int(selected[observation-1])]:
                    continue
                yield {'sample_id': f'{self.name}:{int(selected[0])}', 'recording': self.name,
                       'scene': self.scene, 'start_frame': int(selected[0]), 'end_frame': int(selected[-1])}

    def sample(self, start_frame, observation=8, prediction=12, include_supervision=True):
        frame_ids = start_frame + np.arange(observation+prediction)*self.frame_step
        history = [self.by_frame.get(int(f), {}) for f in frame_ids[:observation]]
        ids = sorted(set().union(*(x.keys() for x in history)))
        if not ids:
            raise ValueError('No historically observable pedestrians')
        index = {person: i for i, person in enumerate(ids)}
        pos = np.zeros((observation, len(ids), 2), np.float32)
        valid = np.zeros((observation, len(ids)), bool)
        future = np.zeros((prediction, len(ids), 2), np.float32)
        supervision = np.zeros((prediction, len(ids)), bool)
        for t, agents in enumerate(history):
            for person, xy in agents.items():
                pos[t, index[person]], valid[t, index[person]] = xy, True
        # Future-only pedestrians are excluded. Vanished-at-cutoff agents are
        # historical context, not forecast targets, even if they later reappear.
        if include_supervision:
            for t, frame in enumerate(frame_ids[observation:]):
                for person, xy in self.by_frame.get(int(frame), {}).items():
                    if person in index and valid[-1, index[person]]:
                        future[t, index[person]], supervision[t, index[person]] = xy, True
        return dict(observed=pos, observed_valid=valid, future=future, future_valid=supervision,
                    timestamps=(frame_ids[:observation]*self.dt/self.frame_step).astype(np.float64),
                    future_timestamps=(frame_ids[observation:]*self.dt/self.frame_step).astype(np.float64),
                    frame_ids=frame_ids, agent_ids=np.asarray(ids, np.int64), scene=self.scene,
                    recording=self.name, sample_id=f'{self.name}:{start_frame}')


def build_manifests(raw_root, output_root, validation_fraction=0.2, embargo_steps=20):
    raw_root, output_root = Path(raw_root), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError('Manifest output exists; choose a new output root')
    audits, recordings, source_files = [], {}, {}
    for scene, files in SCENE_FILES.items():
        for name in files:
            path = raw_root/name
            rec = Recording.read(path, scene)
            audits.append(rec.audit(path))
            recordings[rec.name] = rec
            source_files[rec.name] = dict(file=name, scene=scene, sha256=sha256(path))
    raw_audit = dict(schema_version=1, format='frame_id pedestrian_id x y',
                     files=audits, raw_directory=str(raw_root.resolve()), interpolation=False)
    write_json_new(output_root/'raw_audit.json', raw_audit)
    all_windows = [w for rec in recordings.values() for w in rec.windows()]
    write_json_new(output_root/'observation_windows.json', dict(
        schema_version=1, purpose='descriptive observation coverage only; not model selection',
        observation_frames=8, prediction_frames=12, source_files=source_files,
        windows=all_windows, window_stride=1))
    for test_scene in SCENE_FILES:
        splits = dict(train=[], validation=[], test=[])
        cutoffs = {}
        for rec in recordings.values():
            span = int((rec.frames[-1]-rec.frames[0])//rec.frame_step)+1
            cut = int(rec.frames[0]) + int(span*(1-validation_fraction))*rec.frame_step
            validation_start = cut + embargo_steps*rec.frame_step
            cutoffs[rec.name] = dict(train_end_exclusive=cut, validation_start_inclusive=validation_start)
            for window in rec.windows():
                if rec.scene == test_scene:
                    splits['test'].append(window)
                elif window['end_frame'] < cut:
                    splits['train'].append(window)
                elif window['start_frame'] >= validation_start:
                    splits['validation'].append(window)
        if any(not split for split in splits.values()):
            raise ValueError('Empty split: '+test_scene)
        write_json_new(output_root/f'fold_{test_scene}.json', dict(
            schema_version=1, held_out_scene=test_scene, observation_frames=8, prediction_frames=12,
            source_files=source_files, window_stride=1, seconds_per_sample=0.4, frame_step=10,
            validation_fraction=validation_fraction, embargo_steps=embargo_steps,
            cutoff_by_recording=cutoffs, splits=splits))
    return raw_audit
