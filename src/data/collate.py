import torch


def collate_trajectories(samples):
    if not samples:
        raise ValueError('Empty batch')
    batch, history, horizon = len(samples), samples[0]['observed'].shape[0], samples[0]['future'].shape[0]
    people = max(s['observed'].shape[1] for s in samples)
    observed = torch.zeros(batch, history, people, 2)
    observed_valid = torch.zeros(batch, history, people, dtype=torch.bool)
    future = torch.zeros(batch, horizon, people, 2)
    future_valid = torch.zeros(batch, horizon, people, dtype=torch.bool)
    agent_ids = torch.full((batch, people), -1, dtype=torch.long)
    for b, sample in enumerate(samples):
        n = len(sample['agent_ids'])
        observed[b, :, :n], observed_valid[b, :, :n] = sample['observed'], sample['observed_valid']
        future[b, :, :n], future_valid[b, :, :n] = sample['future'], sample['future_valid']
        agent_ids[b, :n] = sample['agent_ids']
    return dict(observed=observed, observed_valid=observed_valid, future=future, future_valid=future_valid,
                agent_ids=agent_ids, timestamps=torch.stack([s['timestamps'] for s in samples]),
                future_timestamps=torch.stack([s['future_timestamps'] for s in samples]),
                frame_ids=torch.stack([s['frame_ids'] for s in samples]),
                scene=[s['scene'] for s in samples], recording=[s['recording'] for s in samples],
                sample_id=[s['sample_id'] for s in samples])
