"""Raw four-column counterfactual records; preserve every historical row."""
import numpy as np


def variants(raw, context, target):
    future = raw[:, 0] > context.start+70
    neighbors = [int(i) for i in context.ids if int(i) != target]
    counts = {i: int((future & (raw[:, 1] == i)).sum()) for i in neighbors}
    disappearing = next((i for i in neighbors if 0 < counts[i] < 12), None)
    neighbor = disappearing if disappearing is not None else max(neighbors, key=lambda i: counts[i])
    if counts[neighbor] == 0:
        raise ValueError('Counterfactual needs a real neighbor with at least one future record')
    changed = raw.copy()
    changed[future, 2:] += [31., -17.]
    yield 'future_positions', changed, dict(changed_future_rows=int(future.sum()), all_future_changed=True)
    drop = future & (raw[:, 1] == neighbor)
    yield 'neighbor_future_deleted', raw[~drop].copy(), dict(changed_future_rows=int(drop.sum()), neighbor_id=neighbor)
    times = sorted(raw[drop, 0])
    # Remove at least the final record. A one-record neighbor becomes absent.
    keep = max(0, min(3, len(times)-1))
    shorten = drop & ~np.isin(raw[:, 0], times[:keep])
    yield 'neighbor_future_shortened', raw[~shorten].copy(), dict(changed_future_rows=int(shorten.sum()), neighbor_id=neighbor, retained_future_records=keep)
    new_id = int(raw[:, 1].max())+10000
    extra = np.array([[context.start+80+10*t, new_id, 3.+t*.1, -2.] for t in range(12)])
    yield 'future_only_actor', np.concatenate([raw, extra]), dict(changed_future_rows=12, future_only_id=new_id)
    if disappearing is None:
        raise ValueError('Need a genuine unsupervised neighbor for the non-target-GT test')
    changed = raw.copy()
    affected = future & (raw[:, 1] == disappearing)
    changed[affected, 2:] += [-13., 27.]
    yield 'non_target_future_changed', changed, dict(changed_future_rows=int(affected.sum()),
        unsupervised_context_id=disappearing, original_future_records=counts[disappearing])
