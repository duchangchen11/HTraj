"""Topology/count oracle. Counts graph-rule walks, not observed causal events."""
import numpy as np


def count_ordered_walks(adjacencies, continuity, max_order=3):
    """Each distinct timed edge sequence is counted once at its final edge.

    State[target, origin] preserves origin only for return-walk diagnostics.
    Waiting carries counts but does not create another event/another order.
    """
    steps, people, _ = adjacencies.shape
    state = np.zeros((max_order, people, people), dtype=np.int64)
    new_by_agent = np.zeros((max_order, people), dtype=np.int64)
    return_by_agent = np.zeros_like(new_by_agent)
    for t in range(steps):
        old = state*continuity[t][None, :, None]
        new = np.zeros_like(state)
        target, source = np.nonzero(adjacencies[t])
        for q in range(max_order):
            if q == 0:
                np.add.at(new[q], (target, source), 1)
            else:
                np.add.at(new[q], target, old[q-1, source])
        new_by_agent += new.sum(-1)
        return_by_agent += np.diagonal(new, axis1=-2, axis2=-1)
        state = old+new
    return dict(new_by_agent=new_by_agent, return_by_agent=return_by_agent,
                final_counts=state.sum(-1), final_nonreturn=state.sum(-1)-np.diagonal(state, axis1=-2, axis2=-1))
