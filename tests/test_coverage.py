import numpy as np
from src.evaluation.interaction_coverage import count_ordered_walks


def test_walk_oracle_waiting_not_double_counted_and_return_diagnostic():
    adjacency = np.zeros((8, 3, 3), bool)
    adjacency[1, 1, 0] = True
    adjacency[3, 2, 1] = True
    continuity = np.ones((8, 3), bool)
    continuity[0] = False
    result = count_ordered_walks(adjacency, continuity)
    assert result['new_by_agent'].sum(1).tolist() == [2, 1, 0]
    assert result['final_counts'][1, 2] == 1
    reverse = adjacency.copy()
    reverse[1], reverse[3] = adjacency[3], adjacency[1]
    assert count_ordered_walks(reverse, continuity)['new_by_agent'][1].sum() == 0
    adjacency[3] = False
    adjacency[3, 0, 1] = True
    result = count_ordered_walks(adjacency, continuity)
    assert result['return_by_agent'][1, 0] == 1
    assert result['new_by_agent'][1, 0] == 1


def test_oracle_resets_missing_intermediary():
    adjacency = np.zeros((8, 3, 3), bool)
    adjacency[1, 1, 0] = True
    adjacency[4, 2, 1] = True
    continuity = np.ones((8, 3), bool)
    continuity[0] = False
    continuity[3, 1] = False
    result = count_ordered_walks(adjacency, continuity)
    assert result['new_by_agent'][1].sum() == 0
