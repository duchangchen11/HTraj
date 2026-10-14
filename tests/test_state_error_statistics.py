import numpy as np
from scripts.stage2p6_state_error_analysis import ranks,correlations


def test_average_tie_ranks_and_constant_correlation_are_explicit():
    np.testing.assert_array_equal(ranks([3,1,1,8]),[2,.5,.5,3])
    assert correlations(np.ones(4),np.arange(4)) == (None,None)
    pearson,spearman=correlations(np.arange(4),-np.arange(4))
    assert abs(pearson+1)<1e-12 and abs(spearman+1)<1e-12
