"""Pure audit contracts. No forecasting network, learned module, or new loss."""
from pathlib import Path
import numpy as np


def require_validation_file(path):
    path=Path(path)
    if not path.name.endswith('_val.txt'):
        raise ValueError('Only official validation source files are allowed in this probe')
    return path


def source_motion_geometry_witness():
    """Finite, symbolic-style example of the information erased by separate means."""
    position=np.array([1.,3.])
    motion=np.array([1.,-1.])
    unbound=np.array([position.mean(),motion.mean()])
    swapped_unbound=np.array([position.mean(),motion[::-1].mean()])
    return dict(unbound=unbound,swapped_unbound=swapped_unbound,
                joint_moment=float(np.mean(position*motion)),
                swapped_joint_moment=float(np.mean(position*motion[::-1])))


def validate_research_gates(gates):
    required={'problem','mechanism_difference','legal_data','fair_controls','hardware_cost','stop_rule'}
    if set(gates)!=required:
        raise ValueError('Incomplete decision gates')
    return all(value is True for value in gates.values())
