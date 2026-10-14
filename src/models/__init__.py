from .ped_base import PedBase
from .social_baselines import FirstOrderGNN, DynamicGNN
from .tohi_a import TOHIA

MODEL_REGISTRY = {
    'PedBase': PedBase,
    'FirstOrder-GNN': FirstOrderGNN,
    'Dynamic-GNN': DynamicGNN,
    'TOHI-Net-A': TOHIA,
}
