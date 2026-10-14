import pytest
import torch


@pytest.fixture(autouse=True)
def fixed_cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(17)
