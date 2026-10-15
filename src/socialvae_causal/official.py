"""Load pinned source bytes without copying or editing official networks."""
import importlib.util
import json
import sys
from pathlib import Path
from src.data.preprocessing import sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'experiments/stage5/cache/socialvae/eb23cbae562bac2c2dcf0d81529e6ab2ac280962'
EXPECTED = {
    'social_vae.py':'2582714db40220bc03946feeb132841655cd2941f4dff19602e40d0f46f2528f',
    'data.py':'9f50b036168bb7c7de9408049d293646a63cee9720e0535cfca6ea9a9a50440d',
}


def load_official(filename='social_vae.py'):
    assert filename in EXPECTED
    path = SOURCE/filename
    assert sha256(path) == EXPECTED[filename], 'Frozen official source changed'
    name = 'stage5p1_official_'+path.stem
    if name in sys.modules:return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
