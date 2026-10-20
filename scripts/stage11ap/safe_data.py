"""Restricted, data-only pickle loader. Unknown globals fail closed."""
import pickle, importlib, io

ALLOWED={
 ('pandas.core.frame','DataFrame'),('pandas.core.internals.managers','BlockManager'),
 ('pandas._libs.internals','_unpickle_block'),('pandas.core.indexes.base','_new_Index'),
 ('pandas.core.indexes.base','Index'),('pandas.core.indexes.range','RangeIndex'),
 ('pandas.core.indexes.numeric','Int64Index'),('pandas.core.indexes.numeric','Float64Index'),
 ('pandas.core.internals.blocks','new_block'),('pandas.core.internals.blocks','make_block'),
 ('numpy.core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),
 ('numpy','ndarray'),('numpy','dtype'),('numpy.core.numeric','_frombuffer'),
 ('builtins','slice'),('builtins','set'),('builtins','frozenset'),
}
class RestrictedDataUnpickler(pickle.Unpickler):
    def __init__(self,f): super().__init__(f);self.globals_seen=[]
    def find_class(self,module,name):
        self.globals_seen.append((module,name))
        if (module,name) not in ALLOWED:
            raise pickle.UnpicklingError(f'Global blocked: {module}.{name}')
        if module=='pandas.core.indexes.numeric':
            # pandas1 Int64Index/Float64Index merged into Index in pandas2.
            import pandas as pd
            return pd.Index
        return getattr(importlib.import_module(module),name)
def load_data_pickle(path):
    with open(path,'rb') as f:
        loader=RestrictedDataUnpickler(f);data=loader.load()
    return data,loader.globals_seen
