"""Small audited helpers; upstream models/loaders remain byte-identical."""
import argparse
import ast
import hashlib
import importlib.util
import io
import pickle
from pathlib import Path
import numpy as np

ALLOWED_FOLDS=('hotel','univ','zara1','zara2')


def non_eth_fold(fold):
    if fold not in ALLOWED_FOLDS:raise ValueError('Only preregistered non-ETH folds allowed')
    return fold


class ArgsUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('argparse','Namespace'):return argparse.Namespace
        raise pickle.UnpicklingError('Unexpected pickle class')


def load_args(path):return ArgsUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def only_function(path,name,namespace):
    source=Path(path).read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    code=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(code)
    exec(compile(code,str(path),'exec'),namespace)
    return namespace[name],hashlib.sha256(ast.get_source_segment(source,fn).encode()).hexdigest()


def trajectory_errors(predictions,target):
    """[K,T,N,2] and [T,N,2]; per-person ADE/FDE, independent K minima."""
    predictions=np.asarray(predictions);target=np.asarray(target)
    if predictions.ndim!=4 or predictions.shape[1:]!=target.shape or target.shape[-1]!=2:raise ValueError('Trajectory shape')
    error=np.linalg.norm(predictions-target[None],axis=-1)
    return error.mean(1),error[:,-1]


def sample_modes(predictions,target,mean_prediction=None):
    ade,fde=trajectory_errors(predictions,target)
    result={'K1_first_sample':(ade[0],fde[0]),'K20_independent_min':(ade.min(0),fde.min(0))}
    if mean_prediction is not None:
        a,f=trajectory_errors(np.asarray(mean_prediction)[None],target);result['K1_distribution_mean']=(a[0],f[0])
    return result


def state_dict_audit(model,state):
    """Report incompatibility without loading partial/random model weights."""
    required=model.state_dict()
    return dict(required_keys=len(required),checkpoint_keys=len(state),
        missing_keys=sorted(set(required)-set(state)),unexpected_keys=sorted(set(state)-set(required)),
        shape_mismatches=[k for k in sorted(set(required)&set(state)) if required[k].shape!=state[k].shape])
