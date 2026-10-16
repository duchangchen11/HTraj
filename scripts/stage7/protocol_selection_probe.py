"""Execute the original loader's ID filter without importing unrelated map deps."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import ast,json,types,numpy as np
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256

meta=json.load(open('results/tables/stage7/sources/AgentFormer.json'))
p=Path(meta['source_directory'])/'data/preprocessor.py';root=ast.parse(p.read_text())
cls=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name=='preprocess')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['get_valid_id','GetID']]
namespace={'np':np};exec(compile(ast.Module(body=methods,type_ignores=[]),str(p),'exec'),namespace)
obj=types.SimpleNamespace(min_past_frames=8,min_future_frames=12)
obj.GetID=types.MethodType(namespace['GetID'],obj)
select=types.MethodType(namespace['get_valid_id'],obj)
frame=np.zeros((2,18));frame[:,1]=[0,1]
h=[frame.copy() for _ in range(8)];a=[frame.copy() for _ in range(12)];b=[f.copy() for f in a];b[7]=b[7][:1]
base=select(h,a);removed=select(h,b)
assert base==[0.,1.] and removed==[0.]
json_new('results/tables/stage7/official_context_selection_test.json',dict(candidate='AgentFormer',source_path=str(p),source_sha256=sha256(p),
 original_method_executed_via_AST=True,history_equal=True,only_neighbor_future_presence_changed=True,
 original_context_IDs=base,changed_context_IDs=removed,strict_history_only_context_selection=False,
 interpretation='Offline complete20 context selection; forecast values isolate GT but context membership uses future existence. Do not claim Causal8x12 equivalence.',
 new_training_runs=0,ETH_evaluated=False))
print('Official context membership changes with future existence; recorded protocol boundary.')
