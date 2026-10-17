"""CPU-only real initialization and code/data provenance audit after all training."""
from common import *

def run():
    initial={k:torch.load(CACHE/'checkpoints'/('formal_'+k+'_initial.pt'),map_location='cpu',weights_only=True) for k in 'ABC'}
    base=initial['A'];result=[]
    for k in 'ABC':
        shared=[name for name in base if name in initial[k]]
        assert len(shared)==len(base) and all(torch.equal(base[name],initial[k][name]) for name in shared)
        extra={name:list(t.shape) for name,t in initial[k].items() if name not in base}
        count=sum(initial[k][name].numel() for name in extra)
        assert count==(0 if k=='A' else 8192)
        result.append(dict(model=k,shared_state_tensors=len(shared),shared_tensors_bitwise_equal=True,new_weight_shapes=extra,new_weight_count=count,initial_checkpoint_SHA=sha(CACHE/'checkpoints'/('formal_'+k+'_initial.pt'))))
    json_new(OUT/'initialization_and_capacity_audit.json',dict(passed=True,models=result,nominal_params=dict(A=3149192,B=3157384,C=3157384),BC_parameter_difference_fraction=0.,A_extra_fraction=8192/3149192,limits='Same trainable count not same function class. Gradient evidence is in structure_tests.json. No randomly initialized inactive dummy weights.'))
    files=[p for directory in ['scripts/stage9','configs/stage9'] for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    csv_new(OUT/'code_and_config_SHA256.csv',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)])
    json_new(OUT/'evaluation_provenance.json',dict(pre_evaluation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),model_code_SHA=sha(ROOT/'scripts/stage9/common.py'),executor_SHA=sha(ROOT/'scripts/stage9/execute.py'),analysis_SHA=sha(ROOT/'scripts/stage9/analyze.py'),training_config_SHA=sha(ROOT/'configs/stage9/training.json'),manifest_SHA=sha(OUT/'data_manifest.json'),official_source_commits=json.loads((OUT/'source_versions.json').read_text())['repositories'],author_weights_used_for_initialization=False,test_files_or_scores_used_for_model_selection=False))
    print('Actual initial shared tensors bitwise identical; B/C both extra8192; provenance saved')

if __name__=='__main__':run()
