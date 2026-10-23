"""CPU-only source audit and target/config freeze before diagnostic execution."""
from .common import *
import ast,math

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    original=ROOT/'experiments/stage11b0/cache/official/ynet/utils/kmeans.py'
    text=original.read_text();tree=ast.parse(text)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    pairwise=functions['pairwise_distance'];kmeans=functions['kmeans']
    assert pairwise.args.args[-1].arg=='device'
    assert ast.unparse(pairwise.args.defaults[-1])=="torch.device('cpu')"
    loop=next(n for n in ast.walk(kmeans) if isinstance(n,ast.While))
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='pairwise_distance_function']
    assert len(calls)==1 and len(calls[0].args)==2 and not calls[0].keywords
    expected=json.loads((ROOT/'results/tables/stage11ap/frozen_source_manifest.json').read_text())
    matching=[r for r in expected['files'] if r['path'].endswith('ynet/utils/kmeans.py')]
    assert len(matching)==1 and sha(original)==matching[0]['sha256']
    jnew(OUT/'source_audit.json',dict(passed=True,official_commit=OFFICIAL,license='Author Y-Net MIT, inherited frozen source',
        original_path=str(original.relative_to(ROOT)),original_SHA=sha(original),
        sampler_SHA=sha(ROOT/'scripts/stage11b1r/sampler.py'),
        evidence=[dict(lines='43-44,73-75,145-158',fact='Euclidean function called with two args; defaults device=CPU, both input tensors .to(cpu), distance and argmin CPU'),
            dict(lines='50-58,77-89',fact='X and initial_state remain requestedCUDA, cloning, selected.mean and center update CUDA'),
            dict(lines='79-84',fact='CPU choice mask indexes CUDA X per cluster; actual implicit payload is backend-dependent and not measured here'),
            dict(lines='16-17,81-82',fact='np.random.choice without replacement for19initial centers; empty cluster uses torch.randint onCPU'),
            dict(lines='92,102-107',fact='One pairwise call per main iteration with default list initialization; center_shift squared<.001 tested before iteration>=1000')],
        float_dtype='float32; X.float() preserves original',
        randomness='Original per-sample SHA CPU Generator; scoped original NumPy/Torch KMeans seed; no Python hash',
        monitor='sys.setprofile kmeans/pairwise/initialize call and return only; read tensors only after profiling exits',
        no_tensor_operations_inside_monitor=True,mutated_author_source=False,GT_used=False))
    manifests={s:ROOT/f'results/tables/stage11b1c/{s}_target_manifest.csv' for s in ['train','validation']}
    val=list(csv.DictReader(manifests['validation'].open()));samples=[]
    for scene in sorted({r['scene'] for r in val}):
        group=sorted([r for r in val if r['scene']==scene],key=lambda r:r['sample_id'])
        for q in [.2,.5,.8]:
            index=math.floor(q*(len(group)-1));r=group[index]
            samples.append(dict(sample_id=r['sample_id'],scene=scene,history_SHA=r['history_SHA'],reference_SHA=r['reference_SHA'],
                quantile=q,zero_based_index=index,scene_targets=len(group),rule='lexicographic sample_id;floor(q*(N-1));q=.2,.5,.8'))
    assert len(samples)==15 and len({r['sample_id'] for r in samples})==15
    cnew(OUT/'diagnostic_sample_manifest.csv',samples)
    sem=json.loads((ROOT/'configs/stage11c0/generation_config.json').read_text())['semantics']['S0']
    old=json.loads((ROOT/'results/tables/stage11b1c/checkpoint_manifest.json').read_text())['checkpoints']
    states=[]
    for epoch in [1,5,10,30]:
        row=next(r for r in old if r['epoch']==epoch);assert sha(ROOT/row['path'])==row['sha256']
        states.append(dict(name=f'Semantic-epoch{epoch:03d}',role='Semantic',checkpoint=row))
    config=dict(stage='Stage11C-1R',nature='RESOURCE_PROFILING_ONLY',official_commit=OFFICIAL,base=BASE,
        formal_training_authorized=False,diagnostic_epochs=1,history=8,future=12,resize=.25,seed=17,
        optimizer='Adam',learning_rate=.0001,batch=4,targets=7496,optimizer_updates=1882,
        loss='Original GoalBCE*1000+PathBCE*1000, teacher forcing train only',
        AMP=False,TF32=False,augmentation=False,scheduler=None,gradient_clipping=None,
        semantics=sem,semantic_states=states,phases=['diagnostic_train','profile'],
        profiling_states=['NoSemantic-initial','NoSemantic-epoch001']+[r['name'] for r in states],
        passes=['direct','observed','direct_repeat','observed_reverse'],targets_per_state=15,
        raw_KMeans_extra='One unmonitored original call on identical proposals per state/target; compare centers bitwise',
        CPU_diagnostic='Original kmeans CPU resident, unmonitored timing plus separate observed repeat, never used for paths',
        K=20,proposals=10000,centers=19,tol=.001,iter_limit=1000,
        sampling_seeds='Unchanged Stage11B1R stable SHA scheme',
        scope='No validation GT or metric computation; history-only cache and preserved original semantics',
        state_admission_seconds={'NoSemantic':600,'Semantic':300},
        hard_GPU_seconds=2400,safety_GPU_seconds=2250,original_used_seconds=ORIGINAL_USED,combined_hard_seconds=18000,
        per_operation_exit_save_reserve=60,diagnostic_epoch_start_estimate=750,profile_start_estimate=1500,
        no_timing_based_sample_reselection=True,
        budget_scenarios=dict(A='Historical full30training+seven998validation observed costs, explicit startup/checkpoint/repeat/recovery reserve',
            B='Actual new epoch1 training projected30times, post-epoch1 fixed15 paired per-scene timings projected to998, seven times; add10% timing margin',
            C='Actual new epoch1 projected30times, slower of initial/postepoch1 per-scene P90 costs for all seven998 validations; add20% timing margin'),
        recommendation='Use scenarioC safe bound for sufficiency; scenarioB is unverified later-epoch extrapolation, not actual cost',
        sample_manifest_SHA=sha(OUT/'diagnostic_sample_manifest.csv'))
    jnew(CONF/'runtime_protocol.json',config)
    paths=[original,ROOT/'scripts/stage11b1r/sampler.py',ROOT/'scripts/stage11b1r/predictor.py',
        ROOT/'scripts/stage11b2c/adapter.py',ROOT/'scripts/stage11b0/training_adapter.py',ROOT/'scripts/stage11b0/runtime.py',
        ROOT/'scripts/stage11b1c/engine.py',ROOT/'scripts/stage11b1c/data.py']+list((ROOT/'scripts/stage11c1r').rglob('*.py'))
    inputs=list(manifests.values())+[ROOT/'experiments/stage11b2c/cache/validation_history_only.pt',
        ROOT/'configs/stage11c1/scene_shapes.json',ROOT/'configs/stage11c1/train_control.json',ROOT/'results/tables/stage11c1/gpu_budget.json',
        ROOT/'results/tables/stage11c1/preflight_K20.csv',ROOT/'results/tables/stage11c1/preflight_results.json',
        ROOT/'experiments/stage11b1/cache/train_only.pkl',CONF/'runtime_protocol.json',OUT/'diagnostic_sample_manifest.csv']
    inputs +=[ROOT/r['path'] for r in sem.values()]+[ROOT/r['checkpoint']['path'] for r in states]
    def rows(seq):return [dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(set(seq))]
    jnew(OUT/'protocol_frozen.json',dict(stage='Stage11C-1R',scope='RESOURCE_PROFILING_ONLY',base=BASE,
        source_files=rows(paths),input_files=rows(inputs),data_manifest_SHA={s:sha(p) for s,p in manifests.items()},
        initialization_SHA=INITIAL,original_used=ORIGINAL_USED,new_hard=2400,new_safe=2250,combined_hard=18000,
        test_loaded=False,GT_scoring_authorized=False,formal_training_authorized=False))
    print('SOURCE_AND15_TARGETS_FROZEN',flush=True)
if __name__=='__main__':main()
