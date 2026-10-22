"""GT-free GPU entry: original-goal regression, then one full alternative decode."""
from .common import *
from .decoder import decode
from scripts.stage11b0.runtime import build_model,state_digest
from scripts.stage11b0.training_adapter import make_templates
from scripts.stage11b1.common import configure_torch
import torch
import numpy as np
import traceback

def verified_load(record):
    path=ROOT/record['path'];assert sha(path)==record.get('sha256',record.get('file_SHA'))
    return torch.load(path,map_location='cpu',weights_only=True)

@torch.no_grad()
def main():
    mode=sys.argv[1];assert mode in {'regression','decode'}
    log=dict(passed=False,mode=mode,training_steps=0,GT_loaded=False,test_loaded=False,generated=0)
    started=float(os.environ['HTRAJ_GPU_PROCESS_START']);budget=Budget(parent_seconds(),started=started)
    try:
        frozen=json.loads((OUT/'protocol_frozen.json').read_text())
        for r in frozen['executable_files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
        assert sha(CONF/'generation_config.json')==frozen['generation_config_SHA']
        cfgdata=json.loads((CONF/'generation_config.json').read_text())
        assert set(cfgdata)=={'checkpoint_path','checkpoint_SHA','original_state_SHA','history','semantics','regression_reference','regression_ids','experiment_protocol_SHA','budget','input_schema','GT_fields'}
        assert not cfgdata['GT_fields'] and sha(CP)==CK
        admission=json.loads((OUT/'GPU_admission.json').read_text());assert admission['passed']
        assert sha(OUT/'engineering_tests.xml')==admission['engineering_test_XML_SHA']
        if mode=='decode':
            assert json.loads((OUT/'decoder_regression.json').read_text())['passed']
            assert json.loads((OUT/'process_regression.json').read_text())['returncode']==0
        configure_torch();loadstart=time.perf_counter();budget.guard(60)
        net,cfg=build_model(17,'cuda:0');saved=torch.load(CP,map_location='cpu',weights_only=True)
        assert saved['official_commit']==OFFICIAL
        net.load_state_dict(saved['model'],strict=True);net.eval();net.semantic_segmentation.eval()
        initial=state_digest(net.state_dict());assert initial==cfgdata['original_state_SHA'];del saved
        templates=make_templates(cfg,'cuda:0');torch.cuda.synchronize()
        load_seconds=time.perf_counter()-loadstart
        hp=verified_load(cfgdata['history']);assert set(hp)=={'sample_ids','scenes','history'}
        index={sid:i for i,sid in enumerate(hp['sample_ids'])};assert len(index)==998
        semcache={};semread=0.;semtransfer=0.
        def semantic(scene):
            nonlocal semread,semtransfer
            if scene not in semcache:
                t=time.perf_counter();record=cfgdata['semantics'][scene];sem=verified_load(record)
                assert digest(sem)==record['tensor_SHA'];semread+=time.perf_counter()-t
                t=time.perf_counter();semcache[scene]=sem.cuda();torch.cuda.synchronize();semtransfer+=time.perf_counter()-t
            return semcache[scene]
        rows=[]
        if mode=='regression':
            ref=verified_load(cfgdata['regression_reference'])
            assert set(ref)=={'sample_ids','goals','paths','checkpoint_SHA'} and ref['checkpoint_SHA']==CK
            for rep in range(2):
                for j in (range(3) if rep==0 else reversed(range(3))):
                    budget.guard(30);sid=ref['sample_ids'][j];i=index[sid]
                    path,timing=decode(net,cfg,templates,hp['history'][i:i+1],semantic(hp['scenes'][i]),ref['goals'][:,j])
                    expected=ref['paths'][:,j:j+1]
                    diff=float((path.double()-expected.double()).abs().max())
                    rows.append(dict(sample_id=sid,repeat=rep,all20_bitwise=torch.equal(path,expected),head0_bitwise=torch.equal(path[0],expected[0]),max_coordinate_difference=diff,path_SHA=digest(path),**timing))
                    assert torch.equal(path,expected),(sid,diff)
            log.update(passed=True,original_goal_all20paths_bitwise=True,three_real_samples=True,repeat_reverse_bitwise=True,
              model_load_seconds=load_seconds,semantic_cache_read_seconds=semread,semantic_H2D_seconds=semtransfer,
              fresh_segmentation_forward_seconds=0.,frozen_semantics_reused=True,regression_runs=6,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            csvnew(OUT/'decoder_regression_rows.csv',rows)
        else:
            selectinfo=json.loads((OUT/'selection_audit.json').read_text())
            alt=verified_load(dict(path=selectinfo['new_goals_path'],sha256=selectinfo['new_goals_file_SHA']))
            assert set(alt)=={'sample_ids','goals','checkpoint_SHA','metadata','partition_protocol_SHA','algorithm_config_SHA'}
            assert alt['goals'].shape==(20,998,2) and alt['checkpoint_SHA']==CK
            paths=[];save_seconds=0.;start=time.perf_counter()
            for j,sid in enumerate(alt['sample_ids']):
                budget.guard(30);i=index[sid]
                path,timing=decode(net,cfg,templates,hp['history'][i:i+1],semantic(hp['scenes'][i]),alt['goals'][:,j])
                paths.append(path[:,0]);rows.append(dict(sample_id=sid,scene=hp['scenes'][i],goal_SHA=digest(alt['goals'][:,j:j+1]),path_SHA=digest(path),**timing))
                log['generated']=j+1
                if (j+1)%100==0:
                    print('GPU_DECODE',j+1,'/998','budget',round(budget.elapsed(),2),flush=True)
                    p=CACHE/f'partial_paths_{j+1:04}.pt';assert not p.exists();t=time.perf_counter()
                    torch.save(dict(sample_ids=alt['sample_ids'][:j+1],paths=torch.stack(paths,dim=1),goals=alt['goals'][:,:j+1],checkpoint_SHA=CK),p)
                    save_seconds+=time.perf_counter()-t
            tensor=torch.stack(paths,dim=1);assert tensor.shape==(20,998,12,2) and torch.isfinite(tensor).all()
            target=CACHE/'alternative_paths.pt';assert not target.exists();budget.guard(15);t=time.perf_counter()
            torch.save(dict(sample_ids=alt['sample_ids'],paths=tensor,goals=alt['goals'],checkpoint_SHA=CK,
                history_file_SHA=cfgdata['history']['sha256'],goal_file_SHA=selectinfo['new_goals_file_SHA'],
                goal_SHA=digest(alt['goals']),path_SHA=digest(tensor),decoder_source_SHA=sha(Path(__file__).with_name('decoder.py'))),target)
            save_seconds+=time.perf_counter()-t
            csvnew(OUT/'decode_timing.csv',rows)
            log.update(passed=True,generated=998,shape=list(tensor.shape),alternative_paths_path=str(target.relative_to(ROOT)),
              alternative_paths_file_SHA=sha(target),path_SHA=digest(tensor),goal_SHA=digest(alt['goals']),
              model_load_seconds=load_seconds,semantic_cache_read_seconds=semread,semantic_H2D_seconds=semtransfer,
              fresh_segmentation_forward_seconds=0.,frozen_semantics_reused=True,cache_save_seconds=save_seconds,
              feature_seconds=sum(r['feature_seconds'] for r in rows),decode20_seconds=sum(r['decode20_seconds'] for r in rows),
              all_target_loop_seconds=time.perf_counter()-start,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert state_digest(net.state_dict())==initial
        log.update(parameters_unchanged=True,model_state_SHA=initial,official_commit=OFFICIAL,checkpoint_SHA=CK,source_commit=commit())
    except Exception as e:
        traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        log['GPU_cumulative_seconds_at_log']=budget.elapsed()
        jnew(OUT/('decoder_regression.json' if mode=='regression' else 'generation_execution.json'),log)
        print('GPU_WORKER_RESULT',mode,log['passed'],flush=True)

if __name__=='__main__':main()
