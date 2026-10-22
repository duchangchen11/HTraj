"""Small pre-score engineering check, no new experimental condition or GT input."""
from .common import *
import torch,traceback

@torch.no_grad()
def worker():
    from scripts.stage11b0.runtime import build_model,state_digest
    from scripts.stage11b0.training_adapter import make_templates
    from scripts.stage11b1.common import configure_torch
    from scripts.stage11b1.evaluation import history_record
    from scripts.stage11b2e.decoder import decode
    from .worker import input_guard
    log=dict(passed=False,GT_loaded=False,test_loaded=False,training_steps=0)
    budget=Budget(parent_seconds(),started=float(os.environ['HTRAJ_GPU_PROCESS_START']))
    try:
        plan=json.loads((CONF/'fixed_verification_plan.json').read_text());assert sha(Path(__file__))==plan['source_SHA']
        config=json.loads((CONF/'generation_config.json').read_text());hp=checked(config['history']);old=checked(config['original_predictions'])
        status=json.loads((OUT/'fixed_generation.json').read_text());assert status['passed'] and status['generated']==2994
        configure_torch();net,cfg=build_model(17,'cuda:0');saved=torch.load(CP,map_location='cpu',weights_only=True);net.load_state_dict(saved['model'],strict=True);net.eval()
        initial=state_digest(net.state_dict());assert initial==config['model_state_SHA'];del saved
        templates=make_templates(cfg,'cuda:0');rows=[]
        for group in status['groups']:
            pack=checked(group);condition=group['condition'][:2]
            for repeat in range(2):
                for i in (config['regression_indices'] if repeat==0 else list(reversed(config['regression_indices']))):
                    budget.guard(30);sid=hp['sample_ids'][i];h=hp['history'][i:i+1];sem=checked(config['semantics'][condition][hp['scenes'][i]]).cuda()
                    # Only history is projected; dummy future replacements cannot enter the model.
                    raw=dict(sample_id=sid,xy=torch.cat([h[0],torch.full((12,2),float(repeat)*1e6)]).numpy())
                    _,projected=history_record(raw);assert torch.equal(h,projected)
                    with input_guard(net,projected,sem,templates):path,timing=decode(net,cfg,templates,projected,sem,old['goals'][:,i])
                    expected=pack['paths'][:,i:i+1];assert torch.equal(path,expected)
                    rows.append(dict(condition=condition,sample_id=sid,repeat=repeat,all20paths_bitwise=True,max_coordinate_difference=0.,
                        future_dummy_value=float(repeat)*1e6,path_SHA=digest(path),**timing))
        assert state_digest(net.state_dict())==initial
        csvnew(OUT/'fixed_repeat_reverse_rows.csv',rows)
        log.update(passed=True,comparisons=len(rows),all20paths_repeat_reverse_bitwise=True,
            future_prefix_projection_unchanged=True,parameters_unchanged=True,model_state_SHA=initial,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as e:traceback.print_exc();log.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        log['GPU_cumulative_at_log']=budget.elapsed();jnew(OUT/'fixed_repeat_reverse.json',log)
        print('FIXED_REPEAT_REVERSE',log['passed'],flush=True)

def main():
    if '--worker' in sys.argv:return worker()
    prior=parent_seconds();assert prior<6600-60;start=time.perf_counter();env=os.environ.copy();env['HTRAJ_GPU_PROCESS_START']=str(start)
    with (OUT/'fixed_repeat_reverse_stdout.txt').open('x') as f:
        try:p=subprocess.run([sys.executable,'-u','-m','scripts.stage11c0.verify_fixed','--worker'],cwd=ROOT,env=env,
            stdout=f,stderr=subprocess.STDOUT,timeout=max(1,6600-prior-30));code=p.returncode
        except subprocess.TimeoutExpired:code=124
    elapsed=time.perf_counter()-start;jnew(OUT/'process_verification.json',dict(worker='fixed_repeat_reverse',returncode=code,wall_seconds=elapsed,
        prior_seconds=prior,cumulative_seconds=prior+elapsed,hard_seconds=7200,safety_seconds=6600,source_commit=commit(),scope='entire engineering child startup/load/repeat/reverse/save/exit'))
    assert prior+elapsed<7200
    print('VERIFICATION exit',code,'wall',elapsed,'cumulative',prior+elapsed,flush=True)
    if code:raise SystemExit(code)
if __name__=='__main__':main()
