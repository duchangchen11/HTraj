"""Preregistered finite-sample distribution sanity checks; no GT/errors."""
from .common import *
from .sampler import sample_cpu,probability_weights
from .predictor import digest
from scripts.stage11b0.runtime import original_modules
from scripts.stage11b1.evaluation import sample_seed
import torch,numpy as np,math

def spatial_bin(indices,height,width):
    row=indices//width;col=indices%width
    return (row*4//height)*4+col*4//width

def main(guard):
    configure_torch();config=json.loads((CONF/'sampling_protocol.json').read_text())
    test=config['distribution_test'];assert test['trials']==50 and test['draws_per_trial']==10000
    minimal=json.loads((OUT/'minimal_original_case.json').read_text())
    saved=torch.load(ROOT/minimal['artifact_path'],map_location='cpu',weights_only=True)
    sparse=torch.zeros(1,1,16,20);sparse[:,:,:,::4]=1;sparse[0,0,0,1]=.01;sparse[0,0,0,2]=.009
    maps={'uniform16x20':torch.ones(1,1,16,20),'gradient16x20':torch.arange(1.,321.).reshape(1,1,16,20)/320.,
          'sparse16x20':sparse,'first_real_train_goal_probability':saved['probability']}
    assert list(maps)==test['maps']
    N=test['samples_per_map_device'];families=len(maps)*2*16
    epsilon=math.sqrt(math.log(2*families/test['familywise_alpha'])/(2*N))
    bins=[];trials=[];results=[]
    for name,prob in maps.items():
        guard();prob=prob.cuda();H,W=prob.shape[-2:]
        # Independent literal reconstruction of author's weight preparation.
        reference=prob.view(1,-1)
        threshold=reference.max(1)[0].unsqueeze(1).expand(-1,reference.size(1))
        reference=reference*(~(reference<threshold*.01)).int();reference=reference/reference.sum()
        actual=probability_weights(prob);assert torch.equal(actual,reference)
        weights_cpu=actual.cpu();support=weights_cpu.numpy().reshape(-1)>0
        mapping=spatial_bin(torch.arange(H*W),H,W).numpy()
        probabilities=reference.cpu().double().numpy().reshape(-1);probabilities/=probabilities.sum()
        expected=np.bincount(mapping,weights=probabilities,minlength=16)
        counts={'CPU':np.zeros(16,dtype=np.int64),'GPU':np.zeros(16,dtype=np.int64)}
        support_pass=True
        for i in range(test['trials']):
            guard(5);sid=f'distribution:{name}:trial:{i}';seed=sample_seed(sid)
            t=time.perf_counter();new,w,idx=sample_cpu(prob,sid);torch.cuda.synchronize();cpu_seconds=time.perf_counter()-t
            assert torch.equal(w,weights_cpu)
            with torch.random.fork_rng(devices=[0]):
                torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
                torch.cuda.set_rng_state(torch.Generator(device='cuda:0').manual_seed(seed).get_state())
                deterministic=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(False)
                t=time.perf_counter()
                try:old=original_modules().image.sampling(prob,10000,replacement=True,rel_threshold=.01)
                finally:torch.use_deterministic_algorithms(deterministic)
                torch.cuda.synchronize();gpu_seconds=time.perf_counter()-t
            oldindex=(old[...,1].long()*W+old[...,0].long()).reshape(-1).cpu().numpy()
            for device,index,seconds,proposal in [('CPU',idx.reshape(-1).numpy(),cpu_seconds,new),('GPU',oldindex,gpu_seconds,old)]:
                assert len(index)==10000 and index.min()>=0 and index.max()<H*W
                support_ok=bool(support[index].all());support_pass=support_pass and support_ok
                tally=np.bincount(mapping[index],minlength=16);counts[device]+=tally
                trials.append(dict(map=name,device=device,trial=i,seed=seed,draws=10000,proposal_SHA=digest(proposal),
                    weights_SHA=digest(w),seconds=seconds,support_pass=support_ok,bin_counts='|'.join(map(str,tally))))
        freq={d:c/N for d,c in counts.items()}
        cpu_error=float(np.max(np.abs(freq['CPU']-expected)));gpu_error=float(np.max(np.abs(freq['GPU']-expected)))
        between=float(np.max(np.abs(freq['CPU']-freq['GPU'])))
        passed=cpu_error<=epsilon and gpu_error<=epsilon and between<=2*epsilon and support_pass
        results.append(dict(map=name,CPU_max_abs_frequency_error=cpu_error,GPU_max_abs_frequency_error=gpu_error,
            CPU_GPU_max_abs_frequency_difference=between,epsilon=epsilon,CPU_GPU_tolerance=2*epsilon,
            threshold_weights_bitwise_equal=True,nonzero_support_pass=support_pass,passed=passed))
        for b in range(16):bins.append(dict(map=name,bin=b,expected_probability=float(expected[b]),
            CPU_count=int(counts['CPU'][b]),GPU_count=int(counts['GPU'][b]),samples_per_device=N,
            CPU_frequency=float(freq['CPU'][b]),GPU_frequency=float(freq['GPU'][b]),epsilon=epsilon))
        print(name,'CPU error',cpu_error,'GPU error',gpu_error,'bound',epsilon,'PASS',passed,flush=True)
    cnew(OUT/'distribution_trials.csv',trials);cnew(OUT/'distribution_bins.csv',bins)
    jnew(OUT/'distribution_results.json',dict(passed=all(r['passed'] for r in results),maps=results,
         formula=test['tolerance_formula'],epsilon=epsilon,familywise_alpha=test['familywise_alpha'],
         samples_per_map_device=N,total_draws=N*len(maps)*2,trials_per_map_device=50,
         configuration_SHA=sha(CONF/'sampling_protocol.json'),threshold_modified_after_observation=False,
         results='Finite Monte Carlo sanity checks, not equality of CPU/GPU sequences or bitwise equality between devices'))
    assert all(r['passed'] for r in results),'Preregistered distribution gate failed; do not alter threshold'

if __name__=='__main__':timed_run('distribution',main)
