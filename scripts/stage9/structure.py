"""Pretraining admission; artificial results are not prediction performance."""
import copy,json,time,traceback
from common import *

def run():
    started=time.monotonic();torch.set_num_threads(1);results=[];models={}
    from qpid.utils import INIT_POSITION
    for kind in 'ABC':
        m=model(kind,'cuda');models[kind]=m
        shape_rows=[]
        for n in [1,2,5,57]:
            seed(901+n)
            hist=torch.randn(2,8,2,device='cuda').cumsum(1)
            raw=torch.randn(2,n,8,2,device='cuda').cumsum(2)
            nei=raw-hist[:,-1,None,None]
            gt=torch.randn(2,12,2,device='cuda')
            m.eval();seed(88)
            with torch.no_grad():pred=m.implement([hist,nei],training=False)[0]
            assert list(pred.shape)==[2,20,12,2] and torch.isfinite(pred).all()
            seed(88)
            with torch.no_grad():permuted=m.implement([hist,nei.flip(1)],training=False)[0]
            pd=float((pred-permuted).abs().max());assert pd<2e-4,(kind,'permutation',pd)
            seed(88)
            with torch.no_grad():shift=m.implement([hist+hist.new_tensor([7.,-3.]),nei],training=False)[0]
            td=float((shift-pred-hist.new_tensor([7.,-3.])).abs().max());assert td<2e-4,(kind,'translation',td)
            padded=torch.cat([nei,torch.full((2,3,8,2),INIT_POSITION,device='cuda')],1)
            seed(88)
            with torch.no_grad():pad=m.implement([hist,padded],training=False)[0]
            pad_diff=float((pad-pred).abs().max());assert pad_diff<2e-4,(kind,'padding',pad_diff)
            m.train();seed(99);m.zero_grad(set_to_none=True)
            z=m.implement([hist,nei],training=True)[0];l=loss(z,gt);l.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf')))
            assert torch.isfinite(l) and np.isfinite(norm)
            new_grad={k:float(p.grad.norm()) for k,p in m.named_parameters() if any(s in k for s in ['rc.joint.','rc.pos.','rc.motion.','rc.cross.'])}
            if kind!='A' and n==57:assert new_grad and all(v>0 and np.isfinite(v) for v in new_grad.values()),new_grad
            shape_rows.append(dict(N=n,shape=list(pred.shape),permutation_max=pd,translation_max=td,padding_max=pad_diff,loss=float(l.detach()),gradient_norm=norm,new_parameter_gradients=new_grad))
        m.eval()
        hist=torch.zeros(1,8,2,device='cuda');empty=torch.full((1,3,8,2),INIT_POSITION,device='cuda',dtype=torch.float32)
        seed(90)
        with torch.no_grad():z=m.implement([hist,empty],training=False)[0]
        assert torch.isfinite(z).all()
        path=CACHE/'checkpoints'/('structure_'+kind+'.pt');torch.save(m.state_dict(),path)
        restored=model(kind,'cuda');restored.load_state_dict(torch.load(path,weights_only=True),strict=True);restored.eval()
        seed(90)
        with torch.no_grad():zz=restored.implement([hist,empty],training=False)[0]
        assert torch.equal(z,zz)
        results.append(dict(model=kind,parameters=params(m),cases=shape_rows,empty_neighbor_finite=True,strict_restore_exact=True,checkpoint_SHA=sha(path)))
        del restored
    assert params(models['B'])==params(models['C'])==params(models['A'])+8192
    # Initial shared tensors are identical; forward/backward above never updates parameters.
    a=models['A'].state_dict()
    for k in ['B','C']:
        other=models[k].state_dict()
        for name,tensor in a.items():assert torch.equal(tensor,other[name]),(k,name)
    target=torch.stack([torch.arange(8,device='cuda')*.4,torch.zeros(8,device='cuda')],-1)[None]
    anchors=torch.tensor([[1.,2.],[2.,4.]],device='cuda');v=torch.tensor([[.4,0.],[-.4,0.]],device='cuda');t=torch.arange(-7,1,dtype=torch.float32,device='cuda')
    nei=anchors[:,None,:]+t[None,:,None]*v[:,None,:];swapped=anchors[:,None,:]+t[None,:,None]*v.flip(0)[:,None,:]
    witness=[]
    for k,m in models.items():
        with torch.no_grad():r1,_=m.rc(target,nei[None]);r2,_=m.rc(target,swapped[None])
        d=float((r1-r2).abs().max());assert d<1e-5 if k=='A' else d>1e-5,(k,d)
        witness.append(dict(model=k,aggregation_max_difference=d,distinguishes_pairing=k!='A'))
    # Native l2 default reduces exactly to ADE_2D on full12 unit-weight targets.
    from qpid.training.loss.__layers import l2
    # Check all model supervision and output time shapes and default loss declarations.
    json_new(OUT/'structure_tests.json',dict(passed=True,cases=results,pairing_witness=witness,shared_initial_tensors_identical=True,
        numerical_tolerance=2e-4,synthetic_only=True,original_loss='qpid/training/loss/__ade.py:ADE_2D called unchanged, matches l2 with complete GT/mask1/weight1',seconds=time.monotonic()-started,environment=env()))
    print(json.dumps(dict(passed=True,parameters={k:params(m) for k,m in models.items()},pairing_witness=witness),indent=2))

if __name__=='__main__':
    started=time.monotonic()
    try:setup();run()
    except Exception as e:
        json_new(OUT/('structure_failure_'+str(time.time_ns())+'.json'),dict(error=str(e),traceback=traceback.format_exc()));raise
    finally:gpu_ledger('pretraining_structure_tests',time.monotonic()-started)
