"""Stage5.2: sampler identity, official-model preservation and historical groups."""
import copy
import numpy as np
import pytest
import torch
from src.socialvae_baseline.evaluation import noise,batch_noise,FixedNormal,fixed_draws,evaluate
from src.socialvae_baseline.groups import history_features,GROUPS
from src.socialvae_causal.official import load_official
from src.socialvae_causal.data import raw_example,collate_examples,history_context
from src.data.preprocessing import Recording
from scripts.stage5p1.audit_raw_inputs import scenario,change,write_raw


def examples(tmp_path):
    path=tmp_path/'raw.txt';write_raw(path,scenario('multi_person'))
    rec=Recording.read(path,'synthetic')
    items=[raw_example(rec,0,i) for i in [0,1]]
    return rec,items


def test_target_noise_independent_of_order_partition_and_future(tmp_path):
    _,items=examples(tmp_path)
    eps=batch_noise(items,False,'cpu').reshape(12,20,2,32)
    reversed_eps=batch_noise(items[::-1],False,'cpu').reshape(12,20,2,32)
    assert torch.equal(eps,reversed_eps.flip(2))
    assert torch.equal(eps[:,:,0],batch_noise(items[:1],False,'cpu').reshape(12,20,1,32)[:,:,0])
    changed=copy.deepcopy(items[0]);changed['y']+=100
    assert torch.equal(noise(items[0])[0],noise(changed)[0])
    assert noise(items[0])[1]!=noise(items[1])[1]
    assert noise(items[0])[1]!=noise(items[0],True)[1]


def test_fixed_draw_same_conditional_normal_parameters_and_log_prob():
    loc=torch.randn(4,32,requires_grad=True);scale=torch.full_like(loc,.7);eps=torch.randn_like(loc)
    normal=torch.distributions.Normal(loc,scale);fixed=FixedNormal(normal,eps)
    assert torch.equal(normal.mean,fixed.mean) and torch.equal(normal.stddev,fixed.stddev)
    assert torch.equal(normal.log_prob(loc+.1),fixed.log_prob(loc+.1))
    assert not fixed.sample().requires_grad
    fixed.rsample().sum().backward();assert torch.equal(loc.grad,torch.ones_like(loc))


def test_official_forward_batch_permutation_and_partition_sampling(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    net=load_official().SocialVAE(12,2,32).eval()
    _,items=examples(tmp_path);state={k:v.clone() for k,v in net.state_dict().items()}
    module_names=list(dict(net.named_modules()))
    def predict(items):
        x,y,n=collate_examples(items)
        with torch.no_grad(),fixed_draws(net.p_z,batch_noise(items,False,'cpu')):return net(x,n,20)
    a=predict(items);b=predict(items[::-1]).flip(2);single=predict(items[:1])
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(a[:,:,:1],single,rtol=1e-5,atol=1e-5)
    assert list(dict(net.named_modules()))==module_names
    assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
    assert not net.p_z._forward_hooks and not net.q_z._forward_hooks
    assert type(net.p_z(torch.zeros(1,32))) is torch.distributions.Normal


def test_hook_cleanup_on_failure(tmp_path):
    net=load_official().SocialVAE(12,2,32)
    with pytest.raises(RuntimeError):
        with fixed_draws(net.p_z,torch.zeros(12,1,32)):raise RuntimeError('test')
    assert not net.p_z._forward_hooks


def test_evaluation_does_not_consume_training_rng_or_change_model(tmp_path):
    _,items=examples(tmp_path);net=load_official().SocialVAE(12,2,32).train()
    torch.manual_seed(99);rng=torch.get_rng_state().clone()
    summary,rows=evaluate(net,items,1,'cpu')
    assert torch.equal(torch.get_rng_state(),rng) and net.training
    assert all(np.isfinite(v) for v in summary.values())
    assert len(rows)==2 and not net.p_z._forward_hooks and not net.q_z._forward_hooks


@pytest.mark.parametrize('variant',['target_future','neighbor_future','future_only_person','extra_prewindow'])
def test_group_features_never_use_future(tmp_path,variant):
    rows=scenario('multi_person');a=tmp_path/'a.txt';b=tmp_path/'b.txt'
    write_raw(a,rows);write_raw(b,change(rows,variant))
    ca=history_context(Recording.read(a,'synthetic'),0);cb=history_context(Recording.read(b,'synthetic'),0)
    fa=history_features(ca,int(np.where(ca.ids==0)[0][0]));fb=history_features(cb,int(np.where(cb.ids==0)[0][0]))
    assert fa==fb and set(fa['groups'].split('|'))<=set(GROUPS)
