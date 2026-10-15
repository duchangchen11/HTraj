"""Matched padded-batch training and common-noise evaluation, Stage4.2 only."""
import time
import numpy as np
import torch
from .data import pack
from .model import window_nll, gaussian_predictions

MODES = ('K1_distribution_mean', 'K1_first_sample', 'K20_independent_min')


def train_epoch(model, samples, optimizer, config, epoch):
    model.train()
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(config['seed'] + epoch)).tolist()
    loss_sum = 0.; gradient_norms = []; optimizer_steps = 0
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
    for offset in range(0, len(order), config['accumulation_windows']):
        chunk = order[offset:offset + config['accumulation_windows']]
        optimizer.zero_grad(set_to_none=True)
        for j in range(0, len(chunk), config['batch_windows']):
            selected = [samples[i] for i in chunk[j:j + config['batch_windows']]]
            batch = pack(selected, model.kind)
            parameters = model(batch['v'], batch['relation'], batch['valid'])
            losses = window_nll(parameters, batch['velocity'], batch['valid'])
            if not torch.isfinite(losses).all():
                raise FloatingPointError('Nonfinite training NLL; no automatic retry')
            (losses.sum() / len(chunk)).backward()
            loss_sum += float(losses.detach().sum())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
        gradient_norms.append(float(norm)); optimizer.step(); optimizer_steps += 1
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    if not np.isfinite(gradient_norms).all() or max(gradient_norms) == 0:
        raise FloatingPointError('Invalid training gradients')
    return dict(train_NLL_window_mean=loss_sum / len(samples), train_seconds=elapsed,
                gradient_norm_mean=float(np.mean(gradient_norms)), gradient_norm_p95=float(np.quantile(gradient_norms,.95)),
                gradient_norm_max=max(gradient_norms), optimizer_steps=optimizer_steps,
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                windows_per_second=len(samples)/elapsed,
                target_windows_per_second=sum(len(s['agent_ids']) for s in samples)/elapsed)


@torch.no_grad()
def evaluate(model, samples, batch_size=16, return_targets=False):
    model.eval(); sums={m:np.zeros(2,dtype=np.float64) for m in MODES}
    count=0; nll_sum=0.; nll_window_sum=0.; targets=[]
    states={key:[] for key in ('ego_norm','neighbor_norm','correction_norm','mixed_norm','relative_correction')}
    torch.cuda.synchronize(); start=time.perf_counter()
    for offset in range(0,len(samples),batch_size):
        selected=samples[offset:offset+batch_size];batch=pack(selected,model.kind,with_noise=True)
        parameters,state=model(batch['v'],batch['relation'],batch['valid'],return_state=True)
        losses=window_nll(parameters,batch['velocity'],batch['valid'])
        if not torch.isfinite(losses).all():raise FloatingPointError('Nonfinite evaluation NLL')
        sizes=batch['valid'].sum(-1);nll_sum+=float((losses*sizes).sum());nll_window_sum+=float(losses.sum());count+=int(sizes.sum())
        mean,draws=gaussian_predictions(parameters,batch['last'],batch['noise'])
        distance=(draws-batch['future'][:,None]).norm(dim=-1)
        ade=distance.mean(2);fde=distance[:,:,-1];mean_distance=(mean-batch['future']).norm(dim=-1)
        scores=dict(K1_distribution_mean=(mean_distance.mean(1),mean_distance[:,-1]),
            K1_first_sample=(ade[:,0],fde[:,0]),K20_independent_min=(ade.min(1).values,fde.min(1).values))
        for key,value in state.items():states[key].extend(value.masked_select(batch['valid'][:,None].expand_as(value)).cpu().tolist())
        for mode,(aa,ff) in scores.items():
            if not torch.isfinite(aa[batch['valid']]).all() or not torch.isfinite(ff[batch['valid']]).all():raise FloatingPointError('Nonfinite score')
            sums[mode]+=np.array([float(aa[batch['valid']].double().sum()),float(ff[batch['valid']].double().sum())])
            if return_targets:
                av,fv=aa.cpu().tolist(),ff.cpu().tolist()
                for k,s in enumerate(selected):
                    for i,pid in enumerate(s['agent_ids']):
                        feature=s['features'][i]
                        row=dict(recording=s['recording'],window_id=s['window_id'],agent_id=pid,mode=mode,
                                 ADE_m=av[k][i],FDE_m=fv[k][i],**{a:b for a,b in feature.items() if a!='groups'},groups='|'.join(feature['groups']))
                        for key,value in state.items():row[key+'_mean']=float(value[k,:,i].mean())
                        targets.append(row)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    metrics=dict(NLL=nll_sum/count,NLL_window_mean=nll_window_sum/len(samples),target_windows=count,windows=len(samples),evaluation_seconds=elapsed)
    for mode,total in sums.items():metrics.update({mode+'_ADE':float(total[0]/count),mode+'_FDE':float(total[1]/count)})
    for key,values in states.items():
        metrics[key+'_mean']=float(np.mean(values));metrics[key+'_p95']=float(np.quantile(values,.95))
    return metrics,targets
