"""Measured budget -> separately committed fixed budget -> real A/B/C training.
No author weights, no test file access, no adaptive validation selection.
"""
import argparse,copy,gzip,math,random,traceback
from common import *

def rows(split):
    with gzip.open(OUT/(split+'_targets.csv.gz'),'rt') as f:return list(csv.DictReader(f))

def packet(m,opt,epoch,kind):
    return dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=None,epoch=epoch,kind=kind,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_SHA=sha(ROOT/'configs/stage9/training.json'))

def restore(m,opt,p):
    state=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']]);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    return state

def check_loss_equivalence(m,inputs,gt,pred):
    # Invoke native l2 using a tiny accessor context; same all-valid mask/unit target weights.
    from qpid.training.loss.__layers import l2
    from qpid.constant import INPUT_TYPES
    class Access:
        coe=1.
        model=None
    class M:
        def get_label(self,labels,t):return labels[0]
        def get_input(self,x,t):return x[0] if t==INPUT_TYPES.OBSERVED_TRAJ else torch.ones(len(gt),device=gt.device)
    a=Access();a.model=M()
    l=l2.forward(a,[pred],[gt],inputs,mask=torch.ones(len(gt),device=gt.device),training=True)
    assert torch.equal(l,loss(pred,gt))

def timing():
    assert json.loads((OUT/'structure_tests.json').read_text())['passed']
    data=dataset('train');val=dataset('val');train_rows=rows('train');timings=[];summaries=[];elapsed_start=time.monotonic();cfg=json.loads((ROOT/'configs/stage9/design.json').read_text())
    index=np.random.default_rng(9001).permutation(len(data['hist']))
    for kind in 'ABC':
        m=model(kind,'cuda');seed(42);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);m.train();torch.cuda.reset_peak_memory_stats()
        before=time.monotonic()
        for s in range(50):
            torch.cuda.synchronize();t=time.monotonic();idx=torch.as_tensor(index[s*128:(s+1)*128],device='cuda');x,y=batch(data,idx)
            opt.zero_grad(set_to_none=True);pred=m.implement(x,training=True)[0];l=loss(pred,y)
            if s==0:check_loss_equivalence(m,x,y,pred)
            l.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf')))
            if not torch.isfinite(l) or not np.isfinite(gn):raise RuntimeError('nonfinite timing '+kind)
            opt.step();torch.cuda.synchronize();duration=time.monotonic()-t
            timings.append(dict(model=kind,step=s+1,seconds=duration,loss=float(l.detach()),gradient_norm=gn,batch_targets=len(idx),max_context=max(int(train_rows[int(i)]['context_people']) for i in index[s*128:(s+1)*128]),
                allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),whole_card=smi() if s%10==0 else ''))
        values=np.array([r['seconds'] for r in timings if r['model']==kind])
        # Restore exact model and Adam state without another optimizer step (50 update cap).
        p=CACHE/'checkpoints'/('timing_'+kind+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),p)
        mm=model(kind,'cuda');oo=torch.optim.Adam(mm.parameters(),lr=cfg['lr']);state=torch.load(p,weights_only=True);mm.load_state_dict(state['model'],strict=True);oo.load_state_dict(state['optimizer'])
        assert all(torch.equal(v,mm.state_dict()[k]) for k,v in m.state_dict().items())
        assert len(opt.state)==len(oo.state) and len(oo.state)>0
        for p1,p2 in zip(opt.param_groups[0]['params'],oo.param_groups[0]['params']):
            for key,value in opt.state.get(p1,{}).items():
                if isinstance(value,torch.Tensor):assert torch.equal(value,oo.state[p2][key])
        mm.eval();seed(8101);torch.cuda.synchronize();v0=time.monotonic()
        with torch.no_grad():vp=mm.implement([val['hist'][:128],val['nei'][:128]],training=False)[0]
        torch.cuda.synchronize();vtime=time.monotonic()-v0;assert torch.isfinite(vp).all()
        summaries.append(dict(model=kind,parameters=params(m),updates=50,P50=float(np.quantile(values,.5)),P90=float(np.quantile(values,.9)),P99=float(np.quantile(values,.99)),mean=float(values.mean()),max_context=57,
            validation_forward_seconds_128=vtime,validation_error_computed=False,restore_model_optimizer_exact=True,seconds=time.monotonic()-before,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
        del m,mm,opt,oo;torch.cuda.empty_cache()
    csv_new(OUT/'timing_steps.csv',timings);json_new(OUT/'timing.json',dict(models=summaries,seconds=time.monotonic()-elapsed_start,validation_accuracy_seen=False,weights_discarded_for_formal=True,environment=env()))
    # Conservative: use P90 including host sync/finite-gradient checks for every formal step.
    steps=math.ceil(len(data['hist'])/128);alternatives=[]
    for e in cfg['candidate_epochs_descending']:
        training=e*steps*sum(s['P90'] for s in summaries)
        validation=3*math.ceil(len(val['hist'])/128)*max(s['validation_forward_seconds_128'] for s in summaries)
        io=3*(math.ceil(e/20)*5+60)+600 # 5sec/checkpoint + per-run setup + extra10min I/O/analysis reserve
        cost=used()+(time.monotonic()-elapsed_start)+1.2*(training+validation+io)
        alternatives.append(dict(epochs=e,training_seconds=training,validation_seconds=validation,io_seconds=io,safety_factor=1.2,total_projected_seconds=cost,fits=cost<43200))
    chosen=next((r for r in alternatives if r['fits']),None)
    json_new(OUT/'budget_estimate.json',dict(alternatives=alternatives,chosen_epochs=chosen['epochs'] if chosen else None,steps_per_epoch=steps,total_targets_per_epoch=len(data['hist']),method='50 updates/model; P90 for all future steps +20% safety +explicit I/O/validation reserve; no validation error used'))
    if not chosen:raise RuntimeError('40 epochs x3 cannot fit budget; stop')
    frozen=dict(cfg,epochs=chosen['epochs'],steps_per_epoch=steps,data_manifest_SHA=sha(OUT/'data_manifest.json'),design_SHA=sha(ROOT/'configs/stage9/design.json'),timing_SHA=sha(OUT/'timing.json'),budget_SHA=sha(OUT/'budget_estimate.json'))
    json_new(ROOT/'configs/stage9/training.json',frozen)
    print(json.dumps(dict(chosen_epochs=chosen['epochs'],timings=summaries,budget=chosen),indent=2),flush=True)

def train(completed_prefix=None):
    config=ROOT/'configs/stage9/training.json';cfg=json.loads(config.read_text())
    # Must be tracked exactly in a successful commit; no unfrozen code/config changes.
    for p in [config,ROOT/'scripts/stage9/common.py',Path(__file__)]:
        rel=str(p.relative_to(ROOT));frozen=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest()==sha(p),('Uncommitted executable/config',rel)
    assert subprocess.check_output(['git','status','--porcelain','--',str(config),str(ROOT/'scripts/stage9')],cwd=ROOT,text=True).strip()==''
    stage_start=time.monotonic();already=used();data=dataset('train');n=len(data['hist']);summary=list(completed_prefix or []);checkpoints=[]
    for item in summary:
        with (OUT/(item['model']+'_training_curve.csv')).open() as f:
            for row in csv.DictReader(f):
                if row['checkpoint']:checkpoints.append(dict(model=item['model'],epoch=int(row['epoch']),path=row['checkpoint'],sha256=row['checkpoint_SHA'],final=int(row['epoch'])==cfg['epochs']))
    for kind in 'ABC':
        if any(item['model']==kind for item in summary):continue
        m=model(kind,'cuda');seed(42);opt=torch.optim.Adam(m.parameters(),lr=cfg['lr']);m.train()
        initial=CACHE/'checkpoints'/('formal_'+kind+'_initial.pt');torch.save(m.state_dict(),initial)
        curve=[];steps_total=0;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        stepfile=OUT/(kind+'_training_steps.csv.gz')
        with gzip.open(stepfile,'xt',newline='') as ff:
            writer=csv.DictWriter(ff,fieldnames=['epoch','step','loss','gradient_norm','seconds','targets'],lineterminator='\n');writer.writeheader()
            for epoch in range(1,cfg['epochs']+1):
                e0=time.monotonic();loss_sum=0.;norm_sum=0.;seen=0;steps=0
                permutation=np.random.default_rng(9000+epoch).permutation(n)
                for off in range(0,n,cfg['batch_size']):
                    if already+time.monotonic()-stage_start>43020:raise RuntimeError('Global budget stop; reserve180s before12h')
                    torch.cuda.synchronize();step0=time.monotonic()
                    idx=torch.as_tensor(permutation[off:off+cfg['batch_size']],device='cuda');x,y=batch(data,idx)
                    opt.zero_grad(set_to_none=True);pred=m.implement(x,training=True)[0];l=loss(pred,y);l.backward()
                    gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf')))
                    if not torch.isfinite(l) or not np.isfinite(gn):
                        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,offset=off),CACHE/'checkpoints'/('failure_'+kind+'.pt'))
                        raise RuntimeError('Nonfinite training '+kind)
                    opt.step();torch.cuda.synchronize();duration=time.monotonic()-step0;value=float(l.detach())
                    writer.writerow(dict(epoch=epoch,step=steps_total+1,loss=value,gradient_norm=gn,seconds=duration,targets=len(idx)))
                    loss_sum+=value*len(idx);norm_sum+=gn;seen+=len(idx);steps+=1;steps_total+=1
                assert seen==n and steps==cfg['steps_per_epoch']
                checkpoint='';checkpoint_sha=''
                if epoch%20==0 or epoch==cfg['epochs']:
                    path=CACHE/'checkpoints'/(kind+'_epoch'+str(epoch)+'.pt');torch.save(packet(m,opt,epoch,kind),path)
                    checkpoint=str(path.relative_to(ROOT));checkpoint_sha=sha(path)
                    checkpoints.append(dict(model=kind,epoch=epoch,path=checkpoint,sha256=checkpoint_sha,final=epoch==cfg['epochs']))
                row=dict(model=kind,epoch=epoch,loss=loss_sum/seen,gradient_norm_mean=norm_sum/steps,seconds=time.monotonic()-e0,optimizer_updates=steps,updates_total=steps_total,targets=seen,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),whole_card=smi(),checkpoint=checkpoint,checkpoint_SHA=checkpoint_sha,finite=True)
                curve.append(row);ff.flush()
                # Append-only journal for recoverable progress while long jobs run.
                with (OUT/(kind+'_epoch_journal.csv')).open('a',newline='') as journal:
                    jw=csv.DictWriter(journal,fieldnames=list(row),lineterminator="\n")
                    if epoch==1:jw.writeheader()
                    jw.writerow(row)
                print(json.dumps(row),flush=True)
        csv_new(OUT/(kind+'_training_curve.csv'),curve)
        # Restore all optimizer/RNG state from final checkpoint; no additional update.
        last=CACHE/'checkpoints'/(kind+'_epoch'+str(cfg['epochs'])+'.pt')
        restored=model(kind,'cuda');opt2=torch.optim.Adam(restored.parameters(),lr=cfg['lr']);state=restore(restored,opt2,last)
        assert state['epoch']==cfg['epochs'] and state['config_SHA']==sha(config)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())
        for p1,p2 in zip(opt.param_groups[0]['params'],opt2.param_groups[0]['params']):
            for k,v in opt.state.get(p1,{}).items():
                if isinstance(v,torch.Tensor):assert torch.equal(v,opt2.state[p2][k])
        summary.append(dict(model=kind,epochs=cfg['epochs'],updates=steps_total,target_visits=n*cfg['epochs'],seconds=time.monotonic()-start,parameters=params(m),initial_SHA=sha(initial),final_checkpoint=str(last.relative_to(ROOT)),final_SHA=sha(last),restored_model_optimizer_RNG=True,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
        json_new(OUT/(kind+'_training_complete.json'),summary[-1])
        del m,restored,opt,opt2;torch.cuda.empty_cache()
    csv_new(OUT/'checkpoint_index.csv',checkpoints);json_new(OUT/'training_summary.json',dict(models=summary,seconds=sum(r['seconds'] for r in summary),continuation_wall_seconds=time.monotonic()-stage_start,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_SHA=sha(config)))

def evaluate():
    cfg=json.loads((ROOT/'configs/stage9/training.json').read_text());completed=json.loads((OUT/'training_summary.json').read_text())
    assert len(completed['models'])==3 and len({r['epochs'] for r in completed['models']})==1
    data=dataset('val');manifest=rows('val');outputs=[];predictions=[]
    for kind in 'ABC':
        m=model(kind,'cuda');p=CACHE/'checkpoints'/(kind+'_epoch'+str(cfg['epochs'])+'.pt')
        state=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(state['model'],strict=True);m.eval()
        before=time.monotonic();torch.cuda.reset_peak_memory_stats();records=[];allpred=[]
        for start in range(0,len(manifest),128):
            selected=manifest[start:start+128];ids='|'.join(f"{r['sequence']}:{r['frame']}:{r['person_id']}" for r in selected)
            sample_seed=int(hashlib.sha256(ids.encode()).hexdigest()[:8],16)%(2**31)
            seed(sample_seed);idx=torch.arange(start,start+len(selected),device='cuda');x,gt=batch(data,idx)
            with torch.no_grad():pred=m.implement(x,training=False)[0]
            if not torch.isfinite(pred).all():raise RuntimeError('nonfinite validation '+kind)
            allpred.append(pred.cpu().numpy());dist=torch.linalg.vector_norm(pred-gt[:,None],dim=-1).cpu().numpy()
            for j,row in enumerate(selected):
                ade=dist[j].mean(-1);fde=dist[j,:,-1];record=dict(row,model=kind,ADE=float(ade.min()),FDE=float(fde.min()),ADE_head=int(ade.argmin()),FDE_head=int(fde.argmin()),sample_seed=sample_seed)
                records.append(record)
                for head in range(20):outputs.append(dict(model=kind,sequence=row['sequence'],frame=row['frame'],person_id=row['person_id'],head=head,ADE=float(ade[head]),FDE=float(fde[head])))
        torch.cuda.synchronize();seconds=time.monotonic()-before
        # Full arrays kept locally; version only SHA and light target/candidate metrics.
        path=CACHE/(kind+'_validation_predictions.npz');np.savez_compressed(path,predictions=np.concatenate(allpred),GT=data['gt'].cpu().numpy())
        csv_new(OUT/(kind+'_validation_targets.csv.gz'),records)
        predictions.append(dict(model=kind,path=str(path.relative_to(ROOT)),sha256=sha(path),targets=len(records),candidates=20,seconds=seconds,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),whole_card=smi(),checkpoint_SHA=sha(p),target_manifest_SHA=sha(OUT/'val_targets.csv.gz')))
        del m;torch.cuda.empty_cache()
    csv_new(OUT/'validation_candidate_errors.csv.gz',outputs);json_new(OUT/'validation_execution.json',dict(models=predictions,K=20,metrics='separate min of complete-trajectory ADE and endpoint FDE; never per-time head selection',FPC=False,checkpoint='common final epoch only',test_metrics_accessed=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['timing','train','evaluate']);a=parser.parse_args()
    torch.set_num_threads(1);start=time.monotonic()
    try:
        if a.mode=='timing':timing()
        elif a.mode=='train':train()
        else:evaluate()
    except Exception as e:
        json_new(OUT/(a.mode+'_failure_'+str(time.time_ns())+'.json'),dict(error=str(e),traceback=traceback.format_exc()));raise
    finally:gpu_ledger(a.mode,time.monotonic()-start)
