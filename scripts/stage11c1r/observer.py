"""Python call/return observation only; no tensor operations inside callback."""
import sys,time

class KMeansObserver:
    def __init__(self,km):
        self.km=km;self.pairwise=km.__globals__['pairwise_distance'];self.initialize=km.__globals__['initialize']
        self.calls=0;self.rows=[];self.final=None;self.input_device=None;self.initial_center_device=None
    def callback(self,frame,event,arg):
        code=frame.f_code
        if code is self.km.__code__:
            if event=='call':self.input_device=str(frame.f_locals['X'].device)
            elif event=='return':
                local=frame.f_locals
                # Retain the existing tensor reference. Its value is read only AFTER profiling exits.
                self.final=dict(iteration=local['iteration'],tol=local['tol'],iter_limit=local['iter_limit'],
                    shift_tensor=local['center_shift'],choice_device=str(local['choice_cluster'].device),
                    update_device=str(local['initial_state'].device),dtype=str(local['X'].dtype))
        elif code is self.pairwise.__code__:
            if event=='call':
                self.calls+=1
                if self.calls==1:
                    self.rows.append(dict(input1_device=str(frame.f_locals['data1'].device),
                        input2_device=str(frame.f_locals['data2'].device),
                        requested_device=str(frame.f_locals['device'])))
            elif event=='return' and self.calls==1:self.rows[0]['output_device']=str(arg.device)
        elif code is self.initialize.__code__ and event=='return':self.initial_center_device=str(arg.device)
    def __enter__(self):
        assert sys.getprofile() is None,'Do not overwrite another profiler'
        self.started=time.perf_counter();sys.setprofile(self.callback);return self
    def __exit__(self,*exc):sys.setprofile(None)
    def result(self):
        assert self.final is not None and self.calls==self.final['iteration']
        f=self.final;shift=float(f['shift_tensor'].item());converged=shift**2<f['tol']
        return dict(iterations=self.calls,final_center_shift=shift,final_center_shift_squared=shift**2,
            tol=f['tol'],iter_limit=f['iter_limit'],tol_converged=converged,
            reached_iteration_limit=self.calls>=f['iter_limit'],
            stop_reason='TOL' if converged else 'ITERATION_LIMIT',
            proposal_device=self.input_device,initial_center_device=self.initial_center_device,
            choice_cluster_device=f['choice_device'],center_update_device=f['update_device'],dtype=f['dtype'],
            **self.rows[0])
