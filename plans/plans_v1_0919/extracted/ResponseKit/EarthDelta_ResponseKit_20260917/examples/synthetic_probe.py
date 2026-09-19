"""Synthetic algebra only. This is NOT a weather experiment or a benchmark."""
import json
import torch
from earthdelta_response import central_response,ResponseGeometry,box_candidates,verify_candidates

def main():
    torch.set_num_threads(1)
    dtype=torch.float64
    # Two corrections have the same future effect: a useful redundancy test.
    r=torch.tensor([[1.,1.],[.2,.2]],dtype=dtype)
    def replay(a): return r@a
    a0=torch.zeros(2,dtype=dtype)
    truth=torch.tensor([1.,.2],dtype=dtype); w=torch.ones(2,dtype=dtype)
    probe=central_response(replay,a0)
    geometry=ResponseGeometry.from_error(probe.response,truth-probe.reference_output,w)
    candidates=box_candidates(geometry,bound=1,max_active=1,ridge=1e-5)
    checked=verify_candidates(replay,a0,candidates,truth,w,max_nonzero=2)
    print(json.dumps({'experiment_kind':'synthetic_algebra_NOT_weather',
        'probe_trajectories':probe.forward_calls,'candidate_supports':len(candidates),
        'verification_trajectories':checked.forward_calls,
        'gram':geometry.gram.tolist(),'chosen_offset':checked.offset.tolist(),
        'baseline_loss':checked.baseline_loss,'chosen_loss':checked.selected_loss},indent=2))

if __name__=='__main__': main()
