from dataclasses import replace
import inspect
import torch
import pytest
from earthdelta_v5 import *
from earthdelta_v5.spectral import coefficient_diagnostics, assert_same_latitude_nodes

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(7)
    torch.set_num_threads(1)

def version():
    return ArtifactVersion('backbone','static','bank','norm','grid','projection','split','reference_after_hold')

def record(i='a',issue=0,valid=6,available=8,ver='v',event='old'):
    return VerifiedRecord(i,issue,valid,available,ver,event,torch.tensor([1.,0.]),torch.tensor([3.]))

def small_batch():
    plans=pilot_plans()
    model=PairedEditPredictor(6,4,9,6,16)
    history=torch.randn(3,4,3,6)
    memory=torch.randn(3,3,4)
    edits=torch.tensor([p.descriptor() for p in plans])
    leads=torch.tensor([6.,12.,24.])
    enabled=torch.tensor([any(p.groups) for p in plans])
    ref=torch.randn(3,3,3,6)
    edited=ref[:,None].repeat(1,len(plans),1,1,1)+torch.randn(3,len(plans),3,3,6)*0.1
    edited[:,0]=ref
    targets=make_training_targets(ref+torch.randn_like(ref),ref,edited)
    return model,history,memory,edits,leads,enabled,targets

def test_version_digest_stable():
    assert version().digest == version().digest

def test_version_digest_changes_for_policy():
    assert version().digest != replace(version(),continuation='other').digest

def test_version_mismatch_rejected():
    with pytest.raises(ValueError,match='grid'):
        version().assert_matches(replace(version(),grid='new-grid'))

def test_empty_version_rejected():
    with pytest.raises(ValueError): replace(version(),edit_bank='')

def test_plan_continuation_exact():
    p=pilot_plans()[1]
    assert any(p.active_at(3)) and not any(p.active_at(4))

def test_plan_false_mask_has_zero_coefficients():
    with pytest.raises(ValueError): EditPlan('bad',(False,),(1.,))

def test_plan_rank_alternatives_same_budget_different_direction():
    p=pilot_plans()
    assert sum(p[1].groups)==sum(p[2].groups)==1 and p[1].groups != p[2].groups

def test_paired_error_identity():
    ref=torch.randn(2,3,2,7);edit=ref[:,None]+torch.randn(2,4,3,2,7);y=torch.randn_like(ref)
    t=make_training_targets(y,ref,edit)
    torch.testing.assert_close(t.edited_error,y[:,None]-edit)

def test_quadratic_gain_identity_weighted():
    ref=torch.randn(2,3,2,7);edit=ref[:,None]+torch.randn(2,4,3,2,7);y=torch.randn_like(ref)
    w=torch.rand(7);t=make_training_targets(y,ref,edit,w)
    expected=(((y-ref)[:,None].square()-(y[:,None]-edit).square())*(w/w.sum())).sum(-1)
    torch.testing.assert_close(t.quadratic_gain,expected)

def test_no_edit_response_gain_zero():
    ref=torch.randn(2,3,2,7);t=make_training_targets(torch.randn_like(ref),ref,ref[:,None])
    assert t.edit_response.count_nonzero()==0 and t.quadratic_gain.count_nonzero()==0

def test_response_does_not_require_truth():
    assert 'truth' not in inspect.signature(edit_responses).parameters

def test_negative_weight_rejected():
    with pytest.raises(ValueError): quadratic_gain(torch.zeros(1,1,1,2),torch.zeros(1,1,1,1,2),torch.tensor([1.,-1.]))

def test_zero_initialized_editor_exact():
    layer=GroupedLowRankResidual(8,8,4,2);x=torch.randn(2,3,8)
    assert layer(x,torch.ones(2,4,dtype=torch.bool)).count_nonzero()==0

def test_disabled_groups_not_executed():
    layer=GroupedLowRankResidual(8,8,4,2);called=[]
    handles=[m.register_forward_hook(lambda m,a,o,g=g:called.append(g)) for g,m in enumerate(layer.down)]
    layer(torch.randn(2,3,8),torch.tensor([[1,0,0,0],[0,0,0,0]],dtype=torch.bool))
    for h in handles:h.remove()
    assert called==[0]

def test_group_sparse_equals_dense_reference():
    layer=GroupedLowRankResidual(8,6,4,2)
    for m in layer.up:torch.nn.init.normal_(m.weight)
    x=torch.randn(3,5,8);m=torch.tensor([[1,0,1,0],[0,0,1,1],[0,0,0,0]],dtype=torch.bool)
    c=m.float()*0.7
    dense=sum(b(a(x))*c[:,g,None,None] for g,(a,b) in enumerate(zip(layer.down,layer.up)))
    torch.testing.assert_close(layer(x,m,c),dense)

def test_disabled_coefficients_rejected():
    with pytest.raises(ValueError):
        GroupedLowRankResidual(2,2,1,1)(torch.ones(1,1,2),torch.zeros(1,1,dtype=torch.bool),torch.ones(1,1))

def test_adapter_gradient_through_frozen_tail():
    layer=GroupedLowRankResidual(8,8,4,2);tail=torch.nn.Linear(8,3).requires_grad_(False)
    x=torch.randn(2,5,8)
    tail(x+layer(x,torch.ones(2,4,dtype=torch.bool))).square().mean().backward()
    assert layer.up[0].weight.grad.abs().sum()>0 and tail.weight.grad is None

def test_rank_upper_bound_not_claimed_exact():
    layer=GroupedLowRankResidual(3,4,4,2)
    assert layer.rank_upper_bound(torch.ones(1,4,dtype=torch.bool)).item()==3

def test_memory_waits_for_available_time():
    assert eligible_records([record()],7,'v')==()
    assert len(eligible_records([record()],8,'v'))==1

def test_memory_future_records_do_not_affect_snapshot():
    past=[record()];future=record('future',10,16,18)
    assert [r.record_id for r in eligible_records(past+[future],8,'v')]==['a']

def test_memory_versions_fail_closed():
    with pytest.raises(ValueError): eligible_records([record()],9,'new')

def test_memory_duplicate_ids_rejected():
    with pytest.raises(ValueError): eligible_records([record(),record()],9,'v')

def test_memory_event_exclusion():
    assert eligible_records([record()],9,'v',frozenset({'old'}))==()

def test_memory_records_copy_input():
    x=torch.tensor([1.,0.]);r=VerifiedRecord('a',0,6,8,'v','e',x,torch.ones(1));x.zero_()
    assert r.key[0]==1

def test_memory_invalid_chronology_rejected():
    with pytest.raises(ValueError): record(valid=9,available=8)

def test_delta_memory_read_and_no_mutation():
    k=torch.tensor([[[1.,0.]]]);v=torch.tensor([[[3.]]]);m=gated_delta_replay(k,v,torch.ones(1,1,dtype=torch.bool),1.,1.)
    before=m.clone();torch.testing.assert_close(read_memory(m,k[:,0]),v[:,0]);torch.testing.assert_close(m,before)

def test_memory_padding_does_not_decay():
    k=torch.tensor([[[1.,0.],[0.,1.]]]);v=torch.tensor([[[3.],[99.]]]);mask=torch.tensor([[1,0]],dtype=torch.bool)
    m=gated_delta_replay(k,v,mask,0.1,1.)
    torch.testing.assert_close(m,torch.tensor([[[3.,0.]]]))

def test_memory_replay_has_gradients():
    k=torch.randn(2,3,4,requires_grad=True);v=torch.randn(2,3,5,requires_grad=True)
    gated_delta_replay(k,v,torch.ones(2,3,dtype=torch.bool)).square().sum().backward()
    assert k.grad is not None and v.grad is not None

def test_selection_respects_budget():
    assert select_plan(torch.tensor([[0.,10.,2.]]),torch.tensor([1.,9.,3.]),3.,('ref','large','small')).item()==2

def test_negative_gains_fall_back_to_reference():
    assert select_plan(torch.tensor([[0.,-1.,-2.]]),torch.tensor([1.,2.,3.]),3.,('ref','a','b')).item()==0

def test_candidate_permutation_invariance_with_ties():
    ids=('ref','b','a');g=torch.tensor([[0.,1.,1.]]);cost=torch.ones(3)
    i=select_plan(g,cost,2.,ids).item();p=torch.tensor([2,0,1]);ids2=tuple(ids[j] for j in p)
    j=select_plan(g[:,p],cost[p],2.,ids2).item()
    assert ids[i]==ids2[j]=='a'

def test_infeasible_budget_raises():
    with pytest.raises(ValueError): select_plan(torch.zeros(1,1),torch.tensor([2.]),1.,('ref',))

def test_regret_nonnegative_and_zero_at_best():
    gain=torch.tensor([[0.,2.,1.],[0.,-2.,-1.]])
    torch.testing.assert_close(selection_regret(gain,torch.tensor([1,0]),torch.ones(3),2.),torch.zeros(2))

def test_spectrum_equal_power_different_phase():
    y=torch.ones(4,dtype=torch.complex64);p=-y
    d=coefficient_diagnostics(p,y,torch.ones(4))
    assert d['log_energy_error']==0 and d['correlation']==-1

def test_spectrum_zero_field_finite():
    x=torch.zeros(4,dtype=torch.complex64)
    d=coefficient_diagnostics(x,x,torch.ones(4))
    assert d['correlation']==1 and d['log_energy_error']==0

def test_grid_coordinates_not_just_shapes():
    with pytest.raises(ValueError): assert_same_latitude_nodes(torch.tensor([-89.,89.]),torch.tensor([-90.,90.]))

def test_model_shapes_and_exact_no_edit_outputs():
    m,h,mem,e,t,on,_=small_batch();p=m(h,mem,e,t,on)
    assert p['gain'].shape==(3,6,3,3) and p['edit_response'].shape==(3,6,3,3,6)
    assert p['gain'][:,0].count_nonzero()==0 and p['edit_response'][:,0].count_nonzero()==0

def test_model_candidate_permutation_equivariant():
    m,h,mem,e,t,on,_=small_batch();perm=torch.tensor([2,0,5,1,4,3])
    p=m(h,mem,e,t,on);q=m(h,mem,e[perm],t,on[perm])
    torch.testing.assert_close(p['gain'][:,perm],q['gain'])

def test_model_gradients_and_ema_update():
    m,h,mem,e,t,on,targets=small_batch();before=next(m.base_target.parameters()).clone()
    loss=m.training_loss(m(h,mem,e,t,on),targets,on)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3);loss['total'].backward();opt.step();m.update_targets(0.)
    assert torch.isfinite(loss['total']) and not torch.equal(before,next(m.base_target.parameters()))
    assert all(p.grad is None for p in m.base_target.parameters())

def test_future_truth_not_an_inference_argument():
    assert 'truth' not in inspect.signature(PairedEditPredictor.forward).parameters

def test_model_checkpoint_roundtrip():
    m,h,mem,e,t,on,_=small_batch();n=PairedEditPredictor(6,4,9,6,16);n.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(h,mem,e,t,on)['gain'],n(h,mem,e,t,on)['gain'])
