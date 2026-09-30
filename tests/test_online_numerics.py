import numpy as np
import pytest
import torch
from torch import nn
from earthdelta.online.losses import area_weights,normalized_step_mse,cells,wbx_evaluate,validate_wbx_binding
from earthdelta.online.grad_utils import (safe_weight_editor,weight_gradient,unit,cosine,
    split_direction,response_effect,calibrate_amplitude,serialize_gradient)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([nn.Module(),nn.Module()])
        for b in self.blocks:
            b.attn=nn.Module();b.attn.proj=nn.Linear(2,2,bias=False,dtype=torch.float64)
    def forward(self,x):
        for b in self.blocks:x=b.attn.proj(x).tanh()
        return x


@pytest.mark.parametrize('width',[3,8,256])
def test_constant_error_area_and_differentiation(width):
    x=torch.full((2,4,2,4,width),3.,dtype=torch.float64,requires_grad=True)
    loss=normalized_step_mse(x,torch.zeros_like(x),[-60,-20,20,60],np.ones((4,2)))
    assert loss.item()==pytest.approx(9.)
    grad=torch.autograd.grad(loss,x)[0]
    direction=torch.ones_like(x);eps=1e-5
    fd=(normalized_step_mse(x+eps*direction,torch.zeros_like(x),[-60,-20,20,60],np.ones((4,2)))-
        normalized_step_mse(x-eps*direction,torch.zeros_like(x),[-60,-20,20,60],np.ones((4,2))))/(2*eps)
    assert (grad*direction).sum().item()==pytest.approx(fd.item(),rel=1e-9)
    np.testing.assert_allclose(cells(x.detach().numpy(),np.zeros(x.shape),[-60,-20,20,60]),9.)


def fields():
    import xarray as xr
    rng=np.random.default_rng(8);shape=(8,4,8,16);y=rng.normal(size=shape)
    f=y+rng.normal(size=shape);p=y+.8*(f-y)
    inits=np.datetime64('2020-03-01','ns')+np.arange(8)*np.timedelta64(1,'D')
    leads=np.array([6,24,72,120],dtype='timedelta64[h]').astype('timedelta64[ns]')
    lat=np.linspace(-78.75,78.75,8)
    coords={'init_time':inits,'lead_time':leads,'latitude':lat,'longitude':np.arange(16)*22.5}
    make=lambda a:{'synthetic':xr.DataArray(a,dims=tuple(coords),coords=coords)}
    return make(p),make(f),make(y),inits,(inits[:,None]+leads).ravel(),lat


def test_required_wbx_hand_v8_and_new_metric_agree():
    from earthdelta.v8.standard_metrics import relative_rmse
    from earthdelta.wbx import _vendor
    p,f,y,inits,valid,lat=fields()
    a=wbx_evaluate(p,y,init_times=inits,valid_times=valid)['rmse.synthetic'].values
    b=wbx_evaluate(f,y,init_times=inits,valid_times=valid)['rmse.synthetic'].values
    v=y['synthetic'].values;forecast=p['synthetic'].values
    weighted=np.zeros((8,4));den=16*sum(np.cos(np.deg2rad(lat)))
    for h in range(8):
        for w in range(16):weighted+=np.cos(np.deg2rad(lat[h]))*(forecast[:,:,h,w]-v[:,:,h,w])**2/den
    np.testing.assert_allclose(a,np.sqrt(weighted.mean(0)),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(100*(1-a/b),relative_rmse(forecast,v,f['synthetic'].values,lat),atol=1e-12)
    _vendor.assert_no_beam()


def test_wbx_actual_coordinate_and_valid_time_mismatch_refused():
    p,f,y,i,v,lat=fields()
    with pytest.raises(ValueError):validate_wbx_binding(p,y,i,v+np.timedelta64(6,'h'))
    shifted={'synthetic':y['synthetic'].assign_coords(latitude=lat[::-1])}
    with pytest.raises(ValueError):validate_wbx_binding(p,shifted,i,v)
    with pytest.raises(ValueError):validate_wbx_binding(p,y,i,None)


def test_safe_editor_entry_failure_cleans_own_hooks_and_restores(monkeypatch):
    m=Tiny().train();x=torch.ones(1,2,dtype=torch.float64);original=m(x).detach().clone()
    own=m.blocks[0].attn.proj.register_forward_hook(lambda m,i,o:o)
    def fail(*a,**k):raise RuntimeError('injected registration failure')
    monkeypatch.setattr(m.blocks[1].attn.proj,'register_forward_hook',fail)
    deltas={0:torch.ones(2,2,dtype=torch.float64),1:torch.ones(2,2,dtype=torch.float64)}
    with pytest.raises(RuntimeError,match='injected'):
        with safe_weight_editor(m,deltas,blocks=(0,1)):pass
    assert len(m.blocks[0].attn.proj._forward_hooks)==1
    torch.testing.assert_close(m(x),original);assert m.training
    own.remove()


def test_editor_rejects_bad_second_shape_before_entry_and_reentry():
    m=Tiny();a=torch.zeros(2,2,dtype=torch.float64)
    with pytest.raises(ValueError):
        with safe_weight_editor(m,{0:a,1:a[:1]},blocks=(0,1)):pass
    assert not m.blocks[0].attn.proj._forward_hooks
    with safe_weight_editor(m,{0:a},blocks=(0,)):
        with pytest.raises(RuntimeError):
            with safe_weight_editor(m,{0:a},blocks=(0,)):pass
    assert not m.blocks[0].attn.proj._forward_hooks


def test_gradient_agrees_with_eager_and_central_edit_difference():
    torch.manual_seed(31);m=Tiny().train();x=torch.randn(3,2,dtype=torch.float64)
    before=[p.detach().clone() for p in m.parameters()];flags=[p.requires_grad for p in m.parameters()];rng=torch.get_rng_state().clone()
    eager=torch.cat([g.ravel() for g in torch.autograd.grad(m(x).square().mean(),tuple(m.parameters()))])
    g,loss=weight_gradient(m,(0,1),lambda:m(x).square().mean())
    np.testing.assert_allclose(g.numpy(),eager.numpy(),rtol=1e-6,atol=1e-9)
    u=np.arange(1.,9.);u/=np.linalg.norm(u);eps=1e-5;values=[]
    for sign in (1,-1):
        with safe_weight_editor(m,split_direction(m,(0,1),u,eps*sign),blocks=(0,1)):
            values.append(m(x).square().mean().item())
    assert (values[0]-values[1])/(2*eps)==pytest.approx(float(g.numpy()@u),rel=1e-5,abs=1e-8)
    assert m.training and [p.requires_grad for p in m.parameters()]==flags
    assert torch.equal(rng,torch.get_rng_state())
    for p,old in zip(m.parameters(),before):torch.testing.assert_close(p,old,rtol=0,atol=0)


def test_response_uses_all_channels_and_step_four_not_last():
    f=torch.ones((1,5,2,2,3),dtype=torch.float64);y=torch.zeros_like(f);e=f.clone();e[:,4,1]+=2
    assert response_effect(e,f,y,[-45,45])==pytest.approx(np.sqrt(2.))
    padded=lambda x:torch.cat([x,torch.zeros((1,16,2,2,3),dtype=x.dtype)],1)
    assert response_effect(padded(e),padded(f),padded(y),[-45,45])==pytest.approx(np.sqrt(2.))
    with pytest.raises(ValueError):response_effect(e[:,:4],f[:,:4],y[:,:4],[-45,45])


def test_calibration_scope_bracket_zero_and_nonmonotonic_failure():
    got=calibrate_amplitude(lambda a:np.full(16,2*a),10.)
    assert abs(got['amplitude']-.05)<=.0025
    assert calibrate_amplitude(None,10.,all_directions_zero=True)['amplitude']==0
    with pytest.raises(ValueError,match='registered issue'):calibrate_amplitude(lambda a:np.ones(15),10.)
    with pytest.raises(ValueError,match='bracketed'):calibrate_amplitude(lambda a:np.ones(16)*.001,10.)
    with pytest.raises(ValueError,match='nonmonotonic'):calibrate_amplitude(lambda a:np.ones(16)*(1-a),1.)


def test_fp32_small_gradient_roundtrip_and_immutable_cache(tmp_path):
    x=np.linspace(-2,3,1000)*1e-9;p=tmp_path/'g.npy';r=serialize_gradient(p,x)
    assert r['roundtrip_cosine']>.99999 and r['fp16_underflow_fraction']>.99
    assert r['dtype']=='float32'
    with pytest.raises(FileExistsError):serialize_gradient(p,x)
    with pytest.raises(ValueError):serialize_gradient(tmp_path/'nan.npy',np.array([np.nan]))
    with pytest.raises(ValueError):serialize_gradient(tmp_path/'underflow.npy',np.array([1e-100]))
    assert unit(np.zeros(2)) is None and cosine(np.zeros(2),np.ones(2)) is None
    raw=np.array([[100.,0.],[0.,1.]])
    assert not np.allclose(unit(raw.mean(0)),unit(np.mean([unit(x) for x in raw],axis=0)))
