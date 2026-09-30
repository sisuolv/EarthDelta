import numpy as np
import pytest
from earthdelta.online.bank import physical_cells,write_prediction,fit_denominators


def test_physical_bank_layout_uses_variables_then_steps(tmp_path):
    prediction=np.ones((6,6,2,3),np.float32)*np.arange(1,7)[:,None,None,None]
    result=physical_cells(prediction,np.zeros_like(prediction),[-45,45])
    assert result.shape==(6,6)
    np.testing.assert_allclose(result,np.tile(np.arange(1,7)**2,(6,1)))
    with pytest.raises(ValueError):physical_cells(prediction[:4],prediction[:4],[-45,45])
    p=tmp_path/'f0.npz';row=write_prediction(p,prediction,norm69_step4=np.zeros((69,2,3)))
    assert row['lead_steps']==[1,2,3,4,12,20] and row['all69_step4']
    with pytest.raises(FileExistsError):write_prediction(p,prediction)


def test_estimate_denominators_cannot_be_contaminated_by_gap_or_select():
    losses=np.ones((4,6,6));losses[1]=4;losses[2]=100;losses[3]=900
    result=fit_denominators(losses,[4,8,224,240],['estimate','estimate','gap','select'])
    np.testing.assert_allclose(result['short_mse'],2.5)
    np.testing.assert_allclose(result['selection_mse'],2.5)
    assert result['source_indices']==[4,8]
    with pytest.raises(ValueError):fit_denominators(losses[:1]*0,[4],['estimate'])


def test_float32_cache_scoring_is_identical_to_float64_view():
    rng=np.random.default_rng(5)
    p=(rng.normal(size=(6,6,2,3))*1000).astype(np.float32)
    y=(rng.normal(size=p.shape)*500).astype(np.float32)
    expected=((p.astype(np.float64)-y.astype(np.float64))**2).mean((-2,-1)).T
    np.testing.assert_allclose(physical_cells(p,y,[-45,45]),expected,rtol=1e-14)
    np.testing.assert_array_equal(physical_cells(p,y,[-45,45]),physical_cells(p.astype(np.float64),y.astype(np.float64),[-45,45]))
