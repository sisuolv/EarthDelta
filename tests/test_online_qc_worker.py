import importlib.util
from pathlib import Path
import numpy as np
import pytest
from earthdelta.online.runtime import require_fit_index


def inspect():
    p=Path(__file__).resolve().parents[1]/'scripts/online_qc_prepare.py'
    spec=importlib.util.spec_from_file_location('online_qc_prepare_test',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.inspect_row


def test_qc_rejects_layout_and_uses_raw_source_precision():
    f=inspect();x=np.ones((2,2,3),np.float32)
    row=f(x,2,np.array([.999999999,0]),np.ones(2),(2,2,3),1577880000)
    assert row['finite'] and row['max_abs_z']==1.
    with pytest.raises(ValueError):f(x.astype(np.float64),2,np.zeros(2),np.ones(2),(2,2,3),0)
    with pytest.raises(ValueError):f(x,2,np.zeros(2),np.zeros(2),(2,2,3),0)
    x[0,0,0]=np.nan
    bad=f(x,2,np.zeros(2),np.ones(2),(2,2,3),0)
    assert not bad['finite'] and bad['max_abs_z'] is None


@pytest.mark.parametrize('i',[365,712,1463,1464,-1])
def test_fit_access_refuses_analysis_truth_before_loader(i):
    with pytest.raises((PermissionError,ValueError)):require_fit_index(i)


def test_fit_access_accepts_last_registered_target_support():
    assert require_fit_index(360)==360
