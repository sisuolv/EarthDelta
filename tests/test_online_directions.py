import numpy as np
import pytest
from earthdelta.online.clock import SupervisionRecord,Derived
from earthdelta.online.directions import raw_gradient_mean,direction


def rec(i,value,artifact=0):
    return SupervisionRecord(str(i),i,i+4,max(i+4,artifact),'v',frozenset({str(i)}),np.array(value,np.float32))


def provenance(at=0):return Derived('m',at,('estimate',),frozenset({'raw'}))


def test_raw_mean_and_common_centering_for_time_control():
    records=[rec(4,[100,0]),rec(8,[0,1])];mean=raw_gradient_mean(records)
    np.testing.assert_allclose(mean,[50,.5])
    with pytest.raises(PermissionError):raw_gradient_mean([rec(240,[1,1])])
    history=[rec(4,[3,1]),rec(120,[3,1])];mu=np.array([1.,2.])
    args={'mean_provenance':provenance()}
    np.testing.assert_allclose(direction('lag1c',124,history,mu,**args),direction('delay30',124,history,mu,**args))
    assert not np.allclose(direction('lag1',124,history,mu,**args),direction('delay30',124,history,mu,**args))
    assert direction('lag1',128,history,mu,**args) is None


def test_offline_artifact_context_not_retroactive_online_feedback():
    records=[rec(4,[1,2],artifact=236)]
    assert direction('lag1',8,records,np.zeros(2),mean_provenance=provenance()) is None
    assert direction('lag1',8,records,np.zeros(2),mean_provenance=provenance(236),offline_fit=True,artifact_context=236) is not None
    assert direction('lag1c',8,records,np.array([1,2]),mean_provenance=provenance(236),offline_fit=True,artifact_context=236) is None
    assert direction('static',8,[],np.zeros(2),mean_provenance=provenance()) is None
    with pytest.raises(PermissionError):direction('static',8,[],np.ones(2),mean_provenance=provenance(236))
    with pytest.raises(ValueError):direction('random',8,records,np.zeros(2),mean_provenance=provenance())
