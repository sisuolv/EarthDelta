import importlib.util
from pathlib import Path
import numpy as np
import pytest
from earthdelta.online.io import sha256_file


def loader():
    p=Path(__file__).resolve().parents[1]/'scripts/online_p0b_edits.py'
    spec=importlib.util.spec_from_file_location('test_parameter_loader',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.load_gradients


def test_parameter_loader_binds_raw_vectors_and_mean_before_use(tmp_path):
    g=tmp_path/'g.npy';m=tmp_path/'mean.npy'
    with g.open('xb') as f:np.save(f,np.array([1,2],np.float32),allow_pickle=False)
    with m.open('xb') as f:np.save(f,np.array([1,2],np.float64),allow_pickle=False)
    receipt={'records':[{'issue_index':4,'label_available':8,'artifact_available':236,'version':'D','raw_label_ids':['label1','label2'],
                         'gradient':{'path':str(g),'sha256':sha256_file(g),'numel':2}}],
             'mean':{'path':str(m),'sha256':sha256_file(m),'artifact_available':236,'source_indices':[4]}}
    records,mean,provenance=loader()(receipt)
    assert records[0].value.dtype==np.float32 and mean.dtype==np.float64
    with pytest.raises(PermissionError):provenance.require_online(8)
    with g.open('ab') as f:f.write(b'change')
    with pytest.raises(RuntimeError,match='cache changed'):loader()(receipt)
