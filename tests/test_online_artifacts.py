import copy
import json
from pathlib import Path
import numpy as np
import pytest
from earthdelta.online.io import sha256_file,decoded_sha256,write_json_once
from earthdelta.online.qc_reuse import verify_qc_sources,compose_index_manifest,load_bound_cache
from earthdelta.online.outcorr import fit_eof,fit_ridge,select_fit_config,predict_correction,make_fresh_features
from earthdelta.online.clock import FeedbackBundle,SupervisionRecord


def cached(tmp_path,i):
    x=(np.arange(6).reshape(2,3)+i).astype(np.float32);p=tmp_path/f'{i}.npy'
    with p.open('xb') as f:np.save(f,x,allow_pickle=False)
    return {'qc':{'index':i,'finite':True,'max_abs_z':2.,'decoded_sha256':decoded_sha256(x)},
            'cache':{'path':str(p),'sha256':sha256_file(p),'shape':[2,3],'dtype':'float32'},'eligible':True}


def test_bound_cache_refuses_tamper_and_cannot_be_mutated(tmp_path):
    row=cached(tmp_path,1);x=load_bound_cache(row)
    with pytest.raises(ValueError):x.setflags(write=True)
    bad=copy.deepcopy(row);bad['qc']['decoded_sha256']='0'*64
    with pytest.raises(ValueError,match='decoded'):load_bound_cache(bad)
    with Path(row['cache']['path']).open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='byte identity'):load_bound_cache(row)


def test_partial_qc_scope_and_overlap_cannot_be_promoted_to_full_year(tmp_path):
    oldrows={i:cached(tmp_path,i) for i in (1,3,4)};new=cached(tmp_path,2)
    old={'per_index':{str(k):v['qc'] for k,v in oldrows.items()},'cache_files':{str(k):v['cache'] for k,v in oldrows.items()}}
    kwargs=dict(support=range(5),missing=[0],overlap=[1],normalization_identity='frozen_train_std')
    r=compose_index_manifest(old,{1:oldrows[1],2:new},**kwargs)
    assert not r['whole_year_certified'] and len(r['indices'])==4
    assert r['indices']['1']['origin']=='bound_old_cache'
    with pytest.raises(ValueError,match='scope'):compose_index_manifest(old,{2:new},**kwargs)
    changed=copy.deepcopy(oldrows[1]);changed['qc']['decoded_sha256']='0'*64
    with pytest.raises(ValueError,match='overlap'):compose_index_manifest(old,{1:changed,2:new},**kwargs)


def test_qc_source_hashes_verified_before_any_array_open(tmp_path):
    store=tmp_path/'synthetic_store';store.mkdir();(store/'COMPLETE.json').write_text('{}')
    spec=tmp_path/'spec.json';write_json_once(spec,{'normalization':'bound'})
    qc=tmp_path/'qc.json';complete=sha256_file(store/'COMPLETE.json')
    write_json_once(qc,{'status':'OBSERVED','spec_sha256':sha256_file(spec),'store':str(store),'metadata':{'COMPLETE_sha256':complete}})
    kwargs=dict(store=store,metadata_bindings={'COMPLETE.json':complete})
    verify_qc_sources(qc,sha256_file(qc),spec,sha256_file(spec),**kwargs)
    with pytest.raises(ValueError):verify_qc_sources(qc,'0'*64,spec,sha256_file(spec),**kwargs)
    (store/'COMPLETE.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):verify_qc_sources(qc,sha256_file(qc),spec,sha256_file(spec),**kwargs)


def test_eof_ridge_recovers_signal_and_rejects_missing_training_features():
    rng=np.random.default_rng(312);fields=rng.normal(size=(40,2,3));eof=fit_eof(fields,6,[-45,45])
    np.testing.assert_allclose(eof.decode(eof.encode(fields)),fields,atol=1e-12)
    x=rng.normal(size=(80,5));weights=rng.normal(size=(5,3));y=x@weights
    model=fit_ridge(x[:60],y[:60],.001)
    assert np.mean((model.predict(x[60:])-y[60:])**2)<1e-4
    bad=x.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):fit_ridge(bad,y,.001)
    with pytest.raises(ValueError):fit_eof(fields[:3],6,[-45,45])
    assert predict_correction(model,None,eof) is None
    assert select_fit_config({'a':1.,'b':1.-1e-13},['a','b'])=='a'
    with pytest.raises(ValueError):select_fit_config({'a':1.},['a','b'])


def test_fresh_features_use_all_24_blocks_and_same_lead_history():
    rng=np.random.default_rng(4);eof=fit_eof(rng.normal(size=(8,2,3)),2,[-45,45])
    variables=('z','t','t2','p','u','q');records=[]
    for v in variables:
        for k in (1,2,3,4):
            rid=v+str(k);records.append(SupervisionRecord(rid,4,4+k,4+k,'v',frozenset({rid}),np.ones((2,3))*k,v,k))
    b=FeedbackBundle(4,tuple(records),variables);eofs={(v,k):eof for v in variables for k in (1,2,3,4)}
    x=make_fresh_features(b,b.lineage,8,eofs,np.zeros((2,3)),eof)
    assert x.shape==(50,)
    with pytest.raises(ValueError):make_fresh_features(b,b.lineage,8,{('z',1):eof},np.zeros((2,3)),eof)
    assert make_fresh_features(b,b.lineage,8,eofs,None,eof) is None


def test_artifact_output_is_write_once(tmp_path):
    p=tmp_path/'receipt.json';write_json_once(p,{'complete':False})
    with pytest.raises(FileExistsError):write_json_once(p,{'complete':True})
    assert json.loads(p.read_text())=={'complete':False}
