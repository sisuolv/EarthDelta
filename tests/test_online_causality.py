"""Real negative cases for the actor/worker boundary, using synthetic fields."""
from dataclasses import replace
import numpy as np
import pytest
from earthdelta.online.timeindex import index_of,iso_of,daily_issues,calendar_pairs,checked_index
from earthdelta.online.clock import SupervisionRecord,SupervisionQueue,FeedbackBundle,Derived,require_fresh_lineage
from earthdelta.online.dabc import DABC,calendar_ewma
from earthdelta.online.stores import ActorStore,ScorerStore,actor_config,scoring_mask,feedback_eligible
from earthdelta.online.roles import require_fit_rows,role


def record(i=0,k=1,value=None,artifact=None,identity=None,channel='x'):
    identity=identity or f'{i}:{k}:{channel}'
    return SupervisionRecord(identity,i,i+k,i+k if artifact is None else artifact,'v1',frozenset({identity}),np.ones((2,3)) if value is None else value,channel,k)


def bundle(i=0):
    variables=('z','t','t2','p','u','q')
    return FeedbackBundle(i,tuple(record(i,k,channel=v) for v in variables for k in (1,2,3,4)),variables)


@pytest.mark.parametrize('i',[0,1,235,236,239,240,1460,1463])
def test_six_hour_calendar_roundtrip(i):assert index_of(iso_of(i))==i


@pytest.mark.parametrize('value',[-1,1464,True,1.2,'4'])
def test_bad_index_refused(value):
    with pytest.raises(ValueError):checked_index(value)


@pytest.mark.parametrize('s',['2021-01-01T00:00:00Z','2019-12-31T18:00:00Z','2020-01-01T01:00:00Z','2020-01-01T00:00:00'])
def test_invalid_calendar(s):
    with pytest.raises(ValueError):index_of(s)


def test_exact_calendar_lag_never_nearest_survivor():
    assert calendar_pairs([4,8,16,120,124],1)==((0,1),(3,4))
    assert calendar_pairs([4,8,16,120,124],30)==((0,4),)
    assert len(daily_issues('2020-01-01T00:00:00Z','2020-12-26T00:00:00Z'))==361


def test_record_and_returned_snapshots_cannot_mutate_history():
    x=np.array([2.]);r=record(value=x);q=SupervisionQueue();q.enqueue(r);x[0]=99
    a=q.release(1);b=q.released()
    assert a[0].value[0]==b[0].value[0]==2
    with pytest.raises(ValueError):a[0].value[0]=5
    with pytest.raises(ValueError):b[0].value.setflags(write=True)
    assert q.release(1)==()


def test_future_id_conflict_does_not_disclose_pending_record():
    q=SupervisionQueue();q.enqueue(record(identity='r'));q.enqueue(record(i=8,identity='r'))
    assert len(q.release(1))==1
    assert q.release(8)==()
    with pytest.raises(ValueError,match='identity conflict'):q.release(9)
    assert len(q.released())==1


@pytest.mark.parametrize('poison',[None,np.array([np.nan]),np.array([1e30])])
def test_future_truth_poison_does_not_change_actor_trace(poison):
    def replay(future):
        calls=[];logs=[];queue=SupervisionQueue();state=DABC(7)
        actor=ActorStore(lambda i:(calls.append(i),np.array([2.]))[1])
        # Scorer capability is never passed to the actor; no future feedback exists yet.
        scorer=ScorerStore(lambda i:future if i==20 else np.array([1.]))
        queue.enqueue(record(value=np.array([3.])))
        result=[]
        for t in (4,8,12):
            state.update(queue.release(t),t)
            prediction=actor.read_input(t,t)-state.correction(t)
            logs.append(('published',t,len(state._records)))
            result.append(prediction.tolist())
        return result,logs,calls,[(r.record_id,r.value.tolist()) for r in state._records]
    assert replay(poison)==replay(np.array([1.]))


def test_actor_cannot_invoke_future_input_loader():
    calls=[];store=ActorStore(lambda i:calls.append(i))
    with pytest.raises(PermissionError):store.read_input(5,4)
    assert calls==[]
    assert actor_config({'variables':['x'],'future_qc':[5],'mask':True})=={'variables':['x']}


def test_availability_is_max_but_decay_age_is_label_time():
    r1=record(i=0,artifact=20,value=np.array([0.]));r2=record(i=8,artifact=20,value=np.array([10.]))
    assert calendar_ewma([r1,r2],19,2) is None
    expected=10/(1+np.exp(-1))
    assert calendar_ewma([r1,r2],20,2)[0]==pytest.approx(expected)
    assert calendar_ewma([r1,r2],12,2,offline_fit=True,artifact_context=20)[0]==pytest.approx(expected)
    with pytest.raises(ValueError):calendar_ewma([r1],12,2,offline_fit=True)
    d=DABC(2);d.update([r1,r2],20);before=len(d._records);d.correction(24)
    assert len(d._records)==before
    with pytest.raises(ValueError):d.update([r1],24)


def test_fresh_lineage_requires_all_short_variables_and_exact_day():
    b=bundle(4);require_fresh_lineage(b,b.lineage,8)
    with pytest.raises(ValueError):FeedbackBundle(4,b.records[:4],b.variables)
    with pytest.raises(ValueError):FeedbackBundle(4,b.records[::4],b.variables)
    with pytest.raises(ValueError):require_fresh_lineage(b,{'omitted-truth'},8)
    with pytest.raises(ValueError):require_fresh_lineage(b,b.lineage,12)
    with pytest.raises(PermissionError):b.visible(7)


def test_missingness_per_scoring_cell_and_exact_feedback():
    bad={0,1,1096,1097,1098,712}
    idx=lambda d:index_of('2020-'+d+'T00:00:00Z')
    assert scoring_mask(idx('06-26'),(1,4,12,20),bad)==(True,False,True,True)
    assert scoring_mask(idx('06-27'),(1,4,12,20),bad)==(False,False,False,False)
    assert not feedback_eligible(idx('06-26'),bad)
    assert scoring_mask(idx('09-27'),(1,4,12,20),bad)==(True,True,True,True)
    assert not feedback_eligible(idx('09-30'),bad)
    assert not feedback_eligible(idx('10-01'),bad)
    assert feedback_eligible(idx('10-02'),bad)
    assert scoring_mask(idx('09-26'),(1,4,12,20),bad)==(True,True,True,False)


def test_gap_history_is_valid_but_not_a_refit_target():
    feb29=index_of('2020-02-29T00:00:00Z');mar1=feb29+4
    bundle(feb29).visible(mar1)
    assert role(feb29)=='gap'
    with pytest.raises(PermissionError):require_fit_rows([4,feb29],final_refit=True)
    assert require_fit_rows([4,mar1],final_refit=True)==(4,mar1)
    with pytest.raises(PermissionError):require_fit_rows([mar1])
    artifact=Derived('final',index_of('2020-03-31T00:00:00Z'),('estimate','select'),frozenset({'raw'}))
    with pytest.raises(PermissionError):artifact.require_online(index_of('2020-03-27T00:00:00Z'))
    artifact.require_online(index_of('2020-04-01T00:00:00Z'))


def test_invalid_identity_and_clock_fail_closed():
    with pytest.raises(ValueError):replace(record(),record_id=None)
    with pytest.raises(ValueError):replace(record(),lineage='raw-string')
    q=SupervisionQueue();q.release(8)
    with pytest.raises(ValueError):q.release(4)
