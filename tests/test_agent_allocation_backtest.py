"""Financial accounting, causal inputs and archived model decision checks."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import agent_allocation_backtest as a

@pytest.fixture
def prices():
    dates=pd.bdate_range('2019-01-01','2021-03-31'); x=np.arange(len(dates))[:,None]
    values=100*np.exp(x*np.array([.0003,.0005,.0002,.0001])+np.sin(x/7)*np.array([.01,.02,.008,.005]))
    return pd.DataFrame(values,index=dates,columns=a.TICKERS)

@pytest.fixture
def payload(prices):
    return a.make_payload(prices,pd.Timestamp('2020-12-31'),np.full(4,.25),a.Config())

def equal(p):
    return a.decision(np.full(4,.25),p,'Poids fixes',['constraints.sum_weights'])

def test_inputs_unchanged_when_future_prices_change(prices,payload):
    altered=prices.copy(); altered.loc['2021-01-01':]*=np.array([10,.1,5,2])
    assert a.make_payload(altered,pd.Timestamp('2020-12-31'),np.full(4,.25),a.Config())==payload
    assert a.features_at(prices.loc[:'2020-12-31'],pd.Timestamp('2020-12-31'))==payload['features']

@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,'0.25'])
def test_non_numeric_weights_rejected(payload,value):
    output=equal(payload);output['target_weights']['VLUE']=value
    assert a.validate_decision(output,payload)[0] is None

@pytest.mark.parametrize('change,error',[
    ({'decision_date':'2021-01-31'},'wrong_decision_date'),
    ({'target_weights':dict(VLUE=.41,MTUM=.19,QUAL=.2,USMV=.2)},'weight_limit'),
    ({'target_weights':dict(VLUE=.4,MTUM=.1,QUAL=.25,USMV=.25)},'decision_turnover_limit'),
    ({'target_weights':dict(VLUE=.2,MTUM=.2,QUAL=.2,USMV=.2)},'weights_do_not_sum_to_one'),
    ({'target_weights':dict(VLUE=.25,MTUM=.25,QUAL=.25,SPY=.25)},'invalid_assets'),
    ({'cited_fields':['features.VLUE.return_6m.invented']},'invalid_source_path'),
])
def test_contract_rejections(payload,change,error):
    output=equal(payload);output.update(change)
    target,errors=a.validate_decision(output,payload)
    assert target is None and error in errors

@pytest.mark.parametrize('cost',[0,10,30])
def test_fees_are_self_financed_on_actual_trades(cost):
    old=np.full(4,25.); target=np.array([.35,.15,.25,.25])
    new,fee,turnover,error=a.execute_rebalance(old,np.ones(4),target,a.Config(cost_bps=cost,max_turnover=.2))
    assert error is None
    assert new.sum()+fee==pytest.approx(100,abs=1e-10)
    assert fee==pytest.approx(cost/10000*.5*np.abs(new-old).sum(),abs=1e-12)
    assert new/new.sum()==pytest.approx(target)
    assert turnover==pytest.approx(.5*np.abs(new-old).sum()/100)

def test_rejected_decision_keeps_actual_positions_and_charges_nothing(prices):
    equity,records=a.simulate(prices,a.Config(end='2021-03-31'),lambda p:None,'invalid')
    shares=25/prices.loc['2020-12-31'].to_numpy()
    assert equity.iloc[-1]==pytest.approx(float((shares*prices.iloc[-1]).sum()))
    for r in records:
        assert r['positions_before']==r['positions_after']
        assert r['transaction_cost_units']==0 and not r['accepted']

def test_execution_occurs_at_next_close_after_old_positions_receive_gap(prices):
    prices.loc[:,:]=100.;prices.loc['2021-01-01':,'VLUE']=200.
    def policy(p):
        assert p['decision_date']=='2020-12-31' and p['features']['VLUE']['return_1m']==0
        return equal(p)
    equity,records=a.simulate(prices,a.Config(end='2021-01-31',cost_bps=0,max_turnover=.5),policy,'equal')
    r=records[0]
    assert r['execution_date']=='2021-01-01' and r['pre_execution_value']==125
    assert equity.iloc[-1]==pytest.approx(125)
    assert r['positions_after']['VLUE']==pytest.approx(31.25/200)

def test_market_gap_can_block_execution(prices):
    prices.loc[:,:]=100.;prices.loc['2021-01-01':,'VLUE']=300.
    _,records=a.simulate(prices,a.Config(end='2021-01-31'),equal,'equal')
    r=records[0]
    assert r['rejection_reasons']==['execution_turnover_limit']
    assert r['positions_before']==r['positions_after'] and r['transaction_cost_units']==0

def test_identical_decisions_have_identical_pnl(prices):
    curves,records,_=a.run(prices,a.Config(end='2021-03-31'),equal,'identical_agent')
    pd.testing.assert_series_equal(curves.equal_weight,curves.identical_agent,check_names=False)
    assert [r['transaction_cost_units'] for r in records['equal_weight']]==[r['transaction_cost_units'] for r in records['identical_agent']]

def fake_client(provider,raw):
    def create(**kwargs):
        saved=json.loads(provider.path.read_text())
        assert list(saved['records'].values())[-1]['status']=='attempt_started'
        return SimpleNamespace(id='test',model='fake',usage=None,choices=[SimpleNamespace(message=SimpleNamespace(content=raw),finish_reason='stop')])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

def test_archive_replay_is_bound_to_inputs_and_budget(tmp_path,payload,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','dummy-for-offline-test')
    path=tmp_path/'archive.json'; provider=a.ArchivedProvider(path,'live','fake',1)
    provider.client=fake_client(provider,json.dumps(equal(payload)))
    assert provider(payload)==equal(payload)
    replay=a.ArchivedProvider(path,'replay','fake',0)
    assert replay(payload)==equal(payload) and replay.new_calls==0
    changed=copy.deepcopy(payload);changed['constraints']['transaction_cost_bps']=30
    with pytest.raises(a.ReplayMismatch): replay(changed)
    changed['decision_date']='2021-01-29'
    with pytest.raises(a.ReplayMismatch): replay(changed)
    with pytest.raises(a.BudgetExhausted): provider(changed)

@pytest.mark.parametrize('raw',['{"weight":NaN}','{"weight":1e999}','not JSON'])
def test_invalid_json_is_archived_and_replayed_as_rejection(tmp_path,payload,monkeypatch,raw):
    monkeypatch.setenv('DEEPSEEK_API_KEY','dummy-for-offline-test')
    provider=a.ArchivedProvider(tmp_path/'archive.json','live','fake',1)
    provider.client=fake_client(provider,raw)
    assert provider(payload) is None and provider.last['status']=='invalid_json'
    assert a.ArchivedProvider(provider.path,'replay','fake',0)(payload) is None

def test_api_error_is_archived_and_stops_campaign(tmp_path,payload,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','dummy-for-offline-test')
    provider=a.ArchivedProvider(tmp_path/'archive.json','live','fake',1)
    def fail(**kwargs): raise ConnectionError('private details must not be persisted')
    provider.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(RuntimeError,match='ConnectionError'): provider(payload)
    assert 'private details' not in provider.path.read_text()
    assert provider.last['status']=='api_error'
    assert a.ArchivedProvider(provider.path,'replay','fake',0)(payload) is None
