"""Monthly allocation experiment: deterministic execution, optional LLM decisions.

No broker, order API, or changes to the original reporting experiment.
Run --help for offline, live (bounded), and exact archived replay modes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TICKERS = ('VLUE', 'MTUM', 'QUAL', 'USMV')
VERSION = 'allocation-v1'
SYSTEM_PROMPT = '''Tu proposes les poids d'un portefeuille SIMULE de quatre ETF pour la periode suivante.
Utilise exclusivement les indicateurs historiques fournis, en fractions et non en pourcentages.
Recherche un compromis rendement/risque en tenant compte des couts et du turnover; aucune performance n'est garantie.
Ne mobilise pas de connaissance d'evenements posterieurs a decision_date. N'invente aucun indicateur.
Les calculs et la validation sont effectues par Python. Tu n'as aucun outil d'execution reelle.
Respecte exactement les actifs, la somme des poids, la concentration et le turnover autorises dans constraints.
Si un ajustement n'est pas justifie, propose les poids courants (s'ils respectent les contraintes).
Reponds uniquement avec un objet JSON : decision_date (copie exacte), target_weights (objet avec les quatre ETF),
reason (justification courte), cited_fields (liste de chemins exacts des indicateurs fournis, par exemple features.VLUE.return_6m).
Aucun autre champ. La justification ne constitue pas une preuve de performance.'''


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    start: str = '2021-01-01'
    end: str = '2025-12-31'
    initial_capital: float = 100.0
    cost_bps: float = 10.0
    max_weight: float = 0.40
    max_turnover: float = 0.10

    def __post_init__(self):
        if pd.Timestamp(self.start).day != 1 or pd.Timestamp(self.end) != pd.Timestamp(self.end)+pd.offsets.MonthEnd(0):
            raise ValueError('Choisir des mois calendaires complets.')
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError('Dates inversees.')
        values = [self.initial_capital, self.cost_bps, self.max_weight, self.max_turnover]
        if not all(math.isfinite(v) for v in values) or self.initial_capital <= 0:
            raise ValueError('Parametres non finis ou capital invalide.')
        if not 0 <= self.cost_bps <= 100 or not .25 <= self.max_weight <= 1 or not 0 <= self.max_turnover <= 1:
            raise ValueError('Couts ou contraintes invalides.')


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col='date', parse_dates=['date']).loc[:, list(TICKERS)]
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Historique vide, non trie ou dates dupliquees.')
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Prix absents, non finis ou non positifs.')
    if prices.index.to_series().diff().dt.days.max() > 7:
        raise ValueError('Historique avec interruption superieure a sept jours.')
    return prices


def features_at(prices: pd.DataFrame, asof: pd.Timestamp) -> dict:
    history = prices.loc[:asof, list(TICKERS)]
    if len(history) < 253 or history.index[-1] != asof:
        raise ValueError('253 observations passees requises avant une decision.')
    daily = history.pct_change(fill_method=None).iloc[1:]
    features = {}
    for ticker in TICKERS:
        column = history[ticker]
        trailing = column.iloc[-253:]
        features[ticker] = {
            **{f'return_{name}':float(column.iloc[-1]/column.iloc[-1-lag]-1)
               for name,lag in [('1m',21),('3m',63),('6m',126),('12m',252)]},
            'volatility_12m':float(daily[ticker].iloc[-252:].std(ddof=1)*np.sqrt(252)),
            'drawdown_12m':float((trailing/trailing.cummax()-1).min()),
        }
    return features


def make_payload(prices: pd.DataFrame, asof: pd.Timestamp, current: np.ndarray, config: Config) -> dict:
    return {
        'decision_date':asof.date().isoformat(),
        'features':features_at(prices, asof),
        'portfolio':{'current_weights':dict(zip(TICKERS,map(float,current)))},
        'constraints':{'allowed_assets':list(TICKERS), 'sum_weights':1.0, 'min_weight':0.0,
                       'max_weight':config.max_weight, 'max_one_way_turnover':config.max_turnover,
                       'transaction_cost_bps':config.cost_bps, 'simulation_only':True},
    }


def leaf_paths(value: dict, prefix: str = '') -> set[str]:
    paths = set()
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            paths.update(leaf_paths(item,path))
        else:
            paths.add(path)
    return paths


def validate_decision(output, payload: dict) -> tuple[np.ndarray | None, list[str]]:
    errors = []
    if not isinstance(output, dict) or set(output) != {'decision_date','target_weights','reason','cited_fields'}:
        return None,['invalid_contract']
    if output['decision_date'] != payload['decision_date']:
        errors.append('wrong_decision_date')
    weights = output['target_weights']
    if not isinstance(weights,dict) or set(weights) != set(TICKERS):
        return None,errors+['invalid_assets']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in weights.values()):
        return None,errors+['non_finite_or_non_numeric_weights']
    proposed = np.array([weights[t] for t in TICKERS],dtype=float)
    limits = payload['constraints']
    if abs(proposed.sum()-1.0) > 1e-9:
        errors.append('weights_do_not_sum_to_one')
    if (proposed < 0).any() or (proposed > limits['max_weight']+1e-12).any():
        errors.append('weight_limit')
    current = np.array([payload['portfolio']['current_weights'][t] for t in TICKERS])
    if .5*np.abs(proposed-current).sum() > limits['max_one_way_turnover']+1e-12:
        errors.append('decision_turnover_limit')
    if not isinstance(output['reason'],str) or not output['reason'].strip() or len(output['reason']) > 2000:
        errors.append('invalid_reason')
    cited = output['cited_fields']
    if not isinstance(cited,list) or not cited or any(not isinstance(x,str) or x not in leaf_paths(payload) for x in cited):
        errors.append('invalid_source_path')
    # Numerical sum tolerance only; the effective normalized weights are journaled.
    return (None if errors else proposed/proposed.sum()), errors


def decision(weights: np.ndarray, payload: dict, reason: str, cited: list[str]) -> dict:
    return {'decision_date':payload['decision_date'],'target_weights':dict(zip(TICKERS,map(float,weights))),
            'reason':reason,'cited_fields':cited}


def capped_weights(scores: np.ndarray, cap: float) -> np.ndarray:
    scores = np.maximum(np.asarray(scores,dtype=float),1e-12)
    weights, free, remaining = np.zeros(4), np.ones(4,dtype=bool), 1.0
    while free.any():
        candidate = remaining*scores[free]/scores[free].sum()
        indices = np.flatnonzero(free)
        over = candidate > cap+1e-12
        if not over.any():
            weights[indices] = candidate
            break
        weights[indices[over]] = cap
        free[indices[over]] = False
        remaining = 1.0-weights.sum()
    return weights


def inverse_volatility(payload: dict) -> dict:
    vol = np.array([payload['features'][t]['volatility_12m'] for t in TICKERS])
    weights = capped_weights(1/np.maximum(vol,1e-8),payload['constraints']['max_weight'])
    return decision(weights,payload,'Regle deterministe : inverse de la volatilite annualisee, plafonnee.',
                    [f'features.{t}.volatility_12m' for t in TICKERS])


def offline_fixture(payload: dict) -> dict:
    # Smoke-test provider, never described as a model result.
    current = np.array([payload['portfolio']['current_weights'][t] for t in TICKERS])
    leader = max(TICKERS,key=lambda t:payload['features'][t]['return_6m'])
    target = np.full(4,.20); target[TICKERS.index(leader)] = .40
    requested = .5*np.abs(target-current).sum()
    fraction = min(1.0,.8*payload['constraints']['max_one_way_turnover']/max(requested,1e-12))
    weights = current+fraction*(target-current)
    return decision(weights,payload,'FIXTURE HORS RESEAU : regle momentum pour tester le simulateur.',
                    [f'features.{leader}.return_6m'])


def execute_rebalance(shares: np.ndarray, prices: np.ndarray, target: np.ndarray, config: Config):
    """Solve self-financing trading with fees on half the actual traded notional.

fee = cost_rate/2 * sum(abs(target*(pre_value-fee) - current_notional)).
Fractional ETF units at adjusted closes represent total-return accounting units.
"""
    notional = shares*prices
    pre = float(notional.sum())
    current = notional/pre
    if .5*np.abs(target-current).sum() > config.max_turnover+1e-12:
        return shares.copy(),0.0,0.0,'execution_turnover_limit'
    rate = config.cost_bps/10000.0
    lower,upper = 0.0,rate*pre
    for _ in range(70):
        fee = (lower+upper)/2
        required = rate*.5*np.abs(target*(pre-fee)-notional).sum()
        if fee > required: upper = fee
        else: lower = fee
    fee = (lower+upper)/2
    new_shares = target*(pre-fee)/prices
    turnover = float(.5*np.abs(new_shares*prices-notional).sum()/pre)
    if turnover > config.max_turnover+1e-12:
        return shares.copy(),0.0,0.0,'execution_turnover_limit_after_fees'
    if not math.isclose(float((new_shares*prices).sum())+fee,pre,rel_tol=1e-12,abs_tol=1e-12):
        raise ArithmeticError('Capital non conserve apres frais.')
    return new_shares,float(fee),turnover,None


class BudgetExhausted(RuntimeError): pass
class ReplayMismatch(RuntimeError): pass


class ArchivedProvider:
    """One archived response per date. Each HTTP attempt consumes a budget slot.

No automatic retry on rejection/HTTP error; replay preserves those failures.
An attempt is persisted before HTTP so interrupted requests cannot evade budget.
"""
    def __init__(self, path: Path, mode: str, model: str, max_calls: int, max_tokens: int = 1000):
        self.path,self.mode,self.model,self.max_calls,self.max_tokens = path,mode,model,max_calls,max_tokens
        self.client = None
        self.last = None
        self.data = json.loads(path.read_text()) if path.exists() else {'version':VERSION,'records':{}}
        if self.data['version'] != VERSION:
            raise ReplayMismatch('Version de protocole differente.')
        self.new_calls = 0

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        tmp.replace(self.path)

    def __call__(self,payload: dict):
        key = payload['decision_date']
        material = {'prompt':SYSTEM_PROMPT,'payload':payload,'model':self.model,
                    'temperature':0,'max_tokens':self.max_tokens,'thinking':{'type':'disabled'},'version':VERSION}
        request_hash = digest(material)
        if key in self.data['records']:
            record = self.data['records'][key]
            if record['request_sha256'] != request_hash:
                raise ReplayMismatch(f'Contexte modifie pour {key}; utiliser une autre archive.')
            self.last = record
            return record.get('output')  # Interrupted/error records remain missing decisions.
        if self.mode == 'replay':
            raise ReplayMismatch(f'Decision absente de l archive : {key}')
        if len(self.data['records']) >= self.max_calls:
            raise BudgetExhausted(f'Budget total de {self.max_calls} appels atteint; execution arretee.')
        from dotenv import load_dotenv
        load_dotenv(ROOT/'.env')
        if not os.getenv('DEEPSEEK_API_KEY'):
            raise RuntimeError('DEEPSEEK_API_KEY absente.')
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                                 base_url=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com'),
                                 timeout=90,max_retries=0)
        record = {'request_sha256':request_hash, 'request':material,
                  'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'attempt_started'}
        self.data['records'][key] = record
        self.save()
        self.new_calls += 1
        import time
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,temperature=0,max_tokens=self.max_tokens,
                response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}},
                messages=[{'role':'system','content':SYSTEM_PROMPT},
                          {'role':'user','content':canonical(payload)}])
            raw = response.choices[0].message.content or ''
            record.update({'raw_content':raw,'request_id':response.id,'model_returned':response.model,
                           'finish_reason':response.choices[0].finish_reason,
                           'usage':response.usage.model_dump() if response.usage else None})
            try:
                parsed = json.loads(raw,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                canonical(parsed)  # Reject overflow such as 1e999 as well as NaN/Infinity.
                record['output'] = parsed
                record['status'] = 'response_received'
            except (json.JSONDecodeError,ValueError):
                record['status'] = 'invalid_json'
        except Exception as exc:
            # Never persist headers, tokens, environment or arbitrary exception messages.
            record.update({'status':'api_error','error_type':type(exc).__name__})
        record['elapsed_seconds'] = time.perf_counter()-started
        self.last = record
        self.save()
        print(f'DeepSeek {len(self.data["records"]):02d}/{self.max_calls}: {key} {record["status"]}',flush=True)
        if record['status'] == 'api_error':
            raise RuntimeError(f"Appel API interrompu ({record['error_type']}); tentative archivee, verifier la connexion avant de reprendre.")
        return record.get('output')


def simulate(prices: pd.DataFrame, config: Config, provider: Callable, name: str):
    start,end = pd.Timestamp(config.start),pd.Timestamp(config.end)
    prior = prices.loc[prices.index < start]
    if prior.empty or prices.index[-1].to_period('M') < end.to_period('M'):
        raise ValueError('Historique insuffisant pour la periode demandee.')
    anchor = prior.index[-1]
    shares = config.initial_capital*.25/prices.loc[anchor].to_numpy()
    daily = {anchor:config.initial_capital}
    records = []
    for month in pd.period_range(start,end,freq='M'):
        window = prices.loc[(prices.index >= month.start_time)&(prices.index <= min(month.end_time,end))]
        if window.empty: raise ValueError(f'Mois absent : {month}')
        asof = prices.loc[prices.index < window.index[0]].index[-1]
        pre_decision_notional = shares*prices.loc[asof].to_numpy()
        current = pre_decision_notional/pre_decision_notional.sum()
        payload = make_payload(prices,asof,current,config)
        output = provider(copy.deepcopy(payload))
        target,reasons = validate_decision(output,payload)
        # The provider has already returned; only now use the next-session price.
        execution_date = window.index[0]
        execution_price = window.iloc[0].to_numpy()
        old_shares = shares.copy()
        pre_execution_value = float((shares*execution_price).sum())
        fee,turnover = 0.0,0.0
        if target is not None:
            shares,fee,turnover,execution_error = execute_rebalance(shares,execution_price,target,config)
            if execution_error: reasons.append(execution_error)
        accepted = not reasons
        for day,row in window.iterrows():
            daily[day] = float((shares*row.to_numpy()).sum())
        records.append({'strategy':name, 'decision_date':asof.date().isoformat(),
                        'execution_date':execution_date.date().isoformat(),
                        'period_end':window.index[-1].date().isoformat(),
                        'payload':payload,'payload_sha256':digest(payload),'proposal':output,
                        'accepted':accepted,'rejection_reasons':reasons,
                        'fallback':'none' if accepted else 'hold_existing_positions',
                        'effective_target_weights':dict(zip(TICKERS,map(float,target))) if accepted else None,
                        'positions_before':dict(zip(TICKERS,map(float,old_shares))),
                        'positions_after':dict(zip(TICKERS,map(float,shares))),
                        'pre_execution_value':pre_execution_value,'transaction_cost_units':fee,
                        'actual_one_way_turnover':turnover,
                        'post_execution_value':float((shares*execution_price).sum()),
                        'month_end_value':daily[window.index[-1]]})
    equity = pd.Series(daily,name=name).sort_index()
    return equity,records


def summary(equity: pd.Series, records: list[dict]) -> dict:
    returns = equity.pct_change().iloc[1:]
    years = (equity.index[-1]-equity.index[0]).days/365.25
    vol = float(returns.std(ddof=1)*np.sqrt(252))
    total_fees = sum(r['transaction_cost_units'] for r in records)
    return {'final_value':float(equity.iloc[-1]),'cumulative_pnl':float(equity.iloc[-1]-equity.iloc[0]),
            'annualized_return':float((equity.iloc[-1]/equity.iloc[0])**(1/years)-1),
            'annualized_daily_volatility':vol,
            'sharpe_zero_rf_daily':float(returns.mean()/returns.std(ddof=1)*np.sqrt(252)) if vol > 0 else None,
            'max_daily_drawdown':float((equity/equity.cummax()-1).min()),
            'transaction_cost_units':total_fees,
            'average_monthly_one_way_turnover':float(np.mean([r['actual_one_way_turnover'] for r in records])),
            'decisions':len(records),'accepted':sum(r['accepted'] for r in records),
            'rejected':sum(not r['accepted'] for r in records)}


def run(prices: pd.DataFrame, config: Config, provider: Callable, agent_name: str):
    policies = {'equal_weight':lambda p:decision(np.full(4,.25),p,'Allocation equiponderee.', ['constraints.sum_weights']),
                'inverse_volatility':inverse_volatility, agent_name:provider}
    paths,records,metrics = {},{},{}
    for name,policy in policies.items():
        paths[name],records[name] = simulate(prices,config,policy,name)
        metrics[name] = summary(paths[name],records[name])
    return pd.DataFrame(paths),records,metrics


def save_results(output: Path, curves: pd.DataFrame, records: dict, metrics: dict, config: Config, mode: str, provenance: dict):
    output.mkdir(parents=True,exist_ok=True)
    curves.to_csv(output/'equity.csv',index_label='date')
    pd.DataFrame(metrics).T.to_csv(output/'metrics.csv',index_label='strategy')
    (output/'decisions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    report = {'protocol_version':VERSION,'mode':mode,'config':asdict(config),'metrics':metrics,
              'provenance':provenance,'limitations':[
                  'Historical exploratory simulation; LLM pretraining may include future events.',
                  'Adjusted prices are total-return accounting proxies; point-in-time adjustment vintages unavailable.',
                  'Execution at first available close AFTER observation; no spread/impact/liquidity model.',
                  'Initial capital already invested 25% per ETF before first decision; initial entry and terminal exit fees excluded.',
                  'Concentration cap applies to target weights; market drift may exceed it between rebalances.',
                  'Invalid proposals and execution-time turnover breaches hold existing positions; no silent clipping.',
                  'One LLM response per date, no selection among repeated responses; rejection still consumes API budget.',
                  'API and infrastructure costs excluded from portfolio NAV; token usage is reported separately.',
                  'Simulation acceptance is not a recorded human approval or a permission to place real orders.',
                  'Original report metrics use a different execution schedule and monthly risk measures.',
              ]}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'equal_weight':'Équipondéré','inverse_volatility':'Inverse volatilité',
            'deepseek_allocation':'Allocation DeepSeek','offline_fixture':'Fixture hors réseau (non LLM)'}
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for column in curves:
        axes[0].plot(curves.index,curves[column],label=labels[column],linewidth=1.7)
        axes[1].plot(curves.index,100*(curves[column]/curves[column].cummax()-1),linewidth=1.3)
    axes[0].set_ylabel('Capital, base 100'); axes[0].legend(loc='upper left')
    axes[0].set_title(f'Allocation mensuelle — {config.start[:4]}–{config.end[:4]} — coût {config.cost_bps:g} pb')
    axes[1].set_ylabel('Drawdown (%)')
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output/'pnl_comparison.png',dpi=170); plt.close(fig)
    lines=['# Expérience d’allocation mensuelle','',
           f'Mode : {mode}. Du {config.start} au {config.end}, capital initial {config.initial_capital:g} unités.',
           'Décision après clôture du mois précédent ; exécution au premier cours de clôture suivant.',
           f'Coûts : {config.cost_bps:g} pb, poids cibles ≤ {config.max_weight:.0%}, turnover mensuel ≤ {config.max_turnover:.0%}.','',
           '| Stratégie | Capital final | PnL | Rendement annualisé | Volatilité | Drawdown max. | Décisions rejetées |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        lines.append(f'| {labels[name]} | {m["final_value"]:.2f} | {m["cumulative_pnl"]:+.2f} | {m["annualized_return"]:.2%} | {m["annualized_daily_volatility"]:.2%} | {m["max_daily_drawdown"]:.2%} | {m["rejected"]}/{m["decisions"]} |')
    lines += ['', 'Les métriques de risque sont calculées sur les rendements quotidiens ; le taux sans risque est nul.',
              'La politique inverse volatilité utilise les 252 derniers rendements quotidiens, puis plafonne les poids.',
              'Chaque politique est soumise au même validateur et à la même exécution. Un rejet conserve les positions existantes.',
              'Les frais sont financés par le portefeuille ; le coût porte sur la moitié du volume réellement échangé.',
              '', 'Cette expérience historique est exploratoire : le modèle peut connaître des événements futurs par son préentraînement.',
              'Les coûts API et humains sont exclus. Les données ajustées ne sont pas des historiques de prix point-in-time.',
              'Les dates et conventions diffèrent du backtest initial du mémoire ; ses métriques ne sont pas remplacées.',
              '', '![Capital et drawdown](pnl_comparison.png)', '']
    if mode=='offline': lines.insert(2,'**Test technique : aucune décision de ce run ne provient d’un LLM.**')
    (output/'RESULTATS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['offline','live','replay'],default='offline')
    parser.add_argument('--start',default='2021-01-01');parser.add_argument('--end',default='2025-12-31')
    parser.add_argument('--cost-bps',type=float,default=10)
    parser.add_argument('--prices',type=Path,default=ROOT/'data/daily_prices.csv')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--model',default='deepseek-v4-flash')
    parser.add_argument('--max-api-calls',type=int,default=0)
    parser.add_argument('--max-output-tokens',type=int,default=1000)
    args=parser.parse_args()
    config=Config(start=args.start,end=args.end,cost_bps=args.cost_bps)
    prices=load_prices(args.prices)
    if args.max_output_tokens < 1 or args.max_api_calls < 0: parser.error('Budget negatif ou plafond de jetons invalide.')
    if args.mode!='offline':
        if args.archive is None: parser.error('--archive requis pour live/replay.')
        if args.mode=='live' and args.max_api_calls==0: parser.error('--max-api-calls doit etre explicite pour live.')
        provider=ArchivedProvider(args.archive,args.mode,args.model,args.max_api_calls,args.max_output_tokens)
        name='deepseek_allocation'
    else: provider=offline_fixture;name='offline_fixture'
    curves,records,metrics=run(prices,config,provider,name)
    provenance={'price_file':str(args.prices),'price_sha256':hashlib.sha256(args.prices.read_bytes()).hexdigest(),
                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'generated_at_utc':datetime.now(timezone.utc).isoformat(),'new_api_calls':getattr(provider,'new_calls',0)}
    if isinstance(provider,ArchivedProvider):
        provenance.update({'archive':str(args.archive),'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                           'total_archived_attempts':len(provider.data['records']),
                           'total_reported_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in provider.data['records'].values()),
                           'model_requested':args.model})
    save_results(args.output,curves,records,metrics,config,args.mode,provenance)
    print(json.dumps({'metrics':metrics,'provenance':provenance},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
