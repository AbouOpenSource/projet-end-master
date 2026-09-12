"""Offline PnL reconciliation: fixed ETF portfolio versus reporting workflow.

This verifies preservation of portfolio results, not an independent LLM strategy.
Run: experience/.venv/bin/python verify_agentic_pnl.py
Existing input archives and workflow implementations are not modified.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import agent_workflow
import evaluate_deepseek

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
CAPITAL = 100.0  # Same neutral units as the report; not a currency assumption.
COST_RATE = 0.001  # 10 bp per unit of one-way turnover.
TOLERANCE = 1e-12  # Floating-point reconciliation, not the LLM validator tolerance.
TICKERS = list(agent_workflow.FACTOR_TICKERS.values())


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col='date', parse_dates=['date'])
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f'Dates invalides : {path}')
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f'Valeurs absentes ou non finies : {path}')
    return frame


def main() -> None:
    inputs = [ROOT/'data/monthly_prices.csv', ROOT/'data/quality_report.json',
              RESULTS/'monthly_asset_returns.csv', RESULTS/'rebalance_diagnostics_10bps.csv',
              RESULTS/'wealth.csv', RESULTS/'deepseek_evaluation.json',
              ROOT/'agent_workflow.py', ROOT/'evaluate_deepseek.py',
              ROOT/'deepseek_agent_workflow.py', ROOT/'backtest.py', Path(__file__).resolve()]
    inputs += sorted(RESULTS.glob('agent_decision_trace_*.json'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    checks = {}

    def check(label: str, condition: bool) -> None:
        checks[label] = bool(condition)
        if not condition:
            raise ValueError(f'Échec de vérification : {label}')

    def close(label: str, actual, expected, atol: float = TOLERANCE) -> float:
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        check(label, actual.shape == expected.shape and bool(np.allclose(actual, expected, atol=atol, rtol=0)))
        return float(np.max(np.abs(actual-expected)))

    # Independent vector calculation from archived month-end prices.
    prices = read_frame(ROOT/'data/monthly_prices.csv')
    check('positive_prices', bool((prices > 0).all().all()))
    returns = prices.pct_change(fill_method=None).iloc[1:]
    archived_returns = read_frame(RESULTS/'monthly_asset_returns.csv')
    diagnostics = read_frame(RESULTS/'rebalance_diagnostics_10bps.csv')
    archived_wealth = read_frame(RESULTS/'wealth.csv')
    check('complete_monthly_dates', bool(prices.index.to_period('M').equals(
        pd.period_range(prices.index[0], prices.index[-1], freq='M'))))
    for name, frame in [('returns', archived_returns), ('diagnostics', diagnostics), ('wealth', archived_wealth)]:
        check(f'aligned_{name}_dates', returns.index.equals(frame.index))
    return_error = close('prices_reproduce_asset_returns', returns[archived_returns.columns], archived_returns)
    assets = returns[TICKERS]
    weights = pd.Series(1.0/len(TICKERS), index=TICKERS)
    gross = assets.mul(weights).sum(axis=1)
    drifted = (1.0+assets).mul(weights).div(1.0+gross, axis=0)
    turnover = drifted.sub(weights).abs().sum(axis=1)*0.5
    cost = turnover*COST_RATE
    simple = gross-cost
    diagnostic_errors = {}
    for field, computed in [('gross_return', gross), ('turnover', turnover),
                            ('transaction_cost', cost), ('net_return', simple)]:
        diagnostic_errors[field] = close(f'recomputed_{field}', computed, diagnostics[field])
    for ticker in TICKERS:
        close(f'post_return_weight_{ticker}', drifted[ticker], diagnostics[f'weight_{ticker}'])

    # Execute the actual deterministic graph, stopping only before file persistence.
    # No LLM call, synthetic policy, or archived-note overwrite is involved.
    graph = agent_workflow.build_graph()
    states = {}
    for date in returns.index:
        day = date.date().isoformat()
        state = graph.invoke({'requested_date': day}, interrupt_before=['persist'])
        check(f'quality_{day}', state['quality_ok'] is True)
        check(f'permissions_{day}', state['trace']['permissions'] == {
            'can_change_weights': False, 'can_send_orders': False, 'human_validation_required': True})
        check(f'target_weights_{day}', state['rebalance']['target_weights'] == weights.to_dict())
        states[day] = state
    workflow = pd.Series([states[d.date().isoformat()]['rebalance']['net_return'] for d in returns.index], index=returns.index)
    for field, computed in [('turnover', turnover), ('transaction_cost', cost)]:
        close(f'workflow_{field}', [states[d.date().isoformat()]['rebalance'][field] for d in returns.index], computed)
    workflow_return_error = close('workflow_preserves_net_returns', workflow, simple)
    simple_equity = CAPITAL*(1.0+simple).cumprod()
    workflow_equity = CAPITAL*(1.0+workflow).cumprod()
    spy_equity = CAPITAL*(1.0+returns['SPY']).cumprod()
    equity_error = close('workflow_preserves_equity', workflow_equity, simple_equity, atol=1e-9)
    close('reproduces_archived_multifactor_wealth', simple_equity, archived_wealth['Multifactor net (10 bps)'], atol=1e-9)
    close('reproduces_archived_spy_wealth', spy_equity, archived_wealth['Benchmark SPY'], atol=1e-9)
    start_equity = simple_equity.shift(1, fill_value=CAPITAL)
    workflow_start = workflow_equity.shift(1, fill_value=CAPITAL)
    monthly = pd.DataFrame({
        'simple_gross_return': gross, 'one_way_turnover': turnover,
        'transaction_cost_fraction': cost, 'simple_net_return': simple,
        'workflow_net_return': workflow, 'net_return_difference': workflow-simple,
        'simple_start_capital': start_equity, 'simple_monthly_pnl': start_equity*simple,
        'workflow_monthly_pnl': workflow_start*workflow,
        'simple_equity': simple_equity, 'workflow_equity': workflow_equity,
        'simple_cumulative_pnl': simple_equity-CAPITAL,
        'workflow_cumulative_pnl': workflow_equity-CAPITAL,
        'cumulative_pnl_difference': workflow_equity-simple_equity,
        'spy_equity': spy_equity, 'spy_cumulative_pnl': spy_equity-CAPITAL,
    })
    close('pnl_sums_to_equity_change', monthly['simple_monthly_pnl'].sum(), simple_equity.iloc[-1]-CAPITAL, atol=1e-9)
    def compare_trace(actual, expected, label):
        if isinstance(actual, dict) and isinstance(expected, dict):
            check(label+'_keys', actual.keys() == expected.keys())
            for key in actual:
                compare_trace(actual[key], expected[key], label+'.'+key)
        elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
            close(label, actual, expected)
        else:
            check(label, actual == expected)

    for path in sorted(RESULTS.glob('agent_decision_trace_*.json')):
        trace = json.loads(path.read_text())
        compare_trace(trace, states[trace['date']]['trace'], f'archived_trace_{trace["date"]}')

    # Historical responses cover sparse dates, so never compound them as a time series.
    archive = json.loads((RESULTS/'deepseek_evaluation.json').read_text())
    cases = {c.case_id:c for c in evaluate_deepseek.build_evaluation_cases(archive['configuration']['valid_cases'])}
    llm_rows = []
    for archived_case in archive['cases']:
        case = cases[archived_case['case_id']]
        if not case.expects_llm:
            check('blocked_case_excluded_from_pnl', archived_case['model_called'] is False)
            continue
        ref = case.state['rebalance']
        historical = case.kind == 'valid'
        day = case.state['date']
        for repeat in archived_case['repetitions']:
            output = repeat['llm_output']
            check(f'llm_permissions_{case.case_id}_{repeat["repeat"]}',
                  output['can_send_orders'] is False and output['can_change_weights'] is False
                  and output['human_validation_required'] is True)
            reported = output['reported_values']
            errors = {k: float(reported[k])-float(ref[k]) for k in ('net_return','transaction_cost','turnover')}
            capital = float(start_equity.loc[day]) if historical else None
            llm_rows.append({
                'case_id':case.case_id, 'kind':case.kind, 'date':day, 'repeat':repeat['repeat'],
                'historical_pnl_observation':historical,
                'source_net_return':ref['net_return'], 'reported_net_return':float(reported['net_return']),
                **{f'{k}_reporting_error':v for k,v in errors.items()},
                'historical_start_capital':capital,
                'implied_pnl_reporting_error_units':capital*errors['net_return'] if historical else None,
                'current_validator_accepted':evaluate_deepseek.score_model_output(case.state,output)['accepted'],
            })
    llm = pd.DataFrame(llm_rows)
    historic = llm[llm['historical_pnl_observation']]
    expected_repeats = archive['configuration']['repeats']
    check('complete_llm_archive', len(llm) == (archive['configuration']['valid_cases']+1)*expected_repeats)
    check('complete_historical_llm_archive', len(historic) == archive['configuration']['valid_cases']*expected_repeats)
    for rel,digest in hashes.items():
        check(f'unchanged_input_{rel}', hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest)
    report = {
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':'offline-pnl-preservation-v1', 'api_calls':0,
        'interpretation':'Same fixed portfolio; the reporting workflow has no independent positions or trading PnL. '
                         'Equality verifies data transmission and accounting, not LLM investment skill.',
        'initial_capital_units':CAPITAL, 'transaction_cost_rate':COST_RATE,
        'start':returns.index[0].date().isoformat(), 'end':returns.index[-1].date().isoformat(),
        'months':len(returns), 'target_weights':weights.to_dict(),
        'results':{
            'simple_final_equity':float(simple_equity.iloc[-1]),
            'workflow_final_equity':float(workflow_equity.iloc[-1]),
            'simple_cumulative_pnl':float(simple_equity.iloc[-1]-CAPITAL),
            'workflow_cumulative_pnl':float(workflow_equity.iloc[-1]-CAPITAL),
            'final_pnl_difference':float(workflow_equity.iloc[-1]-simple_equity.iloc[-1]),
            'max_absolute_equity_difference':equity_error,
            'max_absolute_net_return_difference':workflow_return_error,
            'max_asset_return_reconstruction_error':return_error,
            'diagnostic_reconstruction_errors':diagnostic_errors,
            'spy_final_equity':float(spy_equity.iloc[-1]),
            'spy_cumulative_pnl':float(spy_equity.iloc[-1]-CAPITAL),
        },
        'llm_archive':{
            'responses_checked':len(llm), 'historical_responses':len(historic),
            'historical_dates':int(historic['date'].nunique()),
            'synthetic_stress_responses_excluded_from_historical_pnl':len(llm)-len(historic),
            'max_absolute_historical_net_return_reporting_error':float(historic['net_return_reporting_error'].abs().max()),
            'max_absolute_historical_implied_pnl_reporting_error_units':float(historic['implied_pnl_reporting_error_units'].abs().max()),
            'max_absolute_all_case_cost_reporting_error':float(llm['transaction_cost_reporting_error'].abs().max()),
            'historical_compounded_llm_pnl':None,
            'reason':'Ten sampled dates are not a continuous monthly return series; no LLM-driven positions exist.',
        },
        'limitations':[
            'Simulated portfolio PnL, not broker-realized PnL.',
            'API, infrastructure and human review costs excluded; the archive has no billed cost amount.',
            'No PnL improvement can be attributed to a model that does not alter the positions.',
            'Historical LLM source dates are reconstructed with the archived case configuration and current evaluation code; original prompt/data hashes were not stored.',
            'Matching structured returns does not validate narrative recommendations or citations.',
        ],
        'source_sha256':hashes, 'checks':checks, 'all_checks_passed':all(checks.values()),
    }
    monthly.to_csv(RESULTS/'pnl_comparison_monthly.csv', index_label='date')
    llm.to_csv(RESULTS/'pnl_llm_archive_checks.csv', index=False)
    (RESULTS/'pnl_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    r=report['results']; l=report['llm_archive']
    text=f'''# Vérification du PnL simple et du workflow agentique

Période : {report['start']} à {report['end']} ; {report['months']} mois.
Capital initial : 100 unités. Coût : 10 pb par unité de turnover « one-way ».

« Simple » désigne le portefeuille équipondéré VLUE, MTUM, QUAL et USMV, sans synthèse LLM.
Le workflow agentique restitue ce même portefeuille : il ne crée aucune position supplémentaire.
SPY est un benchmark distinct.

| Série | Valeur finale | PnL cumulé, unités |
|---|---:|---:|
| Multifactoriel simple | {r['simple_final_equity']:.2f} | {r['simple_cumulative_pnl']:+.2f} |
| Même portefeuille, valeurs transmises par le workflow | {r['workflow_final_equity']:.2f} | {r['workflow_cumulative_pnl']:+.2f} |
| SPY, benchmark | {r['spy_final_equity']:.2f} | {r['spy_cumulative_pnl']:+.2f} |

Écart final du PnL workflow − simple : {r['final_pnl_difference']:.3e} unité.
Écart absolu maximal des capitaux : {r['max_absolute_equity_difference']:.3e} unité.
Écart maximal des rendements nets : {r['max_absolute_net_return_difference']:.3e} en fraction.
Ces écarts numériques sont évalués à une tolérance absolue de 1e-12 pour les rendements et de 1e-9 pour les capitaux.

## Méthode

Les rendements sont recalculés depuis les cours mensuels archivés. Le rendement brut,
les poids après rendement, le turnover, le coût et le rendement net sont recomposés
indépendamment, puis confrontés aux diagnostics existants. Le véritable graphe
LangGraph est exécuté pour chaque mois, avec arrêt avant le seul nœud de persistance
pour préserver les notes archivées. Ses valeurs, poids cibles et permissions sont contrôlés.
Les deux traces déterministes archivées sont également comparées aux traces rejouées.

Pour chaque mois : capital final = capital précédent × (1 + rendement net).
PnL mensuel = capital précédent × rendement net ; PnL cumulé = capital final − 100.
Le portefeuille est déjà investi au départ, conformément au backtest du mémoire.

## Vérification des réponses DeepSeek

{l['historical_responses']} réponses historiques couvrent {l['historical_dates']} dates distinctes.
Écart maximal entre rendement net cité et source : {l['max_absolute_historical_net_return_reporting_error']:.3e}.
Erreur monétaire implicite maximale sur ces dates, au capital du backtest :
{l['max_absolute_historical_implied_pnl_reporting_error_units']:.3e} unité.
Il s'agit d'une vérification de restitution, pas d'un PnL effectivement créé par le modèle.
Les {l['synthetic_stress_responses_excluded_from_historical_pnl']} réponses au stress synthétique
sont vérifiées séparément et exclues du PnL historique. Le cas qualité bloqué n'est pas
converti artificiellement en rendement nul. Les dates LLM espacées ne sont pas chaînées
pour fabriquer une courbe de capital.

## Conclusion et portée

Les deux chemins restituent le même PnL financier à la précision numérique vérifiée.
Cette égalité est attendue puisque l'agent ne décide pas de l'allocation. Elle établit
la cohérence de la transmission des résultats, sans démontrer un avantage financier du LLM.
Les coûts API, d'infrastructure et de revue humaine ne sont pas inclus. Aucun montant
facturé n'est disponible dans l'archive ; une rentabilité économique après ces frais
ne peut donc pas être calculée ici. Le PnL reste celui d'une simulation.
L'accord des chiffres structurés ne valide pas le texte libre ni les citations.
Les dates sources LLM sont reconstituées avec la configuration archivée et le code
actuel ; l'expérience initiale ne conservait pas les empreintes des prompts et données.

Reproduction : `experience/.venv/bin/python verify_agentic_pnl.py`.
Aucun appel API et aucune modification des données, notes ou résultats d'origine.
Détails : `pnl_comparison_monthly.csv`, `pnl_llm_archive_checks.csv`, `pnl_verification.json`.
'''
    (RESULTS/'pnl_verification.md').write_text(text)
    print(json.dumps({k:report[k] for k in ('months','results','llm_archive','all_checks_passed')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
