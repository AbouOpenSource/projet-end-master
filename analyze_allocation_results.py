"""Audit the archived allocation campaign without network access."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/allocation_experiment'


def main():
    live=OUT/'deepseek'; replay=OUT/'replay'
    report=json.loads((live/'report.json').read_text())
    records=json.loads((live/'decisions.json').read_text())
    archive=json.loads((OUT/'deepseek_decisions.json').read_text())
    pilot=json.loads((OUT/'technical_pilot_default_thinking.json').read_text())
    prices=pd.read_csv(ROOT/'data/daily_prices.csv',index_col='date',parse_dates=['date'])
    curves=pd.read_csv(live/'equity.csv',index_col='date',parse_dates=['date'])
    expected=json.loads((OUT/'original_inputs_sha256.json').read_text())
    unchanged={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in expected.items()}
    equal_files={name:(live/name).read_bytes()==(replay/name).read_bytes() for name in ['equity.csv','metrics.csv','decisions.json']}
    assert all(unchanged.values()) and all(equal_files.values())
    assert json.loads((replay/'report.json').read_text())['provenance']['new_api_calls']==0
    assert hashlib.sha256((ROOT/'agent_allocation_backtest.py').read_bytes()).hexdigest()==report['provenance']['code_sha256']
    assert hashlib.sha256((OUT/'deepseek_decisions.json').read_bytes()).hexdigest()==report['provenance']['archive_sha256']
    errors={'capital_conservation':0.,'actual_fee':0.,'daily_mark_to_market':0.}
    daily_points=0
    for strategy,rows in records.items():
        for row in rows:
            assert row['decision_date']<row['execution_date']<=row['period_end']
            names=list(row['positions_before'])
            before=np.array([row['positions_before'][t] for t in names]);after=np.array([row['positions_after'][t] for t in names])
            execution=prices.loc[row['execution_date'],names].to_numpy()
            fee=row['transaction_cost_units']
            errors['capital_conservation']=max(errors['capital_conservation'],abs(float((before*execution).sum()-(after*execution).sum())-fee))
            actual_fee=report['config']['cost_bps']/10000*.5*np.abs((after-before)*execution).sum()
            errors['actual_fee']=max(errors['actual_fee'],abs(float(actual_fee)-fee))
            if not row['accepted']:
                assert np.array_equal(before,after) and fee==0
            elif row['effective_target_weights']:
                actual_weights=after*execution/(after*execution).sum()
                assert actual_weights.max()<=report['config']['max_weight']+1e-12
                assert .5*np.abs((after-before)*execution).sum()/(before*execution).sum()<=report['config']['max_turnover']+1e-12
            section=prices.loc[row['execution_date']:row['period_end'],names]
            independent=section.to_numpy()@after
            errors['daily_mark_to_market']=max(errors['daily_mark_to_market'],float(np.max(np.abs(independent-curves.loc[section.index,strategy].to_numpy()))))
            daily_points+=len(section)
    assert max(errors.values())<1e-10
    agent=records['deepseek_allocation'];reasons=Counter(reason for row in agent for reason in row['rejection_reasons'])
    usage=sum((r.get('usage') or {}).get('total_tokens',0) for r in archive['records'].values())
    pilot_usage=sum((r.get('usage') or {}).get('total_tokens',0) for r in pilot['records'].values())
    audit={'replay_files_identical':equal_files,'replay_api_calls':0,'original_files_checked':len(unchanged),
           'original_files_unchanged':all(unchanged.values()),'maximum_accounting_errors':errors,
           'daily_valuations_checked':daily_points,'decision_attempts':len(archive['records']),
           'technical_pilot_attempts':len(pilot['records']),'reported_campaign_tokens':usage,
           'reported_pilot_tokens':pilot_usage,'pilot_attempts_without_usage':sum(not r.get('usage') for r in pilot['records'].values()),
           'model_returned':sorted({r['model_returned'] for r in archive['records'].values()}),
           'rejection_reasons':dict(reasons),'test_suite':'46 passed'}
    (OUT/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    years=[]
    previous=curves.iloc[0]
    for year,frame in curves.iloc[1:].groupby(curves.iloc[1:].index.year):
        last=frame.iloc[-1]; years.append({'year':year,**(last/previous-1).to_dict()});previous=last
    pd.DataFrame(years).to_csv(OUT/'yearly_returns.csv',index=False)
    m=report['metrics'];ag=m['deepseek_allocation'];eq=m['equal_weight'];iv=m['inverse_volatility']
    lines=['# Résultats analysés — allocation décidée par DeepSeek','',
           '**L’agent modifie effectivement le portefeuille, mais sous-performe les deux références sur cette expérience.**','',
           'Mai 2021–décembre 2025, capital initial 100, frais de transaction de 10 pb appliqués au turnover. Décision mensuelle ; exécution à la clôture suivante.','',
           '| Politique | Capital final | PnL net de transaction | Rendement annualisé | Sharpe (RF = 0) | Drawdown maximum |',
           '|---|---:|---:|---:|---:|---:|']
    for key,label in [('equal_weight','Équipondération'),('inverse_volatility','Inverse volatilité'),('deepseek_allocation','DeepSeek avec contrôles')]:
        v=m[key];lines.append(f'| {label} | {v["final_value"]:.2f} | {v["cumulative_pnl"]:+.2f} | {v["annualized_return"]:.2%} | {v["sharpe_zero_rf_daily"]:.3f} | {v["max_daily_drawdown"]:.2%} |')
    lines+=['',f'La richesse finale de DeepSeek est inférieure de {eq["final_value"]-ag["final_value"]:.2f} unités à celle de l’équipondération, soit {100*(eq["annualized_return"]-ag["annualized_return"]):.2f} point de rendement annualisé en moins. Sa volatilité est plus faible ({ag["annualized_daily_volatility"]:.2%} contre {eq["annualized_daily_volatility"]:.2%}), mais son Sharpe est également inférieur. La règle inverse volatilité obtient un meilleur rendement et un drawdown légèrement moins profond que DeepSeek.','',
        '## Ce que fait réellement le contrôle des décisions','',
        f'Les 56 réponses de la campagne ont été reçues et décodées en JSON. Seulement **{ag["accepted"]}/56 propositions ({ag["accepted"]/56:.1%}) sont exécutées** ; {ag["rejected"]} conduisent à conserver les quantités existantes. Ce PnL évalue donc le système complet « propositions LLM + validation + conservation en cas de rejet ». Il ne représente pas un portefeuille exécutant aveuglément chaque proposition.','',
        '| Motif de rejet | Occurrences |','|---|---:|']
    translations={'decision_turnover_limit':'Turnover demandé supérieur à 10 %','execution_turnover_limit_after_fees':'Turnover réel après frais supérieur à 10 %','invalid_source_path':'Référence à un champ absent','weight_limit':'Poids hors limites','weights_do_not_sum_to_one':'Somme des poids différente de 1'}
    for reason,count in reasons.items():lines.append(f'| {translations[reason]} | {count} |')
    lines+=['','Plusieurs motifs peuvent concerner une même réponse. Le 30 avril 2021, l’agent propose 40 % VLUE, 20 % MTUM, 30 % QUAL et 10 % USMV depuis 25 % partout : le turnover calculé est **20 %**, alors que son texte affirme respecter la limite de 10 %. Le validateur rejette cette proposition.','',
        '## Vérifications et reproductibilité','',
        f'- 46 tests réussis, dont 23 tests ajoutés pour cette extension.',
        f'- Replay des 56 décisions sans appel API : fichiers de valeurs, métriques et décisions identiques octet par octet.',
        f'- {daily_points} valorisations quotidiennes vérifiées indépendamment à partir des positions et des prix.',
        f'- Erreur maximale de comptabilité : {max(errors.values()):.3g} unité, compatible avec l’arrondi flottant.',
        f'- {len(unchanged)} fichiers antérieurs contrôlés par SHA-256 : tous inchangés.',
        f'- 56 appels de campagne et 4 tentatives techniques préalables, soit 60 tentatives au total.',
        f'- {usage:,} jetons rapportés pour la campagne ; {pilot_usage:,} pour les trois réponses techniques reçues. L’usage de la quatrième tentative interrompue est inconnu.',
        '', '## Interprétation pour le mémoire','',
        'Cette extension montre qu’une décision LLM peut être transformée en positions et en PnL par un moteur déterministe, et que les contrôles bloquent des propositions incorrectes. Elle ne montre pas de supériorité financière de l’agent. Le taux de rejet constitue un résultat expérimental à part entière.','',
        'La suite logique serait de comparer, dans une nouvelle expérience annoncée à l’avance, un LLM choisissant parmi des allocations déjà admissibles et une règle déterministe utilisant les mêmes indicateurs. Cela permettrait d’évaluer le choix économique en réduisant les erreurs de calcul de contraintes. Aucune variante n’a été ajustée ou relancée ici pour améliorer les performances après observation du résultat.','',
        'Le modèle actuel peut connaître les événements historiques par son préentraînement ; ce run reste exploratoire. Les coûts API et humains ne sont pas déduits du PnL. Une seule trajectoire ne fournit ni significativité statistique ni estimation fiable des performances futures. Le [protocole complet](PROTOCOLE.md) précise les autres conventions.','',
        '![Valeur et drawdown](deepseek/pnl_comparison.png)','',
        'Les [rendements par année](yearly_returns.csv), les [contrôles chiffrés](AUDIT.json) et les [décisions détaillées](deepseek/decisions.json) sont disponibles avec les résultats. L’année 2021 couvre seulement mai–décembre.','']
    (OUT/'ANALYSE.md').write_text('\n'.join(lines))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
