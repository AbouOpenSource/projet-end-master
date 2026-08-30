# projet-end-master

## Expérience multifactorielle minimale

Ce sous-projet constitue le prototype empirique du rapport sur l'approche agentique
appliquée au trading multifactoriel.

## Protocole retenu

- Univers : ETF factoriels américains `VLUE`, `MTUM`, `QUAL`, `USMV`.
- Benchmark : `SPY`.
- Pondération : 25 % par facteur, rééquilibrage mensuel.
- Période : janvier 2015 à décembre 2025.
- Évaluation hors échantillon : janvier 2021 à décembre 2025.
- Prix : cours ajustés téléchargés via `yfinance`.
- Coûts : 0, 10 et 30 points de base par unité de turnover.
- Taux sans risque : 0 % pour le ratio de Sharpe.

Les ETF sont utilisés comme des portefeuilles factoriels observables. Le backtest
teste le socle quantitatif, pas la capacité d'un LLM à prévoir les rendements.

## Exécution

Depuis la racine du projet :

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backtest.py
.venv/bin/python agent_workflow.py
```

Le premier script écrit les données, résultats et figures dans `data`,
`results` et `figures`. Le second exécute le graphe
LangGraph suivant :

`load_inputs -> quality_gate -> rebalance_analyst -> risk_gate -> supervisor -> persist`.

Le workflow bloque si les données présentent des doublons, valeurs manquantes ou prix
non positifs. Il ne modifie jamais les pondérations, ne crée aucun ordre et exige une
validation humaine. Les traces individuelles sont archivées dans
`results/agent_decision_trace_YYYY-MM-DD.json`.

## Fichiers de sortie

- `metrics.csv` : métriques sur l'échantillon complet.
- `period_metrics.csv` : métriques complètes et hors échantillon.
- `quality_report.json` : contrôles de cohérence et limites connues.
- `agent_decision_notes.md` : notes générées par le workflow.
- `agent_decision_trace_*.json` : permissions, contrôles et sorties structurées.

## Limites

- Les ETF sont américains et ne représentent pas directement un univers actions Europe.
- Le survivorship bias n'est pas éliminé par cette expérience ETF.
- Les frais internes des ETF sont reflétés dans leurs prix mais ne sont pas isolés.
- Le slippage et l'impact de marché ne sont pas modélisés séparément ; les scénarios
  10 et 30 pb sont des proxys de coûts tout compris.
- Le workflow de reference est deterministe et reproductible ; le workflow DeepSeek
  est une couche de synthese optionnelle, active uniquement lorsque la cle API est
  fournie.


## Test DeepSeek de bout en bout

Une seconde version du graphe ajoute une synthese DeepSeek apres les controles
deterministes. Le modele recoit les rendements, le turnover, le cout et les alertes
deja calcules ; il ne recoit aucune permission de modifier les poids ou d'envoyer un
ordre. Sa reponse JSON est comparee a la reference deterministe avant archivage.

Installer la dependance puis definir la cle uniquement dans l'environnement :

```bash
.venv/bin/pip install -r requirements.txt
export DEEPSEEK_API_KEY='votre_cle'
.venv/bin/python deepseek_agent_workflow.py --all-last-two
```

Le modele et l'URL peuvent etre ajustes sans modifier le code :

```bash
export DEEPSEEK_MODEL='deepseek-v4-flash'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
```

Le script produit `deepseek_decision_trace_YYYY-MM-DD.json`,
`deepseek_decision_note_YYYY-MM-DD.md` et `deepseek_comparison.csv`. La comparaison
mesure la fidelite numerique, la completude, le respect des permissions et la
divulgation des limites. La cle API n'est jamais ecrite dans les fichiers de sortie.



## Rapport LaTeX

Le rapport complet et ses figures sont dans `latex/`. Pour reconstruire le PDF :

```bash
cd latex
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error rapport_agentique_multi.tex
```
