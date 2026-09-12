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
- Turnover : convention « one-way », soit la moitié de la somme des variations
  absolues de poids après rendement et avant rééquilibrage.
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
.venv/bin/python -m pytest -q
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
cp .env.example .env
# renseigner DEEPSEEK_API_KEY dans .env
.venv/bin/python deepseek_agent_workflow.py --all-last-two
```

Le modele et l'URL peuvent etre ajustes sans modifier le code :

```bash
# DEEPSEEK_MODEL et DEEPSEEK_BASE_URL peuvent aussi etre definis dans .env
```

Le script produit `deepseek_decision_trace_YYYY-MM-DD.json`,
`deepseek_decision_note_YYYY-MM-DD.md` et `deepseek_comparison.csv`. La comparaison
mesure la fidelite numerique, la completude, le respect des permissions et la
divulgation des limites. La cle API n'est jamais ecrite dans les fichiers de sortie.



## Evaluation DeepSeek

Le benchmark local evalue la fidelite numerique, les permissions, la divulgation des risques, le routage des cas bloques et la stabilite des sorties. Il ne mesure pas une capacite a predire les rendements.

Preparer les cas sans appeler l API :

```bash
.venv/bin/python evaluate_deepseek.py --dry-run
```

Lancer une evaluation rapide avec dix cas valides, un stress de turnover et un cas qualite bloque :

```bash
# renseigner DEEPSEEK_API_KEY dans .env
.venv/bin/python evaluate_deepseek.py --valid-cases 10 --repeats 1
```

Pour mesurer la stabilite, repeter chaque cas trois fois :

```bash
.venv/bin/python evaluate_deepseek.py --valid-cases 10 --repeats 3
```

Les sorties sont `results/deepseek_evaluation.json` et `results/deepseek_evaluation.csv`. Le score est accepte seulement si les chiffres sont fideles, les permissions respectees, le contrat JSON complet et les risques signales.

## Rapport LaTeX

Le rapport complet et ses figures sont dans `latex/`. Pour reconstruire le PDF :

```bash
cd latex
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error rapport_agentique_multi.tex
```


## Comparaison des notes et contre-exemples hors réseau

```bash
experience/.venv/bin/python analyze_note_comparison.py
```

Le script retient la dernière date commune aux notes déterministes et aux cas historiques archivés, puis la première répétition DeepSeek. Il conserve aussi les deux autres répétitions, compare les textes et applique trois altérations indépendantes à un seul champ ainsi que deux témoins de détection. Aucun appel API n'est effectué et les résultats originaux restent inchangés.

Les sorties sont `results/note_comparison.md` et `results/complementary_analysis.json`. Ce dernier contient les textes, transformations, scores et empreintes des sources. Les contre-exemples sont construits : ils ne mesurent pas la fréquence des erreurs du modèle. La discussion est intégrée au chapitre 5 du mémoire.


## Vérification du PnL simple et du workflow agentique

```bash
experience/.venv/bin/python verify_agentic_pnl.py
```

Ce contrôle hors réseau recalcule le portefeuille à 10 pb depuis les cours mensuels
archivés, rejoue le workflow sur chaque mois sans réécrire ses notes et vérifie les
rendements cités dans les réponses DeepSeek historiques. Il compare un même
portefeuille avec et sans restitution agentique ; il ne mesure pas une stratégie
d'allocation décidée par le LLM. Les dates LLM espacées et le stress synthétique ne
sont pas assemblés en un historique de PnL.

Les sorties sont `results/pnl_verification.md`, `results/pnl_verification.json`,
`results/pnl_comparison_monthly.csv` et `results/pnl_llm_archive_checks.csv`.
Les coûts API, d'infrastructure et de revue humaine sont exclus.

## Portefeuille proposé par l’agent : expérience d’allocation

`agent_allocation_backtest.py` permet au LLM de proposer des poids entre VLUE,
MTUM, QUAL et USMV. Python valide le contrat JSON, les actifs, les poids et le
turnover, puis simule l’exécution à la clôture de la séance suivante. Une décision
invalide conserve les positions existantes. Les frais sont financés par le
portefeuille. Deux références passent par le même moteur : équipondération et
allocation inverse de la volatilité. Il n’existe aucune connexion à un courtier.

Test du moteur sans appel réseau (la fixture est une règle, pas un LLM) :

```bash
experience/.venv/bin/python agent_allocation_backtest.py --mode offline --output results/allocation_experiment/offline
experience/.venv/bin/python -m pytest -q
```

Campagne DeepSeek réalisée, bornée à 56 décisions de mai 2021 à décembre 2025 :

```bash
experience/.venv/bin/python agent_allocation_backtest.py --mode live --start 2021-05-01 --archive results/allocation_experiment/deepseek_decisions.json --max-api-calls 56 --max-output-tokens 3000 --output results/allocation_experiment/deepseek
```

Le mode `live` réutilise les dates déjà archivées. Le budget porte sur le nombre
**total de tentatives de cette archive**, y compris les erreurs et interruptions.
Une erreur réseau arrête la collecte ; une reprise conserve cette tentative comme
une absence de proposition. Aucun nouvel appel n’est fait pour remplacer une
proposition rejetée. La campagne utilise explicitement `thinking: disabled`.
Quatre tentatives techniques antérieures, conservées séparément, portent le
budget global de cette exécution à 60 appels au maximum.

Reproduction exacte des décisions archivées, sans clé API ni réseau :

```bash
experience/.venv/bin/python agent_allocation_backtest.py --mode replay --start 2021-05-01 --archive results/allocation_experiment/deepseek_decisions.json --max-output-tokens 3000 --output results/allocation_experiment/replay
```

L’empreinte du prompt, des indicateurs, des poids courants et des paramètres doit
correspondre à chaque requête archivée. Changer les coûts ou le prompt interdit
la réutilisation silencieuse d’une décision. Les résultats, courbes quotidiennes,
poids et justifications sont dans `results/allocation_experiment/`.
Voir [le protocole](results/allocation_experiment/PROTOCOLE.md) et
[les résultats](results/allocation_experiment/deepseek/RESULTATS.md).

Cette expérience rétrospective ne démontre pas une capacité prédictive hors
échantillon du LLM : son préentraînement peut inclure les événements étudiés.
Les résultats du mémoire initial restent des résultats d’un autre protocole.

Après le run et son replay, produire l’analyse et vérifier indépendamment les
positions, frais et valeurs quotidiennes :

```bash
experience/.venv/bin/python analyze_allocation_results.py
```

Ce contrôle écrit [l’analyse détaillée](results/allocation_experiment/ANALYSE.md),
`AUDIT.json` et `yearly_returns.csv`. Les empreintes des fichiers antérieurs
correspondent à l’état précédant cette extension.
