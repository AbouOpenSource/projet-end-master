# Vérification du PnL simple et du workflow agentique

Période : 2015-02-28 à 2025-12-31 ; 131 mois.
Capital initial : 100 unités. Coût : 10 pb par unité de turnover « one-way ».

« Simple » désigne le portefeuille équipondéré VLUE, MTUM, QUAL et USMV, sans synthèse LLM.
Le workflow agentique restitue ce même portefeuille : il ne crée aucune position supplémentaire.
SPY est un benchmark distinct.

| Série | Valeur finale | PnL cumulé, unités |
|---|---:|---:|
| Multifactoriel simple | 345.10 | +245.10 |
| Même portefeuille, valeurs transmises par le workflow | 345.10 | +245.10 |
| SPY, benchmark | 411.73 | +311.73 |

Écart final du PnL workflow − simple : -1.023e-12 unité.
Écart absolu maximal des capitaux : 1.023e-12 unité.
Écart maximal des rendements nets : 2.012e-16 en fraction.
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

30 réponses historiques couvrent 10 dates distinctes.
Écart maximal entre rendement net cité et source : 1.405e-07.
Erreur monétaire implicite maximale sur ces dates, au capital du backtest :
1.405e-05 unité.
Il s'agit d'une vérification de restitution, pas d'un PnL effectivement créé par le modèle.
Les 3 réponses au stress synthétique
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
