# Résultats analysés — allocation décidée par DeepSeek

**L’agent modifie effectivement le portefeuille, mais sous-performe les deux références sur cette expérience.**

Mai 2021–décembre 2025, capital initial 100, frais de transaction de 10 pb appliqués au turnover. Décision mensuelle ; exécution à la clôture suivante.

| Politique | Capital final | PnL net de transaction | Rendement annualisé | Sharpe (RF = 0) | Drawdown maximum |
|---|---:|---:|---:|---:|---:|
| Équipondération | 153.55 | +53.55 | 9.62% | 0.667 | -24.20% |
| Inverse volatilité | 152.26 | +52.26 | 9.42% | 0.674 | -23.44% |
| DeepSeek avec contrôles | 148.39 | +48.39 | 8.82% | 0.645 | -23.50% |

La richesse finale de DeepSeek est inférieure de 5.17 unités à celle de l’équipondération, soit 0.80 point de rendement annualisé en moins. Sa volatilité est plus faible (14.87% contre 15.66%), mais son Sharpe est également inférieur. La règle inverse volatilité obtient un meilleur rendement et un drawdown légèrement moins profond que DeepSeek.

## Ce que fait réellement le contrôle des décisions

Les 56 réponses de la campagne ont été reçues et décodées en JSON. Seulement **22/56 propositions (39.3%) sont exécutées** ; 34 conduisent à conserver les quantités existantes. Ce PnL évalue donc le système complet « propositions LLM + validation + conservation en cas de rejet ». Il ne représente pas un portefeuille exécutant aveuglément chaque proposition.

| Motif de rejet | Occurrences |
|---|---:|
| Turnover demandé supérieur à 10 % | 31 |
| Turnover réel après frais supérieur à 10 % | 2 |
| Somme des poids différente de 1 | 1 |
| Poids hors limites | 1 |
| Référence à un champ absent | 1 |

Plusieurs motifs peuvent concerner une même réponse. Le 30 avril 2021, l’agent propose 40 % VLUE, 20 % MTUM, 30 % QUAL et 10 % USMV depuis 25 % partout : le turnover calculé est **20 %**, alors que son texte affirme respecter la limite de 10 %. Le validateur rejette cette proposition.

## Vérifications et reproductibilité

- 46 tests réussis, dont 23 tests ajoutés pour cette extension.
- Replay des 56 décisions sans appel API : fichiers de valeurs, métriques et décisions identiques octet par octet.
- 3519 valorisations quotidiennes vérifiées indépendamment à partir des positions et des prix.
- Erreur maximale de comptabilité : 5.68e-14 unité, compatible avec l’arrondi flottant.
- 31 fichiers antérieurs contrôlés par SHA-256 : tous inchangés.
- 56 appels de campagne et 4 tentatives techniques préalables, soit 60 tentatives au total.
- 54,639 jetons rapportés pour la campagne ; 11,375 pour les trois réponses techniques reçues. L’usage de la quatrième tentative interrompue est inconnu.

## Interprétation pour le mémoire

Cette extension montre qu’une décision LLM peut être transformée en positions et en PnL par un moteur déterministe, et que les contrôles bloquent des propositions incorrectes. Elle ne montre pas de supériorité financière de l’agent. Le taux de rejet constitue un résultat expérimental à part entière.

La suite logique serait de comparer, dans une nouvelle expérience annoncée à l’avance, un LLM choisissant parmi des allocations déjà admissibles et une règle déterministe utilisant les mêmes indicateurs. Cela permettrait d’évaluer le choix économique en réduisant les erreurs de calcul de contraintes. Aucune variante n’a été ajustée ou relancée ici pour améliorer les performances après observation du résultat.

Le modèle actuel peut connaître les événements historiques par son préentraînement ; ce run reste exploratoire. Les coûts API et humains ne sont pas déduits du PnL. Une seule trajectoire ne fournit ni significativité statistique ni estimation fiable des performances futures. Le [protocole complet](PROTOCOLE.md) précise les autres conventions.

![Valeur et drawdown](deepseek/pnl_comparison.png)

Les [rendements par année](yearly_returns.csv), les [contrôles chiffrés](AUDIT.json) et les [décisions détaillées](deepseek/decisions.json) sont disponibles avec les résultats. L’année 2021 couvre seulement mai–décembre.
