# Protocole de l’expérience d’allocation

L’objectif est de mesurer l’effet des poids proposés par DeepSeek sur un portefeuille simulé. Les calculs, contraintes et transactions sont effectués par Python. Le workflow de restitution du mémoire et ses résultats sont conservés.

## Période et budget

La campagne comparative porte sur **mai 2021 à décembre 2025**, soit 56 décisions mensuelles. Les portefeuilles commencent avec 100 unités réparties à 25 % entre VLUE, MTUM, QUAL et USMV à la clôture du 30 avril 2021.

Un essai technique initial a consommé quatre tentatives : trois réponses sans JSON, arrêtées à 3 000 jetons, et une requête interrompue. Les réponses reçues indiquent que les 3 000 jetons ont été utilisés pour le raisonnement. Ces traces sont conservées dans `technical_pilot_default_thinking.json`. La collecte comparative utilise explicitement `thinking: disabled`, sans modification du prompt financier, et reste limitée aux 56 appels disponibles. Le choix de la période résulte de cette contrainte technique et budgétaire, avant examen des performances de la campagne.

Le réglage est documenté par le fournisseur : [DeepSeek — Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/), consulté le 12 septembre 2026. Le modèle demandé est `deepseek-v4-flash`, température 0, sortie JSON, plafond 3 000 jetons, aucun retry automatique. La température n’est pas une garantie de reproductibilité du service ; les réponses archivées assurent la reproductibilité locale.

## Informations accessibles à la décision

À chaque fin de mois, le modèle reçoit seulement :

- les rendements sur 21, 63, 126 et 252 séances pour chaque ETF ;
- la volatilité annualisée des 252 derniers rendements quotidiens et le drawdown sur les 253 derniers cours ;
- les poids réellement détenus à cette date et les contraintes d’allocation.

La fonction d’indicateurs coupe les données à la date de décision. Aucun rendement du mois suivant, cours d’exécution futur, actualité ou résultat final n’est transmis. Un test modifie tous les cours futurs et vérifie que les entrées demeurent identiques. Cela ne supprime pas la connaissance éventuelle d’événements futurs acquise pendant le préentraînement du modèle.

## Décision et exécution

Le JSON contient exactement `decision_date`, `target_weights`, `reason` et `cited_fields`. Les quatre actifs sont obligatoires ; poids finis, positifs ou nuls, somme 1, plafond de 40 % par actif, absence de levier et de liquidités. La tolérance sur la somme est de 10⁻⁹ ; une normalisation dans cette seule tolérance est tracée. Les références doivent désigner des champs présents dans les entrées. Ce contrôle ne valide pas la pertinence économique de la justification ni son exactitude sémantique.

Le turnover à sens unique est limité à 10 % à la décision et revérifié au cours d’exécution. Une proposition invalide ou un dépassement après mouvement des prix laisse les quantités détenues inchangées, sans frais. Il n’y a ni correction discrétionnaire des poids ni nouvelle réponse recherchée après rejet. Le plafond de concentration concerne les poids cibles ; les variations de marché peuvent ensuite faire dériver les poids.

L’exécution intervient à la **première clôture disponible du mois suivant**. Les anciennes positions supportent donc la variation de prix entre observation et exécution. Les frais valent 10 points de base multipliés par la moitié du volume réellement échangé. Ils sont déduits du capital avant calcul des nouvelles quantités. Pour une valeur avant transaction V, des montants détenus N et des poids cibles w, le moteur résout :

`frais = 0,001 × 0,5 × somme(|w × (V − frais) − N|)`.

Les nouvelles positions valent `w × (V − frais)`. Un test vérifie la conservation du capital, le coût effectivement échangé et l’exécution différée. Les unités fractionnaires calculées sur des cours ajustés représentent une comptabilité de rendement total ; ce ne sont pas des quantités à envoyer à un courtier. Il n’y a pas de coût d’entrée initial ni de liquidation finale.

## Références et métriques

Trois politiques suivent exactement le même calendrier, le même capital initial, les mêmes prix, les mêmes contrôles et le même calcul de frais :

1. Équipondération : cible de 25 % par ETF.
2. Inverse volatilité : poids proportionnels à l’inverse de la volatilité passée, plafonnés à 40 %.
3. DeepSeek : poids du JSON validé, ou conservation des positions si rejet.

Les allocations sont mensuelles et les valeurs de portefeuille quotidiennes. Le rendement annualisé utilise le temps calendaire effectivement écoulé ; volatilité et Sharpe utilisent les rendements quotidiens avec facteur 252 et taux sans risque nul. Le drawdown inclut le capital initial. La somme des frais est exprimée en unités de capital ; elle n’est pas l’écart final avec un scénario sans frais, puisque la capitalisation diffère.

Le test hors réseau `offline/` porte sur janvier 2021–décembre 2025 et utilise une règle de momentum pour éprouver le moteur. Ses chiffres ne sont **pas** les résultats de DeepSeek et ne doivent pas être mélangés au tableau comparatif principal.

## Traçabilité et limites

Chaque tentative est écrite sur disque avant l’appel HTTP. L’archive conserve la requête, son empreinte SHA-256, la réponse, le modèle retourné, le statut de fin, les jetons rapportés et la durée. Le replay vérifie l’identité du prompt, des entrées et des paramètres, puis réexécute les transactions sans réseau. Une requête interrompue peut avoir été facturée sans que son usage ait été reçu ; le total des jetons rapportés est alors incomplet.

Cette unique trajectoire rétrospective ne permet pas de conclure à un alpha, à une significativité statistique ou à une généralisation. Les données ajustées ne sont pas archivées dans leur version disponible à chaque date. Spread, impact, fiscalité, coûts API et revue humaine sont exclus du PnL. Les contraintes sont expérimentales et n’ont pas été optimisées sur les résultats. Une validation Python dans cette simulation ne constitue pas une approbation humaine ni une autorisation d’ordre réel.

Le nom demandé à l’API est `deepseek-v4-flash` ; les 56 réponses identifient le modèle retourné comme `deepseek-flash`. Les deux identifiants sont conservés. L’archive ne permet pas d’identifier plus précisément les poids internes ou la version d’entraînement du service.
