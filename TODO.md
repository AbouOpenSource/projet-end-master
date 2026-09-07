# TODO - Améliorations du rapport

État mis à jour le 30 août 2026.

## Réalisé

- [x] Construire un mini-backtest multifactoriel reproductible.
- [x] Comparer quatre sleeves factoriels à SPY sur 2015--2025.
- [x] Tester les scénarios de coûts de 0, 10 et 30 points de base.
- [x] Produire rendement annualisé, volatilité, Sharpe, drawdown, valeur finale et turnover.
- [x] Ajouter un rapport qualité des données : doublons, valeurs manquantes, prix non positifs, gaps calendaires et mouvements extrêmes.
- [x] Ajouter une période hors échantillon 2021--2025.
- [x] Ajouter un workflow LangGraph déterministe avec garde qualité, garde risque, journal JSON et validation humaine obligatoire.
- [x] Générer les notes de décision à partir des sorties structurées sans accès aux ordres.
- [x] Ajouter des tests unitaires, des tests de parcours LangGraph et des tests de non-régression hors réseau.
- [x] Integrer les resultats experimentaux au resume, a l abstract et aux objectifs.
- [x] Ajouter une grille de scoring 0-3 et expliciter les statuts des solutions.
- [x] Ajouter les schemas d architecture, la boucle de rejet, le journal JSON et le monitoring agentique.
- [x] Ajouter le tableau des risques, les trois niveaux d explicabilite et l evaluation des deux demonstrations.
- [x] Reorganiser la conclusion autour de la problematique, des resultats, limites et perspectives.
- [x] Ajouter une revue réglementaire datée et une matrice de due diligence GitHub.
- [x] Nettoyer les blocs du modèle initial dans les sources LaTeX.
- [x] Corriger les principaux débordements de mise en page.

## À compléter avant un usage institutionnel

- [ ] Remplacer les ETF par un univers historique titres avec données point-in-time pour réduire le survivorship bias.
- [ ] Vérifier séparément dividendes, splits, corporate actions et délais de publication auprès d'un fournisseur institutionnel.
- [ ] Modéliser le slippage, l'impact de marché, les contraintes de liquidité et les limites d'exposition.
- [ ] Ajouter des tests de stress, une validation indépendante et une vraie évaluation de la couche LLM.
- [ ] Compléter la due diligence par un audit juridique des licences, des dépendances et de la sécurité.
- [ ] Faire valider toute trajectoire de production par les fonctions risque, conformité, juridique et IT.
