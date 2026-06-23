# Analyse du data drift

Étude de la dérive des données de production de l'API de scoring crédit,
réalisée avec Evidently (preset `DataDriftPreset`). Référence : échantillon
stratifié de 10 000 lignes d'entraînement (`data/reference_sample.parquet`).
Production : champ `inputs` extrait de `logs/predictions.jsonl`.

## Protocole

Deux vagues de requêtes ont été envoyées à l'API :

- **Normale** — vraies lignes clients, tirées dans le complément de la
  référence (aucun recouvrement). Proba de défaut moyenne : **0,358**.
- **Dérivée** — mêmes lignes, perturbées selon un scénario de récession
  (revenus ×0,6 ; crédits ×1,4 ; annuités ×1,2 ; scores externes ×0,5).
  Proba de défaut moyenne : **0,697**.

Deux comparaisons ont été produites : référence vs **toute la production**
(1 000 lignes) et référence vs **vague dérivée** seule (500 lignes).

## Résultats par feature

Le drift est concentré, comme attendu, sur les features perturbées. Scores
de drift (distance de Wasserstein normée pour les numériques, Jensen-Shannon
pour les catégorielles) sur la **vague dérivée** :

| Feature | Drift score | Détecté |
|---|---|---|
| EXT_SOURCE_3 | 1,37 | oui |
| EXT_SOURCE_2 | 1,30 | oui |
| EXT_SOURCE_1 | 1,23 | oui |
| AMT_INCOME_TOTAL | 0,65 | oui |
| AMT_CREDIT | 0,61 | oui |
| AMT_ANNUITY | 0,38 | oui |
| DAYS_BIRTH | 0,26 | oui |
| NAME_INCOME_TYPE | 0,25 | oui |
| DAYS_ID_PUBLISH | 0,14 | oui |
| DAYS_REGISTRATION | 0,13 | oui |

Les features **non perturbées** (`OWN_CAR_AGE`, `FLAG_OWN_CAR`,
`CODE_GENDER`, `NAME_CONTRACT_TYPE`, `DAYS_EMPLOYED`, `CNT_FAM_MEMBERS`,
`AMT_GOODS_PRICE`, `CNT_CHILDREN`, `NAME_EDUCATION_TYPE`,
`NAME_FAMILY_STATUS`, `FLAG_OWN_REALTY`) restent « Not Detected ». Evidently
isole donc correctement le signal injecté.

Sur **toute la production** (mélange des deux vagues), les scores sont
mécaniquement divisés par ~2 (ex. EXT_SOURCE_3 : 1,37 → 0,73), puisqu'on
moyenne une moitié stable et une moitié déviante.

## Le paradoxe du verdict global

Point central de cette analyse :

| Comparaison | Colonnes driftées | Part | Verdict « Dataset Drift » |
|---|---|---|---|
| Vague dérivée | 10 / 21 | 0,476 | **NON détecté** |
| Toute la production | 11 / 21 | 0,524 | **détecté** |

C'est contre-intuitif : la vague **la plus déviante** (drift scores jusqu'à
1,37) n'est **pas** flaggée au niveau dataset, alors que le mélange, moins
intense, l'est.

**Explication.** Le verdict global d'Evidently repose sur la **proportion de
colonnes** en dérive (seuil 0,5), et non sur l'**intensité** du drift. Sur la
vague dérivée, le drift est très fort mais concentré sur 10 features — soit
0,476, juste sous la barre des 50 %. Le drapeau global ne se lève donc pas,
alors même que les drift scores individuels sont énormes.

## Conséquences opérationnelles

1. **Ne jamais se fier au seul verdict binaire** « Dataset Drift detected ».
   Un drift à fort impact peut passer sous le seuil de proportion.
2. **Analyser le drift par feature**, et le **pondérer par l'importance** des
   features dans le modèle. Ici, les 3 features les plus driftées
   (`EXT_SOURCE_1/2/3`) sont aussi les **3 plus importantes** du modèle
   (cf. feature importance du P06) : le drift le plus dangereux pour la
   qualité des prédictions est précisément celui que le verdict global masque.
3. **Corréler avec la sortie du modèle.** Le dashboard montre que la proba de
   défaut moyenne double (0,358 → 0,697) sur la vague dérivée : le drift
   d'entrée se propage en drift de prédiction, avec un basculement massif des
   décisions vers le refus.

## Recommandations de surveillance

- Définir des **alertes par feature** sur les variables à forte importance
  (seuils sur les drift scores des `EXT_SOURCE_*`, `AMT_*`), plutôt qu'un
  unique seuil global.
- Suivre en parallèle la **distribution des scores de sortie** (prediction
  drift), indicateur synthétique et sans label de vérité terrain.
- Déclencher une **revue / un réentraînement** lorsque le drift touche les
  features les plus contributives ou lorsque la distribution des scores
  s'écarte durablement de la référence.
