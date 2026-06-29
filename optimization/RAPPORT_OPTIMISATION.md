# Rapport d'optimisation des performances (Étape 4)

Mission : « Analysez et optimisez les performances du modèle. Testez des
stratégies d'optimisation (quantification, optimisation de code, hardware)
pour améliorer le temps d'inférence/réponse. »

Ce rapport documente la démarche, les stratégies **testées**, celles
**retenues** et celles **écartées sur preuve**, ainsi que la justification de
la configuration finale.

## 1. Démarche : profiler avant d'optimiser

Le principe directeur de l'étape est de **mesurer avant d'agir**. Le script
`profile_inference.py` décompose une prédiction unitaire en quatre blocs
chronométrés séparément (construction du DataFrame, feature engineering,
prétraitement, arbre LightGBM) et complète l'analyse par un passage
`cProfile` au niveau fonction.

### Résultat fondateur

La structure du coût est stable sur deux architectures CPU différentes
(x86 Windows et Apple Silicon), ce qui prouve qu'elle est **structurelle** et
non liée au matériel. Répartition pour un client « noyau » (cas réaliste en
production : ~20 features fournies, le reste à NaN) :

| Bloc | Part du temps |
|---|---|
| Prétraitement (ColumnTransformer) | ~55 % |
| Feature engineering (9 ratios) | ~25 % |
| Arbre LightGBM | ~10 % |
| Construction du DataFrame | ~8 % |

**Conclusion clé : l'arbre — le « modèle » au sens propre — ne représente
qu'environ 10 % du temps.** Les ~90 % restants sont de l'overhead de
manipulation pandas/sklearn (validation, conversion, encodage), payé
intégralement à chaque appel unitaire car ces bibliothèques sont conçues pour
le traitement par lots. Le `cProfile` le confirme : `isinstance` appelé des
dizaines de millions de fois, `sanitize_array` plus d'un million de fois, le
tout dans les routines de validation de sklearn et de construction de pandas.

Cette découverte oriente toute la stratégie : optimiser le calcul du modèle
(quantification, accélération de l'arbre) serait sans effet ; le levier réel
est l'**overhead du framework** autour du modèle.

## 2. Stratégies testées

### 2.1 Optimisation de code (retenue)

Deux pistes de serving testées via `benchmark_optim.py`, qui mesure la latence
**et** vérifie une non-régression stricte (les probabilités produites doivent
être identiques au pipeline d'origine, tolérance 1e-9).

**`assume_finite` — écartée.** L'hypothèse était que la vérification de
finitude de sklearn (contrôle des NaN/inf répété sur 658 colonnes) pesait
lourd. La mesure l'a infirmée : gain de +1,8 % (x86) à −0,6 % (Apple Silicon),
soit du bruit. Hypothèse raisonnable, rejetée par la mesure.

**Template pré-alloué — retenue et intégrée.** La construction du DataFrame
d'entrée (`pd.DataFrame([...]).reindex` sur 658 colonnes) recrée et valide
chaque colonne à chaque appel. On la remplace par la **copie d'un template
pré-alloué** (une ligne, 658 colonnes, dtypes corrects), construit une seule
fois au démarrage, puis copié et rempli à chaque requête. Gain mesuré :
**+5,8 % (x86) à +10,1 % (Apple Silicon)**, avec **zéro régression** (probas
strictement identiques). Cette optimisation est désormais **intégrée dans
`app/model.py`** (fonction `_to_frame`), donc active en production.

### 2.2 Conversion ONNX Runtime (testée puis rejetée sur preuve)

L'idée : convertir le pipeline (`ColumnTransformer` + `LGBMClassifier`) en un
graphe ONNX exécuté en C++ par ONNX Runtime, pour éliminer l'overhead Python.
Note : ONNX n'a d'intérêt que si l'on convertit **tout le prétraitement** —
convertir le seul arbre (~10 %) serait inutile, voire contre-productif.

**La conversion a fini par aboutir**, mais au prix de cinq obstacles
successifs (investigation menée séparément, non conservée dans le dépôt par
souci de simplicité) :

1. le `FunctionTransformer` (9 ratios) n'est pas convertible → exclu du graphe,
   les ratios sont calculés en numpy côté serving ;
2. `SimpleImputer` sur colonnes chaînes avec valeur manquante `float` →
   `NotImplementedError` de skl2onnx ;
3. après contournement, le `LabelEncoder` généré reçoit `default_string=None`,
   rejeté par onnx → on **retire l'imputation du graphe** (faite en Python via
   les imputers ajustés, donc exacte) ;
4. opset `ai.onnx.ml` 5 non supporté → forcé à 3 (`target_opset`) ;
5. skl2onnx **renomme** les colonnes à caractères spéciaux (espaces, /, ,) →
   appariement des entrées par **position** plutôt que par nom.

**La vérification de non-régression a disqualifié ONNX.**
Comparaison sur 500 clients, sklearn vs ONNX :

| Population | Écart proba max | Décisions qui basculent (seuil 0,49) |
|---|---|---|
| Clients complets | 0,108 | 10 / 500 (2 %) |
| Clients noyau | 0,048 | 1 / 500 (0,2 %) |

L'écart est très supérieur à un simple arrondi (qui serait ~1e-6). La cause est
une **limitation fondamentale et documentée** : LightGBM représente les seuils
de ses arbres en double précision (float64), alors qu'ONNX-ML n'utilise que des
float32 pour l'opérateur `TreeEnsemble`. Les seuils (et les valeurs d'entrée)
sont tronqués, ce qui fait basculer du mauvais côté les clients proches d'une
frontière de décision. Détail confirmant le diagnostic : les **clients complets
dérivent plus** que les clients noyau, car ils contiennent davantage de vraies
valeurs numériques de grande magnitude sensibles à la troncature float32.

**Décision : ONNX est rejeté.** Pour un modèle de scoring crédit, une décision
d'octroi qui change à cause d'un artefact numérique de conversion — et non de
l'intention du modèle — est un problème de gouvernance et de reproductibilité
inacceptable, même à 0,2 %. La mission exige explicitement l'absence de
régression de précision.

Ce résultat n'est pas un échec mais l'aboutissement attendu d'une démarche
rigoureuse : la vérification de non-régression a joué son rôle de garde-fou et
a intercepté une régression qu'une approche naïve (« la conversion réussit,
donc je déploie ») aurait laissée passer en production.

### 2.3 Quantification (inadaptée)

La quantification (réduction de précision des poids, typiquement float32 → int8)
est une technique d'optimisation des **réseaux de neurones**. Elle n'a pas de
sens sur un modèle d'arbres de gradient comme LightGBM, dont l'inférence ne
repose pas sur des produits matriciels en virgule flottante mais sur des
comparaisons de seuils. Ironiquement, le problème rencontré avec ONNX est
*déjà* un effet de réduction de précision (float64 → float32) qui a dégradé les
décisions : pousser plus loin dans cette direction serait contre-productif.

### 2.4 Matériel (mesure)

La démarche multi-machines a fourni une mesure « hardware » involontaire mais
parlante : le **même code s'exécute environ 6 à 8 fois plus vite sur Apple
Silicon (M-series) que sur le poste x86** (latence unitaire noyau de l'ordre de
5 ms contre 25–35 ms). En production, le choix de la machine hôte du conteneur
Docker est donc un levier de performance réel et mesurable.

### 2.5 Réécriture numpy du prétraitement (identifiée, non retenue)

Un gain supplémentaire serait théoriquement possible en réécrivant le
prétraitement en numpy pur (calcul des ratios, imputation et encodage à la main
en réutilisant les paramètres ajustés), afin de contourner l'overhead du
`ColumnTransformer`. Cette piste est **identifiée mais non retenue** : elle
présente exactement le même risque que la conversion ONNX — toute divergence
dans la reproduction manuelle de l'imputation, de l'encodage ou de l'ordre des
colonnes modifierait les prédictions, ce qui est incompatible avec un usage de
scoring crédit. Le gain potentiel ne justifie pas ce risque, d'autant que la
latence actuelle est déjà confortablement dans les exigences du « quasi temps
réel ».

## 3. Configuration finale retenue

L'optimisation **template** est intégrée au serving (`app/model.py`), validée
sans régression (proba de référence inchangée : 0,7765756075) et couverte par
les 23 tests de la suite (couverture 97 %). Les autres pistes sont écartées
avec justification : `assume_finite` (gain nul), ONNX (régression de précision),
quantification (inadaptée), réécriture numpy (risque).

Le modèle de production reste donc le pipeline **scikit-learn d'origine**,
intact, précédé du seul `_to_frame` optimisé.

## 4. Reproductibilité

Toute la pile numérique est épinglée dans `requirements.txt`
(numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, scikit-learn 1.8.0, lightgbm 4.6.0)
pour garantir des prédictions identiques entre les postes Windows, macOS et
l'image Docker. L'investigation ONNX ayant été écartée, ses dépendances ne
font pas partie de l'environnement du projet.
