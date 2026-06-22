# P08 — Déployez et monitorez votre modèle de scoring

Mise en production du modèle de scoring crédit développé lors du
[Projet 06 — Initiez-vous au MLOps](https://github.com/saidmo/p06_mlops_1_mlflow)
(LightGBM optimisé avec Optuna, AUC-ROC ~0.79, suivi MLflow).

Le contexte métier est celui de **Prêt à Dépenser**, dont le
département *Crédit Express* doit traiter en quasi temps réel des
demandes de crédit à la consommation.

## Objectifs de ce dépôt

1. Exposer le modèle via une **API REST** (FastAPI), consommée par une
   **interface Streamlit** (scoring + monitoring) — *à venir, Étape 2*.
2. **Conteneuriser** l'application avec Docker — *à venir, Étape 2*.
3. Automatiser tests et build via **CI/CD** GitHub Actions — *à venir,
   Étape 2*.
4. **Journaliser** chaque prédiction (inputs, output, latence) via un
   **logging structuré JSON** et **détecter le data drift** avec
   Evidently — *à venir, Étape 3*.
5. **Optimiser** les performances post-déploiement — *à venir, Étape 4*.

## Structure du dépôt

```
.
├── app/                    code de l'API FastAPI (Étape 2)
├── model/
│   └── model_credit_scoring.pkl   artefact UNIQUE réutilisé du P06
├── features.py             feature engineering partagé train/serving
├── tests/                  tests unitaires pytest (Étape 2)
├── monitoring/             dashboard Streamlit + Evidently + logs JSON (Étape 3)
├── .github/workflows/      pipeline CI/CD
├── Dockerfile
└── requirements*.txt
```

## Artefact modèle

`model/model_credit_scoring.pkl` est un dictionnaire pickle contenant :

- un **`Pipeline` scikit-learn auto-portant** chaînant le calcul des
  ratios métier, l'imputation et l'encodage des variables, puis le
  classifieur LightGBM ;
- les listes de colonnes (`input_cols`, `binary_cols`, `multi_cols`,
  `num_cols`) ;
- le **seuil métier optimal** (0.49) issu de l'optimisation coût
  FN×10 / FP×1 du P06 ;
- l'AUC-ROC de référence et la version du modèle.

## Lancement (à compléter à l'Étape 2)

