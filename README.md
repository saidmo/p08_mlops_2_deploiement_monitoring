# P08 — Déployez et monitorez votre modèle de scoring

Mise en production du modèle de scoring crédit développé lors du
[Projet 06 — Initiez-vous au MLOps](https://github.com/saidmo/p06_mlops_1_mlflow)
(LightGBM optimisé avec Optuna, AUC-ROC ~0.79, suivi MLflow).

Le contexte métier — fictif — est celui de **Prêt à Dépenser**, dont le
département *Crédit Express* doit traiter en quasi temps réel des
demandes de crédit à la consommation.

## Objectifs de ce dépôt

1. Exposer le modèle via une **API REST** (FastAPI) — ✅ Étape 2.
2. **Conteneuriser** l'application avec Docker — ✅ Étape 2.
3. Automatiser tests et build via **CI/CD** GitHub Actions — ✅ Étape 2.
4. **Journaliser** chaque prédiction (inputs, output, latence) via un
   **logging structuré JSON** et **détecter le data drift** avec
   Evidently, visualisé dans un dashboard **Streamlit** — *Étape 3*.
5. **Optimiser** les performances post-déploiement — *Étape 4*.

## Structure du dépôt

```
.
├── app/                    code de l'API FastAPI
│   ├── main.py             endpoints, middleware de latence, logging JSON
│   ├── model.py            chargement unique du modèle + scoring
│   └── schemas.py          validation Pydantic (entrée / sortie)
├── model/
│   └── model_credit_scoring.pkl   artefact UNIQUE réutilisé du P06
├── features.py             feature engineering partagé train/serving
├── tests/                  tests unitaires pytest (TestClient)
├── monitoring/             dashboard Streamlit + Evidently + logs JSON (Étape 3)
├── logs/
│   └── predictions.jsonl   journal des prédictions (généré au runtime)
├── .github/workflows/      pipeline CI/CD
├── Dockerfile
├── pytest.ini
├── requirements.txt        dépendances runtime de l'API
└── requirements-dev.txt    dépendances de test / monitoring
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

## Installation

Prérequis : Python 3.12 (ou 3.11+), et Git.

```bash
# Cloner le dépôt puis, à la racine :
python -m venv .venv

# Activer l'environnement
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # Linux / macOS

# Dépendances : runtime seul, ou dev/test/monitoring
pip install -r requirements.txt        # pour lancer l'API
pip install -r requirements-dev.txt    # pour tests + monitoring
```

> Les versions de `scikit-learn` et `lightgbm` sont épinglées dans
> `requirements.txt` pour garantir la compatibilité du pickle du modèle.

## Lancer l'API

```bash
uvicorn app.main:app --reload --port 8800
```

- Documentation interactive (Swagger) : <http://localhost:8800/docs>
- L'API charge le modèle **une seule fois** au démarrage.

### Endpoints

| Méthode | Route      | Description                                            |
|---------|------------|--------------------------------------------------------|
| GET     | `/health`  | Sonde de disponibilité ; confirme le modèle chargé.    |
| POST    | `/predict` | Score une demande de crédit ; retourne proba + décision.|

#### `GET /health`

```json
{
  "status": "ok",
  "model_version": "credit-scoring-final-v3-fe",
  "n_features_attendues": 658
}
```

#### `POST /predict`

Le corps accepte les **21 features du noyau** (9 requises, 12 optionnelles)
et, optionnellement, toute autre feature attendue par le modèle (les ~637
agrégations) ; les features absentes sont complétées à `NaN` côté serveur.
Les 9 ratios métier ne sont **pas** à fournir : le pipeline les recalcule.

Champs requis : `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `AMT_INCOME_TOTAL`,
`AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `DAYS_BIRTH`,
`CNT_FAM_MEMBERS`, `DAYS_EMPLOYED`.

Exemple de requête :

```bash
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{"NAME_CONTRACT_TYPE":"Cash loans","CODE_GENDER":"M","AMT_INCOME_TOTAL":180000,"AMT_CREDIT":450000,"AMT_ANNUITY":24700,"AMT_GOODS_PRICE":405000,"DAYS_BIRTH":-14200,"CNT_FAM_MEMBERS":2,"DAYS_EMPLOYED":-2400,"EXT_SOURCE_2":0.62,"EXT_SOURCE_3":0.51}'
```

Réponse :

```json
{
  "proba_defaut": 0.406,
  "decision": "accordé",
  "seuil": 0.49,
  "model_version": "credit-scoring-final-v3-fe"
}
```

`decision` vaut `refusé` si `proba_defaut >= seuil`, sinon `accordé`.
Une entrée invalide (champ requis manquant, valeur hors plage, mauvais
type) renvoie un **422** avec le détail de l'erreur.

## Tests

```bash
pytest -v --cov=app --cov-report=term-missing
```

23 tests (TestClient FastAPI) couvrant les cas nominaux et les cas
d'erreur critiques (champ requis manquant, valeur hors plage, mauvais
type, enum invalide, cohérence décision/seuil). Couverture ~96 %.

## Docker

```bash
# Construire l'image
docker build -t credit-scoring-api .

# Lancer le conteneur
docker run -p 8800:8800 credit-scoring-api
```

L'API est ensuite disponible sur <http://localhost:8800>. L'image installe
`libgomp1` (requis par LightGBM), embarque `features.py` (nécessaire à la
désérialisation du pipeline) et s'exécute en utilisateur non-root.

## Intégration continue (CI/CD)

Le workflow `.github/workflows/ci-cd.yml` se déclenche sur les push et les
pull requests vers `main`. Il enchaîne deux jobs :

1. **`test`** — installe les dépendances et exécute `pytest --cov` ;
2. **`build`** — construit l'image Docker (uniquement si les tests passent).

## Monitoring (Étape 3 — à venir)

Chaque appel à `/predict` est journalisé dans `logs/predictions.jsonl`
(une ligne JSON par prédiction : `timestamp`, `inputs`, `proba_defaut`,
`decision`, `inference_ms`, `latency_ms`). Ce journal alimentera le
dashboard Streamlit et l'analyse de data drift (Evidently).
