"""
app/model.py — Chargement unique du modèle et fonction de scoring.

Le modèle est chargé UNE SEULE FOIS à l'import de ce module (donc au
démarrage de l'API), conformément au point de vigilance de la mission :
« ne pas charger le modèle à chaque requête ». Toutes les requêtes
réutilisent ensuite l'objet `PIPELINE` déjà en mémoire.
"""
from pathlib import Path
import pickle

import pandas as pd

# Le pipeline contient un FunctionTransformer qui référence
# features.add_application_features : ce module DOIT être importable ici,
# sinon la désérialisation échoue. On l'importe explicitement pour échouer
# tôt, avec un message clair, plutôt qu'au milieu du pickle.load.
import features  # noqa: F401

# Chemin de l'artefact, robuste en local comme dans le conteneur Docker
# (app/model.py -> racine du projet -> model/model_credit_scoring.pkl)
_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model_credit_scoring.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _model_data = pickle.load(_f)

# Objets exposés au reste de l'application (chargés une seule fois)
PIPELINE      = _model_data["pipeline"]
INPUT_COLS    = list(_model_data["input_cols"])
SEUIL         = float(_model_data["seuil"])
MODEL_VERSION = _model_data.get("modele_version", "inconnue")
AUC_ROC       = _model_data.get("auc_roc")


def _to_frame(features_dict: dict) -> pd.DataFrame:
    """
    Construit un DataFrame d'UNE ligne aligné EXACTEMENT sur INPUT_COLS :

    - colonnes manquantes      -> NaN (imputées par le pipeline / gérées par LGBM)
    - colonnes inattendues      -> ignorées
    - ordre des colonnes        -> identique à l'entraînement (garanti par reindex)

    C'est ce réalignement qui rend l'API robuste : l'appelant n'envoie que
    les features dont il dispose, le serveur complète le reste.
    """
    df = pd.DataFrame([features_dict])
    return df.reindex(columns=INPUT_COLS)


def predict_one(features_dict: dict) -> dict:
    """
    Score un client unique.

    Renvoie la probabilité de défaut, la décision (au regard du seuil
    métier 0.49) et le seuil utilisé.
    """
    X = _to_frame(features_dict)
    proba = float(PIPELINE.predict_proba(X)[0, 1])
    decision = "refusé" if proba >= SEUIL else "accordé"
    return {
        "proba_defaut": proba,
        "decision": decision,
        "seuil": SEUIL,
        "model_version": MODEL_VERSION,
    }
