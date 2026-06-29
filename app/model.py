"""
app/model.py — Chargement unique du modèle et fonction de scoring.

Le modèle est chargé UNE SEULE FOIS à l'import de ce module (donc au
démarrage de l'API), conformément au point de vigilance de la mission :
« ne pas charger le modèle à chaque requête ». Toutes les requêtes
réutilisent ensuite l'objet `PIPELINE` déjà en mémoire.
"""
from pathlib import Path
import pickle

import numpy as np
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

# --- Optimisation de la latence unitaire (Étape 4) -------------------------
# Le profiling a montré que ~30 % du temps d'une prédiction unitaire partait
# dans la CONSTRUCTION du DataFrame (pd.DataFrame([...]).reindex sur 658
# colonnes : pandas valide et convertit chaque colonne). On évite ce coût en
# pré-allouant UN template (une ligne, 658 colonnes, dtypes corrects) construit
# une seule fois ici, puis copié et rempli à chaque requête.
#
# Gain mesuré : ~6 % (x86) à ~10 % (Apple Silicon), SANS aucune régression
# (probas strictement identiques à l'ancienne construction). Voir
# optimization/RAPPORT_OPTIMISATION.md.
_COLSET = set(INPUT_COLS)
# Colonnes catégorielles -> dtype object obligatoire dans le template, sinon
# pandas refuse d'y écrire une chaîne (colonnes float par défaut).
_CAT_COLS = set(_model_data.get("binary_cols", [])) | set(_model_data.get("multi_cols", []))

_TEMPLATE = pd.DataFrame([{c: np.nan for c in INPUT_COLS}])
for _c in _CAT_COLS:
    if _c in _TEMPLATE.columns:
        _TEMPLATE[_c] = _TEMPLATE[_c].astype("object")


def _to_frame(features_dict: dict) -> pd.DataFrame:
    """
    Construit un DataFrame d'UNE ligne aligné EXACTEMENT sur INPUT_COLS :

    - colonnes manquantes  -> NaN (imputées par le pipeline / gérées par LGBM)
    - colonnes inattendues  -> ignorées
    - ordre des colonnes    -> identique à l'entraînement

    Implémentation optimisée : on copie le template pré-alloué (dtypes déjà
    corrects) et on n'écrit que les valeurs fournies, au lieu de reconstruire
    un DataFrame de 658 colonnes à chaque appel. La copie par requête garantit
    l'absence d'effet de bord entre requêtes concurrentes (thread-safe).
    """
    df = _TEMPLATE.copy()
    for cle, valeur in features_dict.items():
        if cle in _COLSET:
            df.at[0, cle] = valeur
    return df


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
