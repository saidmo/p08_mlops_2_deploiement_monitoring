"""
features.py — Feature engineering ligne-à-ligne pour le scoring crédit.

⚠️  MODULE PARTAGÉ entre l'ENTRAÎNEMENT (notebook P06) et le SERVING (API P08).

FunctionTransformer ne pickle PAS le code de la fonction : il pickle une
référence par nom qualifié, ici `features.add_application_features`.
Au chargement du modèle, Python réimporte le module `features` et y cherche
la fonction. Conséquences impératives :

  • Ce fichier doit être importable sous le MÊME nom de module — `features` —
    dans les deux environnements :
        - P06 : placer features.py à côté du notebook d'export
                puis  `from features import add_application_features`
        - P08 : placer features.py à la RACINE du dépôt (copié dans l'image
                Docker) puis  `from features import add_application_features`
  • Les deux copies doivent rester STRICTEMENT identiques : toute divergence
    introduirait un skew train/serving silencieux.

Dépendances volontairement minimales (pandas seul) : déterministe, léger,
auditable.
"""
import pandas as pd

# Les 7 colonnes de base nécessaires au calcul des ratios.
# C'est le contrat d'entrée minimal pour ces features.
BASE_COLS = [
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "CNT_FAM_MEMBERS",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]

# Les 9 ratios produits par la fonction (dérivés -> calculés par le pipeline,
# l'API n'a donc PAS à les recevoir de l'appelant).
DERIVED_COLS = [
    "DAYS_EMPLOYED_PERC",
    "INCOME_CREDIT_PERC",
    "INCOME_PER_PERSON",
    "ANNUITY_INCOME_PERC",
    "PAYMENT_RATE",
    "CREDIT_TERM",
    "GOODS_CREDIT_RATIO",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
]


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute 9 ratios métier ligne-à-ligne à la table application.

    Sans effet de bord : travaille sur une copie. Idempotent : si les colonnes
    dérivées existent déjà, elles sont recalculées (mêmes valeurs).
    Formules identiques au notebook 01 du P06 (aucun skew train/serving).
    """
    df = df.copy()
    df["DAYS_EMPLOYED_PERC"]  = df["DAYS_EMPLOYED"]    / (df["DAYS_BIRTH"]       + 1e-6)
    df["INCOME_CREDIT_PERC"]  = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"]       + 1e-6)
    df["INCOME_PER_PERSON"]   = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"]  + 1)
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"]      / (df["AMT_INCOME_TOTAL"] + 1e-6)
    df["PAYMENT_RATE"]        = df["AMT_ANNUITY"]      / (df["AMT_CREDIT"]       + 1e-6)
    df["CREDIT_TERM"]         = df["AMT_CREDIT"]       / (df["AMT_ANNUITY"]      + 1e-6)
    df["GOODS_CREDIT_RATIO"]  = df["AMT_GOODS_PRICE"]  / (df["AMT_CREDIT"]       + 1e-6)
    df["AGE_YEARS"]           = df["DAYS_BIRTH"]       / -365
    df["EMPLOYED_YEARS"]      = df["DAYS_EMPLOYED"]    / -365
    return df
