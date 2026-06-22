"""
app/schemas.py — Contrat d'entrée/sortie de l'API (validation Pydantic v2).

Stratégie face aux 658 features attendues par le modèle :

  • NOYAU VALIDÉ : 21 features brutes « métier » (9 requises, 12
    optionnelles) explicitement typées et contraintes. Ce sont les
    features les plus contributives du modèle (cf. feature importance)
    et celles qu'une demande de crédit contient réellement. Les 7
    colonnes de base des ratios figurent toutes parmi les 9 requises.
    Elles portent la validation exigée par la mission (champs
    obligatoires, valeurs hors plage, types).

  • EXTRAS AUTORISÉS (`extra="allow"`) : les ~640 features d'agrégation
    (BURO_*, PREV_*, CC_*, INSTAL_*, POS_*…) sont acceptées sans être
    déclarées. Le serveur les réaligne ensuite sur input_cols ; toute
    feature absente est complétée à NaN (gérée nativement par LightGBM).

  • Les 9 ratios métier (PAYMENT_RATE, CREDIT_TERM…) ne sont PAS dans le
    contrat : ils sont recalculés par le pipeline. L'appelant fournit les
    colonnes de base, pas les ratios.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClientFeatures(BaseModel):
    """Données d'une demande de crédit soumise à l'API."""

    model_config = ConfigDict(
        extra="allow",  # accepte les ~640 features d'agrégation non déclarées
        json_schema_extra={
            "example": {
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "AMT_INCOME_TOTAL": 180000.0,
                "AMT_CREDIT": 450000.0,
                "AMT_ANNUITY": 24700.0,
                "AMT_GOODS_PRICE": 405000.0,
                "DAYS_BIRTH": -14200,
                "CNT_FAM_MEMBERS": 2.0,
                "EXT_SOURCE_2": 0.62,
                "EXT_SOURCE_3": 0.51,
                "DAYS_EMPLOYED": -2400,
                "OWN_CAR_AGE": 8.0,
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Married",
                "CNT_CHILDREN": 0,
            }
        },
    )

    # ----- Champs REQUIS (validés) -----
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"]
    CODE_GENDER: Literal["M", "F", "XNA"]
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Revenu total (> 0)")
    AMT_CREDIT: float = Field(..., gt=0, description="Montant du crédit (> 0)")
    AMT_ANNUITY: float = Field(..., gt=0, description="Annuité (> 0)")
    AMT_GOODS_PRICE: float = Field(..., ge=0, description="Prix du bien (>= 0)")
    DAYS_BIRTH: int = Field(
        ...,
        ge=-40000,
        le=-6570,
        description="Âge en jours négatifs (≈ -6570 = 18 ans, ≈ -40000 = 109 ans)",
    )
    CNT_FAM_MEMBERS: float = Field(..., ge=1, description="Taille du foyer (>= 1)")
    # Requis pour le calcul des ratios DAYS_EMPLOYED_PERC / EMPLOYED_YEARS.
    # NON contraint en plage : la sentinelle 365243 (sans emploi) est légitime.
    DAYS_EMPLOYED: float = Field(..., description="Ancienneté en jours négatifs (365243 = sans emploi)")

    # ----- Champs OPTIONNELS (validés s'ils sont fournis) -----
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1, description="Score externe 1 [0,1]")
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1, description="Score externe 2 [0,1]")
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1, description="Score externe 3 [0,1]")
    CNT_CHILDREN: int | None = Field(None, ge=0, description="Nombre d'enfants (>= 0)")
    OWN_CAR_AGE: float | None = Field(None, ge=0, description="Âge du véhicule (>= 0)")
    # Champs en jours-avant-demande : valeurs typiquement <= 0.
    DAYS_ID_PUBLISH: float | None = None
    DAYS_REGISTRATION: float | None = None
    # Champs catégoriels descriptifs (laissés libres : encodés par le pipeline)
    FLAG_OWN_CAR: Literal["Y", "N"] | None = None
    FLAG_OWN_REALTY: Literal["Y", "N"] | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_INCOME_TYPE: str | None = None


class PredictionResponse(BaseModel):
    """Réponse renvoyée par l'endpoint /predict."""

    proba_defaut: float = Field(..., description="Probabilité de défaut [0,1]")
    decision: str = Field(..., description="'accordé' ou 'refusé' selon le seuil")
    seuil: float = Field(..., description="Seuil métier appliqué")
    model_version: str = Field(..., description="Version du modèle ayant produit le score")


class HealthResponse(BaseModel):
    """Réponse de l'endpoint /health."""

    status: str
    model_version: str
    n_features_attendues: int
