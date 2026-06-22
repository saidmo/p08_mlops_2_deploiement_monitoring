"""
tests/test_api.py — Tests unitaires de l'API de scoring.

Utilisent le TestClient de FastAPI : l'application est exercée en mémoire,
sans serveur à démarrer, ce qui les rend rapides et exécutables en CI.

Les cas couverts répondent aux exigences de la mission (gestion des erreurs,
cas critiques) : champ requis manquant, valeur hors plage, mauvais type,
enum invalide, plus la cohérence décision/seuil.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Demande de crédit VALIDE : les 9 champs requis + 2 scores externes.
# N'utilise que des features du noyau, donc aucun fichier de données externe
# n'est nécessaire (les ~637 agrégations sont complétées à NaN côté serveur).
VALID_PAYLOAD = {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 450000,
    "AMT_ANNUITY": 24700,
    "AMT_GOODS_PRICE": 405000,
    "DAYS_BIRTH": -14200,
    "CNT_FAM_MEMBERS": 2,
    "DAYS_EMPLOYED": -2400,
    "EXT_SOURCE_2": 0.62,
    "EXT_SOURCE_3": 0.51,
}


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features_attendues"] > 0
    assert isinstance(body["model_version"], str)


# --------------------------------------------------------------------------
# /predict — cas nominal
# --------------------------------------------------------------------------
def test_predict_nominal():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["proba_defaut"] <= 1.0
    assert body["decision"] in {"accordé", "refusé"}
    assert body["seuil"] == pytest.approx(0.49)
    assert isinstance(body["model_version"], str)


def test_predict_decision_coherente_avec_seuil():
    """La décision doit découler de la comparaison proba vs seuil."""
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    attendu = "refusé" if body["proba_defaut"] >= body["seuil"] else "accordé"
    assert body["decision"] == attendu


def test_predict_header_latence_present():
    """Le middleware doit exposer la latence end-to-end en header."""
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert "X-Process-Time-Ms" in r.headers
    assert float(r.headers["X-Process-Time-Ms"]) >= 0


# --------------------------------------------------------------------------
# /predict — cas d'erreur (doivent tous renvoyer 422)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "modif, description",
    [
        ({"AMT_INCOME_TOTAL": 0}, "revenu = 0 (hors plage)"),
        ({"AMT_ANNUITY": -100}, "annuité négative (hors plage)"),
        ({"AMT_CREDIT": 0}, "crédit = 0 (hors plage)"),
        ({"EXT_SOURCE_2": 1.5}, "score externe > 1 (hors plage)"),
        ({"EXT_SOURCE_1": -0.2}, "score externe < 0 (hors plage)"),
        ({"DAYS_BIRTH": 1825}, "âge positif / impossible (hors plage)"),
        ({"CNT_FAM_MEMBERS": 0}, "foyer = 0 (hors plage)"),
        ({"CODE_GENDER": "X"}, "enum genre invalide"),
        ({"NAME_CONTRACT_TYPE": "Prêt magique"}, "enum contrat invalide"),
        ({"AMT_CREDIT": "beaucoup"}, "mauvais type (texte au lieu de nombre)"),
    ],
)
def test_predict_entrees_invalides(modif, description):
    payload = {**VALID_PAYLOAD, **modif}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"Devrait être rejeté : {description}"


@pytest.mark.parametrize("champ_requis", list(VALID_PAYLOAD.keys())[:9])
def test_predict_champ_requis_absent(champ_requis):
    """Retirer l'un des champs requis doit déclencher un 422."""
    # On ne teste l'absence que sur les champs réellement requis
    requis = {
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "CNT_FAM_MEMBERS",
        "DAYS_EMPLOYED",
    }
    if champ_requis not in requis:
        pytest.skip("champ optionnel, absence autorisée")
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != champ_requis}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
