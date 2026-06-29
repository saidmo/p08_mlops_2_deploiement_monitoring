"""
monitoring/simulate_production.py

Simule du trafic de production en envoyant des requêtes à l'API locale
(POST /predict). Chaque appel est journalisé par l'API dans
logs/predictions.jsonl, qui alimente ensuite le dashboard Streamlit et
l'analyse de data drift (Evidently).

Deux vagues sont envoyées :
  1. NORMALE  — vraies lignes clients échantillonnées (référence du
     comportement attendu).
  2. DÉRIVÉE  — mêmes lignes mais perturbées (scénario « récession » :
     revenus abaissés, crédits/annuités relevés, scores externes
     diminués). Provoque un data drift volontaire et détectable.

Prérequis : l'API doit tourner en parallèle, par ex. :
    uvicorn app.main:app --port 8800

Usage (dans un second terminal, venv activé, à la racine du projet) :
    python monitoring/simulate_production.py
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import ClientFeatures  # noqa: E402

# URL de l'API : configurable pour pouvoir amorcer une instance distante
# (ex. un Space Hugging Face) avec le même script. Défaut = API locale.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8800").rstrip("/")
SOURCE = ROOT / "data" / "p06_df_final.parquet"
N_PER_WAVE = 500
SEED = 42

# Paramètres du split de référence — DOIVENT être identiques à ceux de
# build_reference_sample.py, pour reproduire la même partition et donc
# exclure de la production les lignes utilisées comme référence.
REF_SIZE = 10_000
REF_SEED = 42

# Opener urllib qui ignore tout proxy système (appels vers l'API locale)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Colonnes du noyau (source de vérité = le schéma) et sous-ensemble requis
CORE_COLS = list(ClientFeatures.model_fields.keys())
REQUIRED_COLS = [n for n, f in ClientFeatures.model_fields.items() if f.is_required()]

# Perturbations de la vague dérivée (facteurs multiplicatifs)
DRIFT = {
    "AMT_INCOME_TOTAL": 0.6,   # revenus en baisse
    "AMT_CREDIT": 1.4,         # crédits plus élevés
    "AMT_ANNUITY": 1.2,        # annuités plus lourdes
    "EXT_SOURCE_1": 0.5,       # scores externes dégradés
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
}


def _post(path: str, payload: dict | None = None):
    """POST/GET JSON via urllib. Retourne (status, body) ; lève sur réseau.

    Utilise un opener SANS proxy : sur une machine de domaine, le proxy
    système intercepterait sinon les appels vers l'API locale.
    """
    url = f"{API_URL}{path}"
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
    try:
        with _OPENER.open(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def _row_to_payload(row: pd.Series) -> dict:
    """Construit un payload JSON à partir d'une ligne, en excluant les NaN."""
    payload = {}
    for col in CORE_COLS:
        val = row[col]
        if pd.isna(val):
            continue
        payload[col] = val.item() if hasattr(val, "item") else val
    return payload


def _send_wave(df: pd.DataFrame, label: str) -> None:
    ok, erreurs, probas = 0, 0, []
    for _, row in df.iterrows():
        status, body = _post("/predict", _row_to_payload(row))
        if status == 200:
            ok += 1
            probas.append(body["proba_defaut"])
        else:
            erreurs += 1
    moy = sum(probas) / len(probas) if probas else float("nan")
    print(f"  Vague {label:8s} : {ok} OK, {erreurs} erreurs | proba moyenne = {moy:.3f}")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source introuvable : {SOURCE}")

    # Vérifie que l'API répond
    try:
        status, _ = _post("/health")
        if status != 200:
            raise RuntimeError
    except Exception:
        sys.exit(
            f"❌ L'API ne répond pas sur {API_URL}.\n"
            "   • En local : lance d'abord  uvicorn app.main:app --port 8800\n"
            "   • Pour une API distante : définis API_URL avant de lancer, ex.\n"
            '     PowerShell :  $env:API_URL = "https://...hf.space"\n'
            "     cmd        :  set API_URL=https://...hf.space"
        )

    df = pd.read_parquet(SOURCE, columns=CORE_COLS + ["TARGET"])
    df = df[df["TARGET"].notnull()]

    # --- Exclusion des lignes de la référence ---
    # On reproduit EXACTEMENT le split de build_reference_sample.py pour
    # isoler les lignes de référence, puis on tire la production dans le
    # COMPLÉMENT => aucun recouvrement référence ↔ production.
    _reference, prod_pool = train_test_split(
        df, train_size=REF_SIZE, stratify=df["TARGET"], random_state=REF_SEED
    )

    # On ne garde que les lignes dont tous les champs requis sont renseignés
    prod_pool = prod_pool.dropna(subset=REQUIRED_COLS)

    # Deux échantillons disjoints tirés du complément de la référence
    besoin = 2 * N_PER_WAVE
    tirage = prod_pool.sample(n=min(besoin, len(prod_pool)), random_state=SEED).reset_index(drop=True)
    wave_normale = tirage.iloc[:N_PER_WAVE]
    wave_drift = tirage.iloc[N_PER_WAVE:2 * N_PER_WAVE].copy()

    # Applique les perturbations de drift
    for col, facteur in DRIFT.items():
        if col in wave_drift.columns:
            wave_drift[col] = wave_drift[col] * facteur
    # Re-borne les scores externes dans [0, 1]
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if col in wave_drift.columns:
            wave_drift[col] = wave_drift[col].clip(0, 1)

    print(f"Envoi de {N_PER_WAVE} requêtes par vague vers {API_URL} ...")
    _send_wave(wave_normale, "NORMALE")
    _send_wave(wave_drift, "DÉRIVÉE")
    print("✅ Terminé. Voir logs/predictions.jsonl")


if __name__ == "__main__":
    main()
