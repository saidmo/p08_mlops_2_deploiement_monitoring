"""
monitoring/dashboard_streamlit.py

Dashboard de monitoring de l'API de scoring crédit. Affiche :

  • des indicateurs clés (volume, latence, temps d'inférence, taux de refus) ;
  • la distribution des scores prédits ;
  • la répartition des décisions accordé / refusé ;
  • la latence et le temps d'inférence au fil des requêtes ;
  • le score de défaut au fil des requêtes (révèle un éventuel drift de score) ;
  • un volet opérationnel (latence p95 / max, taux d'erreur).

Source des données — deux modes, selon la variable d'environnement API_URL :
  • API_URL définie  -> appelle l'endpoint {API_URL}/logs (mode distant,
                        utilisé en ligne : le dashboard et l'API sont deux
                        services séparés) ;
  • API_URL absente  -> lit le fichier local logs/predictions.jsonl
                        (mode développement, sans API à interroger).

Le dashboard est autonome : il ne charge jamais le modèle.

Usage (venv activé, à la racine du projet) :
    # mode fichier local
    streamlit run monitoring/dashboard_streamlit.py
    # mode API distante (exemple en local)
    API_URL=http://127.0.0.1:8800 streamlit run monitoring/dashboard_streamlit.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "predictions.jsonl"

# Source des données : API distante si API_URL est définie, sinon fichier local.
API_URL = os.environ.get("API_URL", "").rstrip("/")
LOGS_LIMIT = int(os.environ.get("LOGS_LIMIT", "2000"))

# Mot de passe d'accès (facultatif) : fourni via la variable d'environnement
# DASHBOARD_PASSWORD (definie en "secret" du Space sur Hugging Face). Si elle
# est absente, le dashboard reste librement accessible (mode developpement).
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")

st.set_page_config(page_title="Monitoring — Scoring crédit", layout="wide")


def exiger_mot_de_passe() -> None:
    """Porte d'acces : si DASHBOARD_PASSWORD est definie, exige le bon mot de
    passe avant d'afficher quoi que ce soit. Sans cette variable, ne fait rien.

    Protection de niveau demonstration (le code du Space est public) : convient
    pour reserver l'acces a l'evaluateur, pas pour proteger un secret critique.
    """
    if not DASHBOARD_PASSWORD:
        return  # pas de mot de passe configure -> acces libre

    if st.session_state.get("authentifie"):
        return  # deja valide pour cette session

    st.title("🔒 Monitoring — accès protégé")
    saisie = st.text_input("Mot de passe", type="password")
    if saisie:
        if saisie == DASHBOARD_PASSWORD:
            st.session_state["authentifie"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()  # bloque tout le reste du dashboard tant que non authentifie


def _normaliser(df: pd.DataFrame) -> pd.DataFrame:
    """Tri chronologique commun aux deux sources."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_logs_fichier(path: Path) -> pd.DataFrame:
    """Mode développement : lit le journal local."""
    if not path.exists():
        return pd.DataFrame()
    return _normaliser(pd.read_json(path, lines=True))


def load_logs_api(base_url: str, limit: int) -> pd.DataFrame:
    """Mode distant : interroge GET {base_url}/logs."""
    r = requests.get(f"{base_url}/logs", params={"limit": limit}, timeout=10)
    r.raise_for_status()
    predictions = r.json().get("predictions", [])
    if not predictions:
        return pd.DataFrame()
    return _normaliser(pd.DataFrame(predictions))


def load_logs() -> pd.DataFrame:
    """Charge les données depuis la source configurée (API ou fichier)."""
    if API_URL:
        return load_logs_api(API_URL, LOGS_LIMIT)
    return load_logs_fichier(LOG_FILE)


exiger_mot_de_passe()

st.title("📊 Monitoring de l'API de scoring crédit")

source_label = f"API distante ({API_URL})" if API_URL else "fichier local"
col_refresh, col_src = st.columns([1, 6])
if col_refresh.button("🔄 Rafraîchir"):
    st.rerun()
col_src.caption(f"Source des données : **{source_label}**")

try:
    df = load_logs()
except requests.exceptions.RequestException as e:
    st.error(
        f"Impossible de joindre l'API à `{API_URL}`.\n\n"
        f"Détail : {e}\n\n"
        "Vérifie que l'API est démarrée et accessible, puis rafraîchis."
    )
    st.stop()

if df.empty:
    if API_URL:
        st.warning(
            "L'API est joignable mais n'a encore journalisé aucune prédiction.\n\n"
            "Envoie du trafic vers l'API (puis rafraîchis) :\n"
            f"`API_URL={API_URL} python monitoring/simulate_production.py`"
        )
    else:
        st.warning(
            "Aucune prédiction journalisée pour l'instant.\n\n"
            "Lance l'API puis la simulation :\n"
            "1. `uvicorn app.main:app --port 8800`\n"
            "2. `python monitoring/simulate_production.py`"
        )
    st.stop()

seuil = float(df["seuil"].iloc[0]) if "seuil" in df.columns else 0.49
version = df["model_version"].iloc[0] if "model_version" in df.columns else "inconnue"
st.caption(
    f"Modèle **{version}** · seuil métier **{seuil:.2f}** · "
    f"période : {df['timestamp'].min():%Y-%m-%d %H:%M} → {df['timestamp'].max():%Y-%m-%d %H:%M}"
)

# ---------------------------------------------------------------------------
# Indicateurs clés
# ---------------------------------------------------------------------------
total = len(df)
taux_refus = (df["decision"] == "refusé").mean() * 100
lat_med = df["latency_ms"].median()
inf_med = df["inference_ms"].median()
proba_moy = df["proba_defaut"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Prédictions", f"{total}")
k2.metric("Taux de refus", f"{taux_refus:.1f} %")
k3.metric("Latence médiane", f"{lat_med:.1f} ms")
k4.metric("Inférence médiane", f"{inf_med:.1f} ms")
k5.metric("Proba moyenne", f"{proba_moy:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Distribution des scores + décisions
# ---------------------------------------------------------------------------
g1, g2 = st.columns(2)

with g1:
    st.subheader("Distribution des scores prédits")
    counts, edges = np.histogram(df["proba_defaut"], bins=25, range=(0, 1))
    hist = pd.DataFrame(
        {"proba_defaut": np.round((edges[:-1] + edges[1:]) / 2, 3), "effectif": counts}
    ).set_index("proba_defaut")
    st.bar_chart(hist)
    st.caption(f"Seuil de décision = {seuil:.2f} : au-delà, la demande est refusée.")

with g2:
    st.subheader("Répartition des décisions")
    decisions = df["decision"].value_counts().rename_axis("décision").to_frame("effectif")
    st.bar_chart(decisions)

st.divider()

# ---------------------------------------------------------------------------
# Séries temporelles (au fil des requêtes)
# ---------------------------------------------------------------------------
st.subheader("Latence et temps d'inférence au fil des requêtes")
st.line_chart(df[["latency_ms", "inference_ms"]])
st.caption(
    "Le pic initial correspond au « réveil » du modèle (première prédiction) ; "
    "la latence se stabilise ensuite à quelques millisecondes."
)

st.subheader("Score de défaut au fil des requêtes")
st.line_chart(df["proba_defaut"])
st.caption(
    "Une hausse durable du score en seconde moitié traduit le scénario de "
    "drift simulé (revenus en baisse, crédits relevés, scores externes dégradés)."
)

st.divider()

# ---------------------------------------------------------------------------
# Volet opérationnel
# ---------------------------------------------------------------------------
st.subheader("Indicateurs opérationnels")
lat_p95 = df["latency_ms"].quantile(0.95)
lat_max = df["latency_ms"].max()
if "http_status" in df.columns:
    taux_erreur = (df["http_status"] != 200).mean() * 100
else:
    taux_erreur = 0.0

o1, o2, o3 = st.columns(3)
o1.metric("Latence p95", f"{lat_p95:.1f} ms")
o2.metric("Latence max", f"{lat_max:.1f} ms")
o3.metric("Taux d'erreur", f"{taux_erreur:.1f} %")

st.caption(
    "Note : seules les prédictions abouties (HTTP 200) sont journalisées. "
    "Les requêtes invalides sont rejetées en amont (422) par la validation "
    "Pydantic et n'apparaissent donc pas ici ; capturer ces erreurs "
    "nécessiterait une journalisation dédiée dans un gestionnaire d'exception."
)
