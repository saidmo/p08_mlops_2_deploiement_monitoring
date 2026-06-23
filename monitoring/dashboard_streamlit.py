"""
monitoring/dashboard_streamlit.py

Dashboard de monitoring de l'API de scoring crédit. Lit le journal des
prédictions (logs/predictions.jsonl) produit par l'API et affiche :

  • des indicateurs clés (volume, latence, temps d'inférence, taux de refus) ;
  • la distribution des scores prédits ;
  • la répartition des décisions accordé / refusé ;
  • la latence et le temps d'inférence au fil des requêtes ;
  • le score de défaut au fil des requêtes (révèle un éventuel drift de score) ;
  • un volet opérationnel (latence p95 / max, taux d'erreur).

Le dashboard est autonome : il ne lit que le journal, ne charge pas le modèle.

Usage (venv activé, à la racine du projet) :
    streamlit run monitoring/dashboard_streamlit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "predictions.jsonl"

st.set_page_config(page_title="Monitoring — Scoring crédit", layout="wide")


def load_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


st.title("📊 Monitoring de l'API de scoring crédit")

col_refresh, _ = st.columns([1, 6])
if col_refresh.button("🔄 Rafraîchir"):
    st.rerun()

df = load_logs(LOG_FILE)

if df.empty:
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
