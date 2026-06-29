"""
optimization/benchmark_optim.py

Banc d'essai des optimisations de LATENCE UNITAIRE (1 client à la fois).

Compare le chemin d'inférence baseline à des variantes optimisées de
serving, et VÉRIFIE la non-régression : les probabilités produites doivent
être identiques au baseline (tolérance 1e-9). On mesure sur le client
« noyau » (cas réaliste : l'API reçoit ~20 features, le reste à NaN).

Variantes testées :
  • baseline                — predict_one actuel
  • assume_finite           — sklearn.config_context(assume_finite=True)
  • assume_finite+template  — + construction du DataFrame par copie d'un
                              template pré-alloué de 658 colonnes

Usage (venv activé, à la racine du projet) :
    python optimization/benchmark_optim.py
"""
import warnings

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model import PIPELINE, INPUT_COLS, SEUIL, _to_frame, predict_one  # noqa: E402
from app.schemas import ClientFeatures  # noqa: E402

SOURCE = ROOT / "data" / "p06_df_final.parquet"
N = 2_000
CORE_COLS = list(ClientFeatures.model_fields.keys())
_COLSET = set(INPUT_COLS)

# Colonnes catégorielles (depuis le .pkl) : elles doivent être de dtype object
# dans le template, sinon pandas refuse d'y écrire une chaîne (LossySetitemError).
import pickle  # noqa: E402
_model_data = pickle.load(open(ROOT / "model" / "model_credit_scoring.pkl", "rb"))
_CAT_COLS = set(_model_data.get("binary_cols", [])) | set(_model_data.get("multi_cols", []))

# Template pré-alloué : une ligne, 658 colonnes, dtypes corrects, construit une fois
_TEMPLATE = pd.DataFrame([{c: np.nan for c in INPUT_COLS}])
for _c in _CAT_COLS:
    if _c in _TEMPLATE.columns:
        _TEMPLATE[_c] = _TEMPLATE[_c].astype("object")


def build_core_client() -> dict:
    df = pd.read_parquet(SOURCE, columns=INPUT_COLS + ["TARGET"])
    row = df[df["TARGET"].notnull()].iloc[0]
    return {c: row[c] for c in CORE_COLS if c in row.index and pd.notna(row[c])}


# --- Variantes ---
def predict_baseline(client: dict) -> float:
    return predict_one(client)["proba_defaut"]


def predict_assume_finite(client: dict) -> float:
    with sklearn.config_context(assume_finite=True):
        X = _to_frame(client)
        return float(PIPELINE.predict_proba(X)[0, 1])


def predict_template(client: dict) -> float:
    with sklearn.config_context(assume_finite=True):
        df = _TEMPLATE.copy()
        for k, v in client.items():
            if k in _COLSET:
                df.at[0, k] = v
        return float(PIPELINE.predict_proba(df)[0, 1])


def bench_interleaved(variantes: dict, client: dict, n: int) -> dict:
    """
    Mesure ENTRELACÉE : à chaque itération, toutes les variantes sont
    exécutées dans un ordre tiré au hasard. Neutralise les dérives lentes
    (throttling thermique sur portable, effets d'ordre) qui biaisent une
    mesure séquentielle « tout A, puis tout B ».
    """
    for fn in variantes.values():
        fn(client)  # warm-up
    temps = {nom: 0.0 for nom in variantes}
    items = list(variantes.items())
    for _ in range(n):
        random.shuffle(items)
        for nom, fn in items:
            s = time.perf_counter()
            fn(client)
            temps[nom] += time.perf_counter() - s
    return {nom: t / n * 1000 for nom, t in temps.items()}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source introuvable : {SOURCE}")
    client = build_core_client()

    variantes = {
        "baseline": predict_baseline,
        "assume_finite": predict_assume_finite,
        "assume_finite+template": predict_template,
    }

    # Non-régression : toutes les variantes doivent renvoyer la proba du baseline
    ref = predict_baseline(client)
    print("Vérification de non-régression (proba de référence = "
          f"{ref:.10f}) :")
    for nom, fn in variantes.items():
        proba = fn(client)
        diff = abs(proba - ref)
        etat = "OK" if diff < 1e-9 else "⚠️ ÉCART"
        print(f"  {nom:24s} : {proba:.10f}  (écart {diff:.2e}) {etat}")

    print(f"\nLatence unitaire moyenne ({N} itérations, mesure entrelacée) :")
    temps = bench_interleaved(variantes, client, N)
    base = temps["baseline"]
    for nom in variantes:
        ms = temps[nom]
        if nom == "baseline":
            print(f"  {nom:24s} : {ms:7.3f} ms")
        else:
            gain = (1 - ms / base) * 100
            print(f"  {nom:24s} : {ms:7.3f} ms  ({gain:+.1f} % vs baseline)")


if __name__ == "__main__":
    main()
