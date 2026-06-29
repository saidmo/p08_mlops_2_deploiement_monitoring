"""
optimization/profile_inference.py

Profiling de l'inférence pour identifier les goulots d'étranglement avant
toute optimisation (Étape 4).

L'inférence est décomposée en 4 blocs chronométrés séparément :
  1. construction du DataFrame d'entrée (1 ligne, réalignée sur 658 colonnes) ;
  2. feature engineering (FunctionTransformer) ;
  3. prétraitement (ColumnTransformer : imputation + encodage) ;
  4. arbre LightGBM (predict_proba).

Deux profils d'entrée sont comparés :
  • CLIENT COMPLET — 658 features renseignées (vraie ligne de df_final) ;
  • CLIENT NOYAU   — seules les ~11 features du noyau, le reste à NaN.

Un passage cProfile complète l'analyse au niveau fonction.

Usage (venv activé, à la racine du projet) :
    python optimization/profile_inference.py
"""
import warnings

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model import PIPELINE, INPUT_COLS, _to_frame, predict_one  # noqa: E402
from app.schemas import ClientFeatures  # noqa: E402

SOURCE = ROOT / "data" / "p06_df_final.parquet"
OUT_DIR = ROOT / "optimization"
N = 10_000          # itérations pour la mesure des blocs
N_CPROFILE = 2_000  # itérations pour cProfile (overhead plus élevé)
CORE_COLS = list(ClientFeatures.model_fields.keys())


def build_clients() -> tuple[dict, dict]:
    """Construit un client complet (658 features) et un client noyau."""
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source introuvable : {SOURCE}")
    df = pd.read_parquet(SOURCE, columns=INPUT_COLS + ["TARGET"])
    row = df[df["TARGET"].notnull()].iloc[0]

    complet = {c: row[c] for c in INPUT_COLS}
    noyau = {c: row[c] for c in CORE_COLS if c in row.index and pd.notna(row[c])}
    return complet, noyau


def time_blocks(client: dict, n: int) -> dict:
    """Chronomètre les 4 blocs de l'inférence sur n itérations."""
    feat = PIPELINE.named_steps["features"]
    prep = PIPELINE.named_steps["preprocessor"]
    mdl = PIPELINE.named_steps["model"]

    # Warm-up (le premier appel paie le coût d'initialisation)
    X = _to_frame(client)
    mdl.predict_proba(prep.transform(feat.transform(X)))

    t = {"build": 0.0, "feat": 0.0, "prep": 0.0, "tree": 0.0}
    for _ in range(n):
        s = time.perf_counter(); X = _to_frame(client);              t["build"] += time.perf_counter() - s
        s = time.perf_counter(); Xf = feat.transform(X);             t["feat"] += time.perf_counter() - s
        s = time.perf_counter(); Xp = prep.transform(Xf);            t["prep"] += time.perf_counter() - s
        s = time.perf_counter(); _ = mdl.predict_proba(Xp);          t["tree"] += time.perf_counter() - s

    return {k: v / n * 1000 for k, v in t.items()}  # ms moyens par appel


def print_blocs(label: str, blocs: dict) -> None:
    total = sum(blocs.values())
    noms = {
        "build": "1. Construction DataFrame",
        "feat": "2. Feature engineering   ",
        "prep": "3. Prétraitement         ",
        "tree": "4. Arbre LightGBM        ",
    }
    print(f"\n=== {label} ===")
    for k in ("build", "feat", "prep", "tree"):
        ms = blocs[k]
        pct = ms / total * 100 if total else 0
        print(f"  {noms[k]} : {ms:7.3f} ms  ({pct:4.1f} %)")
    print(f"  {'-' * 40}")
    print(f"  Total (somme des blocs)   : {total:7.3f} ms")


def run_cprofile(client: dict, n: int, out_file: Path) -> None:
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        predict_one(client)
    pr.disable()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
    out_file.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\ncProfile ({n} itérations) écrit dans : {out_file.name}")
    # Aperçu des 12 premières lignes
    print("\n".join(buf.getvalue().splitlines()[:18]))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Profiling de l'inférence — {N} itérations par bloc")
    print("(plusieurs minutes : ~30 ms/appel × itérations)")

    complet, noyau = build_clients()

    print_blocs(f"CLIENT COMPLET (658 features) — {N} it.", time_blocks(complet, N))
    print_blocs(f"CLIENT NOYAU ({len(noyau)} features, reste NaN) — {N} it.", time_blocks(noyau, N))

    run_cprofile(complet, N_CPROFILE, OUT_DIR / "profile_cprofile.txt")


if __name__ == "__main__":
    main()
