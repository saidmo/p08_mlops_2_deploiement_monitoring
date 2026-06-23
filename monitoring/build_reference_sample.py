"""
monitoring/build_reference_sample.py

Construit l'échantillon de RÉFÉRENCE pour la détection de data drift.

- Source : data/p06_df_final.parquet (données d'entraînement du P06, ~280 Mo,
  NON versionnée).
- Sortie : data/reference_sample.parquet — un échantillon stratifié sur TARGET
  de 10 000 lignes, restreint aux 21 features du noyau (celles que l'API
  journalise) + TARGET. Léger (< 1 Mo) et versionné, il sert de baseline à
  Evidently pour comparer la distribution de la production à celle de
  l'entraînement.

Le choix « mêmes colonnes que celles loguées par l'API » garantit l'alignement
référence ↔ production lors de l'analyse de drift.

Usage (depuis la racine du projet, venv activé) :
    python monitoring/build_reference_sample.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Rend le package `app` importable quel que soit le répertoire d'exécution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import ClientFeatures  # noqa: E402  (après l'ajout au sys.path)

SOURCE = ROOT / "data" / "p06_df_final.parquet"
OUTPUT = ROOT / "data" / "reference_sample.parquet"
N_SAMPLE = 10_000
SEED = 42

# Les 21 features du noyau = source de vérité unique (définie dans le schéma)
CORE_COLS = list(ClientFeatures.model_fields.keys())


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source introuvable : {SOURCE}\n"
            "Dépose p06_df_final.parquet dans le dossier data/ à la racine."
        )

    df = pd.read_parquet(SOURCE)

    # On ne garde que les lignes d'entraînement (label TARGET connu)
    df = df[df["TARGET"].notnull()].copy()

    # Colonnes du noyau réellement présentes + TARGET
    cols = [c for c in CORE_COLS if c in df.columns] + ["TARGET"]
    manquantes = [c for c in CORE_COLS if c not in df.columns]
    if manquantes:
        print(f"Colonnes du noyau absentes de la source (ignorées) : {manquantes}")
    df = df[cols]

    # Échantillon stratifié sur TARGET (préserve le déséquilibre de classes)
    sample, _ = train_test_split(
        df,
        train_size=N_SAMPLE,
        stratify=df["TARGET"],
        random_state=SEED,
    )

    OUTPUT.parent.mkdir(exist_ok=True)
    sample.to_parquet(OUTPUT, index=False)

    print(f"Échantillon écrit : {OUTPUT}")
    print(f"   {len(sample)} lignes, {sample.shape[1]} colonnes")
    print("   Répartition TARGET :")
    print(sample["TARGET"].value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
